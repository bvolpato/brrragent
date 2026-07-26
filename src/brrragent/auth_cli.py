"""Create isolated Codex credentials for brrragent."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from brrragent.codex_oauth import CLIENT_ID, _auth_path, _codex_cli_auth_path

ISSUER = "https://auth.openai.com"
DEVICE_URL = f"{ISSUER}/codex/device"
DEVICE_CODE_URL = f"{ISSUER}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{ISSUER}/api/accounts/deviceauth/token"
TOKEN_URL = f"{ISSUER}/oauth/token"
DEVICE_REDIRECT_URI = f"{ISSUER}/deviceauth/callback"
USER_AGENT = "brrragent-auth"


def _request_json(
    url: str,
    *,
    json_body: dict[str, str] | None = None,
    form_body: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    else:
        data = parse.urlencode(form_body or {}).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=60) as response:
            value = json.loads(response.read().decode())
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{url} failed: HTTP {exc.code}; {detail}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected response from {url}")
    return value


def _request_device_code() -> dict[str, Any]:
    return _request_json(DEVICE_CODE_URL, json_body={"client_id": CLIENT_ID})


def _poll_device_authorization(device: dict[str, Any], timeout: int) -> dict[str, Any]:
    device_auth_id = str(device.get("device_auth_id") or "")
    user_code = str(device.get("user_code") or "")
    if not device_auth_id or not user_code:
        raise RuntimeError(f"Unexpected device response: {device!r}")

    interval = max(int(device.get("interval") or 5), 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = json.dumps(
            {"device_auth_id": device_auth_id, "user_code": user_code}
        ).encode()
        req = request.Request(
            DEVICE_TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                value = json.loads(response.read().decode())
        except error.HTTPError as exc:
            if exc.code in {403, 404}:
                time.sleep(interval + 3)
                continue
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(
                f"{DEVICE_TOKEN_URL} failed: HTTP {exc.code}; {detail}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{DEVICE_TOKEN_URL} failed: {exc}") from exc

        if (
            isinstance(value, dict)
            and value.get("authorization_code")
            and value.get("code_verifier")
        ):
            return value
        raise RuntimeError(f"Unexpected authorization response: {value!r}")
    raise TimeoutError(f"Device authorization timed out after {timeout}s")


def _exchange_tokens(authorization: dict[str, Any]) -> dict[str, Any]:
    return _request_json(
        TOKEN_URL,
        form_body={
            "grant_type": "authorization_code",
            "code": str(authorization["authorization_code"]),
            "redirect_uri": DEVICE_REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": str(authorization["code_verifier"]),
        },
    )


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(payload).decode())
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _extract_account_id(tokens: dict[str, Any]) -> str | None:
    for token_name in ("id_token", "access_token"):
        claims = _decode_jwt_payload(str(tokens.get(token_name) or ""))
        auth_claims = claims.get("https://api.openai.com/auth") or {}
        organizations = claims.get("organizations") or []
        first_organization = (
            organizations[0]
            if isinstance(organizations, list)
            and organizations
            and isinstance(organizations[0], dict)
            else {}
        )
        account_id = (
            claims.get("chatgpt_account_id")
            or (
                auth_claims.get("chatgpt_account_id")
                if isinstance(auth_claims, dict)
                else None
            )
            or first_organization.get("id")
        )
        if account_id:
            return str(account_id)
    return None


def _auth_documents(tokens: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    id_token = str(tokens.get("id_token") or "")
    access_token = str(tokens.get("access_token") or "")
    refresh_token = str(tokens.get("refresh_token") or "")
    account_id = _extract_account_id(tokens)
    if not id_token or not access_token or not refresh_token or not account_id:
        raise RuntimeError("Token exchange returned incomplete Codex credentials")

    expires_ms = int(time.time() * 1000) + int(tokens.get("expires_in") or 3600) * 1000
    direct_auth = {
        "openai": {
            "type": "oauth",
            "access": access_token,
            "refresh": refresh_token,
            "expires": expires_ms,
            "accountId": account_id,
        }
    }
    cli_auth = {
        "OPENAI_API_KEY": None,
        "auth_mode": "chatgpt",
        "tokens": {
            "id_token": id_token,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return direct_auth, cli_auth


def _stage_auth(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
        pending.chmod(0o600)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return pending


def _write_auth_documents(
    direct_path: Path,
    cli_path: Path,
    direct_auth: dict[str, Any],
    cli_auth: dict[str, Any],
) -> None:
    if direct_path.resolve() == cli_path.resolve():
        raise ValueError("Direct and Codex CLI auth paths must differ")

    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_auth(direct_path, direct_auth), direct_path))
        staged.append((_stage_auth(cli_path, cli_auth), cli_path))
        for pending, target in staged:
            os.replace(pending, target)
    finally:
        for pending, _target in staged:
            pending.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create isolated brrragent Codex OAuth credentials."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    login = subparsers.add_parser("login", help="Authenticate with Codex device login")
    login.add_argument("--auth-file", type=Path, help="Direct OAuth output path")
    login.add_argument("--codex-home", type=Path, help="Codex CLI home directory")
    login.add_argument(
        "--timeout", type=int, default=900, help="Authorization timeout in seconds"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    direct_path = (args.auth_file or _auth_path()).expanduser()
    cli_path = (
        args.codex_home.expanduser() / "auth.json"
        if args.codex_home
        else _codex_cli_auth_path()
    )
    if direct_path.resolve() == cli_path.resolve():
        parser.error("--auth-file must differ from Codex CLI auth path")

    device = _request_device_code()
    user_code = str(device.get("user_code") or "")
    if not user_code:
        raise RuntimeError(f"Unexpected device response: {device!r}")
    print(f"Open: {DEVICE_URL}")
    print(f"Code: {user_code}")
    print("Waiting for authorization...", flush=True)

    authorization = _poll_device_authorization(device, args.timeout)
    tokens = _exchange_tokens(authorization)
    direct_auth, cli_auth = _auth_documents(tokens)
    _write_auth_documents(direct_path, cli_path, direct_auth, cli_auth)
    print(f"Saved brrragent Codex auth: {direct_path}")
    print(f"Saved isolated Codex CLI auth: {cli_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
