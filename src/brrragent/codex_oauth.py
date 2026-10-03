"""Codex OAuth backend using brrragent-managed credentials."""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from urllib import error, parse, request

from brrragent.images import ImageInput, _responses_user_content
from brrragent.openai_direct import _to_responses_tool, parse_openai_model
from brrragent.prompt_cache import AgentUsage, PromptCacheConfig, openai_usage

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_AUTH_PATH = "~/.cache/brrragent/codex-auth.json"
DEFAULT_CODEX_HOME = "~/.cache/brrragent/codex-home"
DEFAULT_PROFILE_ROOT = "~/.cache/brrragent/profiles"
PROFILE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "temporarily unavailable",
)


@dataclass(frozen=True)
class CodexAuthConfig:
    """Per-call Codex credential selection."""

    profile: str | None = None
    auth_path: str | Path | None = None
    codex_home: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.profile and self.auth_path is None and self.codex_home is None:
            raise ValueError("Codex auth config requires a profile or explicit paths")
        if self.profile:
            _validate_profile_name(self.profile)
            if self.auth_path is not None or self.codex_home is not None:
                raise ValueError("Codex auth profile cannot be combined with paths")
        elif (self.auth_path is None) != (self.codex_home is None):
            raise ValueError("Codex auth_path and codex_home must be provided together")
        elif self.auth_path is not None and (
            Path(self.auth_path).expanduser().resolve()
            == (Path(self.codex_home).expanduser() / "auth.json").resolve()
        ):
            raise ValueError("Codex direct and CLI auth paths must differ")


_ACTIVE_AUTH_CONFIG: ContextVar[CodexAuthConfig | None] = ContextVar(
    "brrragent_codex_auth_config", default=None
)


@contextmanager
def codex_auth_context(config: CodexAuthConfig | None) -> Iterator[None]:
    token = _ACTIVE_AUTH_CONFIG.set(config)
    try:
        yield
    finally:
        _ACTIVE_AUTH_CONFIG.reset(token)


def parse_codex_model(model: str) -> tuple[str, str | None]:
    if model.startswith("codex/"):
        return parse_openai_model("openai/" + model[len("codex/") :])
    return parse_openai_model(model)


def _auth_path() -> Path:
    active = _ACTIVE_AUTH_CONFIG.get()
    if active:
        if active.profile:
            return _profile_paths(active.profile)[0]
        if active.auth_path is not None:
            return Path(active.auth_path).expanduser()
    configured = (
        os.getenv("BRRRAGENT_AUTH_PATH", "").strip()
        or os.getenv("BRRRAGENT_OPENCODE_AUTH_PATH", "").strip()
    )
    if configured:
        return Path(configured).expanduser()
    profile = _configured_profile()
    if profile:
        return _profile_paths(profile)[0]
    return Path(DEFAULT_AUTH_PATH).expanduser()


def _codex_timeout_seconds() -> int:
    raw = os.getenv("BRRRAGENT_CODEX_TIMEOUT_SECONDS") or os.getenv(
        "AI_CLIENT_TIMEOUT_SECONDS"
    )
    try:
        return max(30, int(raw or "900"))
    except ValueError:
        return 900


def _auth_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.brrragent.lock")


@contextmanager
def _exclusive_path_lock(path: Path) -> Iterator[None]:
    lock_path = _auth_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _validate_profile_name(profile: str) -> str:
    profile = profile.strip()
    if not PROFILE_NAME_PATTERN.fullmatch(profile):
        raise ValueError(
            "Profile must contain 1-64 letters, numbers, dots, underscores, or hyphens"
        )
    return profile


def _configured_profile() -> str | None:
    profile = os.getenv("BRRRAGENT_PROFILE", "").strip()
    return _validate_profile_name(profile) if profile else None


def _profile_paths(profile: str) -> tuple[Path, Path]:
    profile = _validate_profile_name(profile)
    configured_root = os.getenv("BRRRAGENT_PROFILE_ROOT", "").strip()
    root = Path(configured_root or DEFAULT_PROFILE_ROOT).expanduser()
    profile_root = root / profile
    return profile_root / "codex-auth.json", profile_root / "codex-home"


def _codex_home() -> Path:
    active = _ACTIVE_AUTH_CONFIG.get()
    if active:
        if active.profile:
            return _profile_paths(active.profile)[1]
        if active.codex_home is not None:
            return Path(active.codex_home).expanduser()
    configured = os.getenv("BRRRAGENT_CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    profile = _configured_profile()
    if profile:
        return _profile_paths(profile)[1]
    legacy_home = os.getenv("CODEX_HOME", "").strip()
    return Path(legacy_home or DEFAULT_CODEX_HOME).expanduser()


def _codex_cli_auth_path() -> Path:
    return _codex_home() / "auth.json"


def _read_auth_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"brrragent auth file not found: {path}. Run `brrragent-auth login`."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"brrragent auth file is invalid JSON: {path}") from exc


def _write_auth_file(path: Path, auth: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            json.dump(auth, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)
        tmp_path.chmod(0o600)
        os.replace(tmp_path, path)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _is_codex_cli_auth(auth: dict | None) -> bool:
    if not isinstance(auth, dict):
        return False
    tokens = auth.get("tokens")
    return (
        isinstance(tokens, dict)
        and bool(tokens.get("id_token"))
        and bool(tokens.get("access_token"))
        and bool(tokens.get("refresh_token"))
    )


def _sync_codex_cli_auth_unlocked(path: Path, openai_auth: dict) -> None:
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        auth = {}

    id_token = openai_auth.get("idToken")
    if not id_token and _is_codex_cli_auth(auth):
        existing_account_id = auth["tokens"].get("account_id")
        if existing_account_id in {None, openai_auth.get("accountId")}:
            id_token = auth["tokens"]["id_token"]
    if not id_token:
        return

    tokens = {
        "id_token": id_token,
        "access_token": openai_auth["access"],
        "refresh_token": openai_auth["refresh"],
    }
    if openai_auth.get("accountId"):
        tokens["account_id"] = openai_auth["accountId"]
    _write_auth_file(
        path,
        {
            "OPENAI_API_KEY": None,
            "auth_mode": "chatgpt",
            "tokens": tokens,
            "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _sync_codex_cli_auth(openai_auth: dict) -> None:
    path = _codex_cli_auth_path()
    with _exclusive_path_lock(path):
        _sync_codex_cli_auth_unlocked(path, openai_auth)


def has_opencode_oauth() -> bool:
    try:
        auth = _read_auth_file(_auth_path())
    except Exception:
        return False
    openai = auth.get("openai") or {}
    return openai.get("type") == "oauth" and bool(openai.get("refresh"))


def _decode_jwt_payload(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        return json.loads(
            base64.urlsafe_b64decode((payload + padding).encode()).decode()
        )
    except Exception:
        return {}


def _validate_openai_auth(auth: dict, path: Path) -> dict:
    openai = auth.get("openai") or {}
    if openai.get("type") != "oauth" or not openai.get("refresh"):
        raise ValueError(f"Codex OAuth token not found in {path}")
    return openai


def _valid_stored_access(openai_auth: dict) -> tuple[str, str | None] | None:
    access_token = openai_auth.get("access")
    try:
        expires_ms = int(openai_auth.get("expires") or 0)
    except (TypeError, ValueError):
        expires_ms = 0
    if access_token and expires_ms > int(time.time() * 1000) + 120_000:
        return access_token, openai_auth.get("accountId")
    return None


def _refresh_access_token(openai_auth: dict) -> tuple[dict, str, str | None]:
    body = parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": openai_auth["refresh"],
            "client_id": CLIENT_ID,
        }
    ).encode()
    req = request.Request(
        TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise ValueError(
            f"Codex OAuth refresh failed: HTTP {exc.code}; {detail}"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Codex OAuth refresh failed: {exc}") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Codex OAuth refresh returned no access_token")

    try:
        expires_ms = (
            int(time.time() * 1000) + int(payload.get("expires_in") or 0) * 1000
        )
    except (TypeError, ValueError):
        expires_ms = int(time.time() * 1000) + 600_000
    claims = _decode_jwt_payload(access_token)
    account_id = openai_auth.get("accountId") or (
        claims.get("https://api.openai.com/auth") or {}
    ).get("chatgpt_account_id")
    refreshed_auth = {
        **openai_auth,
        "access": access_token,
        "refresh": payload.get("refresh_token") or openai_auth["refresh"],
        "expires": expires_ms,
    }
    if payload.get("id_token"):
        refreshed_auth["idToken"] = payload["id_token"]
    if account_id:
        refreshed_auth["accountId"] = account_id
    return refreshed_auth, access_token, account_id


def _get_codex_access_token() -> tuple[str, str | None]:
    path = _auth_path()
    with _exclusive_path_lock(path):
        auth = _read_auth_file(path)
        openai_auth = _validate_openai_auth(auth, path)
        stored = _valid_stored_access(openai_auth)
        if stored:
            return stored

        refreshed_auth, access_token, account_id = _refresh_access_token(openai_auth)
        auth["openai"] = refreshed_auth
        _write_auth_file(path, auth)
        _sync_codex_cli_auth(refreshed_auth)
        return access_token, account_id


def _codex_headers(
    access_token: str,
    account_id: str | None,
    *,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "originator": "opencode",
        "User-Agent": "opencode/brrragent",
        "session_id": session_id or f"brrragent-{uuid.uuid4()}",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def _extract_response_text(response_obj: dict | None) -> str:
    if not response_obj:
        return ""
    parts: list[str] = []
    for item in response_obj.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


def _extract_function_calls(response_obj: dict | None) -> list[dict]:
    if not response_obj:
        return []
    return [
        item
        for item in response_obj.get("output") or []
        if item.get("type") == "function_call"
    ]


def _stream_codex_response(
    payload: dict, headers: dict[str, str], timeout: int = 900
) -> dict:
    req = request.Request(
        CODEX_RESPONSES_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    text_parts: list[str] = []
    output_items: list[dict] = []
    final_response: dict | None = None
    response_id: str | None = None

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "response.output_text.delta" and event.get("delta"):
                    text_parts.append(event["delta"])

                response_obj = event.get("response")
                if response_obj:
                    final_response = response_obj
                    response_id = response_obj.get("id") or response_id
                    if response_obj.get("output"):
                        output_items = response_obj["output"]

                item = event.get("item")
                if item and event_type == "response.output_item.done":
                    output_items.append(item)

    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise ValueError(f"Codex Responses failed: HTTP {exc.code}; {detail}") from exc

    if output_items:
        final_response = {**(final_response or {}), "output": output_items}

    final_text = "".join(text_parts).strip() or _extract_response_text(final_response)
    return {
        "id": response_id,
        "text": final_text,
        "function_calls": _extract_function_calls(final_response),
        "usage": openai_usage((final_response or {}).get("usage")),
    }


def _build_payload(
    *,
    model: str,
    instructions: str | None,
    input_items: list[dict],
    tools: list[dict],
    response_schema: dict | None,
    previous_response_id: str | None = None,
    prompt_cache: PromptCacheConfig | None = None,
) -> dict:
    del previous_response_id
    bare_model, reasoning_effort = parse_codex_model(model)
    payload = {
        "model": bare_model,
        "input": input_items,
        "store": False,
        "stream": True,
    }
    if instructions:
        payload["instructions"] = instructions
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    if prompt_cache:
        payload["prompt_cache_key"] = prompt_cache.key
    if tools:
        payload["tools"] = [_to_responses_tool(tool) for tool in tools]
    if response_schema:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "agent_response",
                "strict": True,
                "schema": response_schema,
            }
        }
    return payload


def _is_transient_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return any(marker in err_str for marker in _RETRYABLE_MARKERS)


def run_codex_oauth_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    mcp,
    extra_tools: list[dict] | None,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Callable | None,
    response_schema: dict | None,
    prompt_cache: PromptCacheConfig | None = None,
    on_usage: Callable[[AgentUsage], None] | None = None,
    images: tuple[ImageInput, ...] = (),
) -> str:
    """Run agent against ChatGPT/Codex Responses using managed OAuth."""
    del temperature, max_tokens

    access_token, account_id = _get_codex_access_token()
    headers = _codex_headers(
        access_token,
        account_id,
        session_id=(
            f"brrragent-{uuid.uuid5(uuid.NAMESPACE_URL, prompt_cache.key)}"
            if prompt_cache
            else None
        ),
    )

    tools = mcp.get_openai_tools()
    if extra_tools:
        tools.extend(extra_tools)

    input_items = [
        {"role": "user", "content": _responses_user_content(user_prompt, images)}
    ]

    for turn in range(max_turns):
        bare_model, _ = parse_codex_model(model)
        logger.info(
            "[agent] Turn %d/%d - calling Codex OAuth %s",
            turn + 1,
            max_turns,
            bare_model,
        )

        payload = _build_payload(
            model=model,
            instructions=system_prompt,
            input_items=input_items,
            tools=tools,
            response_schema=response_schema,
            prompt_cache=prompt_cache,
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = _stream_codex_response(
                    payload, headers, timeout=_codex_timeout_seconds()
                )
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[agent] Codex OAuth attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    str(exc)[:200],
                )
                if not _is_transient_error(exc) or attempt >= max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
        else:
            raise ValueError(
                f"Codex OAuth failed after {max_retries} retries: {last_err}"
            )

        if on_usage:
            on_usage(result["usage"])

        function_calls = result.get("function_calls") or []
        if not function_calls:
            final_text = result.get("text", "")
            logger.info(
                "[agent] Final response after %d turn(s) (%d chars)",
                turn + 1,
                len(final_text),
            )
            return final_text

        logger.info("[agent] Turn %d: %d tool call(s)", turn + 1, len(function_calls))
        next_items = [*input_items]
        for tool_call in function_calls:
            fn_name = tool_call.get("name") or ""
            try:
                fn_args = json.loads(tool_call.get("arguments") or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[agent] Tool: %s(%s)", fn_name, list(fn_args.keys()))
            if on_tool_call:
                on_tool_call(fn_name, fn_args)

            result_text = mcp.call_tool(fn_name, fn_args)
            next_items.append(tool_call)
            next_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.get("call_id"),
                    "output": result_text,
                }
            )
        input_items = next_items

    logger.warning(
        "[agent] Reached max_turns=%d without final response; requesting final answer without tools",
        max_turns,
    )
    input_items.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Stop calling tools. Provide the final answer using the evidence already gathered.",
                }
            ],
        }
    )
    payload = _build_payload(
        model=model,
        instructions=system_prompt,
        input_items=input_items,
        tools=[],
        response_schema=response_schema,
        prompt_cache=prompt_cache,
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = _stream_codex_response(
                payload, headers, timeout=_codex_timeout_seconds()
            )
            if on_usage:
                on_usage(result["usage"])
            final_text = result.get("text", "")
            logger.info(
                "[agent] Final no-tool response after max_turns (%d chars)",
                len(final_text),
            )
            return final_text or "[No final response after max tool turns]"
        except Exception as exc:
            last_err = exc
            logger.warning(
                "[agent] Codex OAuth final synthesis attempt %d/%d failed: %s",
                attempt,
                max_retries,
                str(exc)[:200],
            )
            if not _is_transient_error(exc) or attempt >= max_retries:
                raise
            time.sleep(min(2**attempt, 30))

    raise ValueError(
        f"Codex OAuth final synthesis failed after {max_retries} retries: {last_err}"
    )
