"""
Codex OAuth backend using OpenCode's stored ChatGPT/Codex login.

This backend mirrors OpenCode's OpenAI provider behavior:
  - read ~/.local/share/opencode/auth.json
  - refresh the OpenAI OAuth token
  - call https://chatgpt.com/backend-api/codex/responses
"""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional
from urllib import error, parse, request

from brrragent.openai_direct import parse_openai_model, _to_responses_tool

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_AUTH_PATH = "~/.local/share/opencode/auth.json"
_RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "timeout", "temporarily unavailable")


def parse_codex_model(model: str) -> tuple[str, str | None]:
    if model.startswith("codex/"):
        return parse_openai_model("openai/" + model[len("codex/") :])
    return parse_openai_model(model)


def _auth_path() -> Path:
    return Path(os.getenv("BRRRAGENT_OPENCODE_AUTH_PATH", DEFAULT_AUTH_PATH)).expanduser()


def _codex_timeout_seconds() -> int:
    raw = os.getenv("BRRRAGENT_CODEX_TIMEOUT_SECONDS") or os.getenv("AI_CLIENT_TIMEOUT_SECONDS")
    try:
        return max(30, int(raw or "900"))
    except ValueError:
        return 900


def _auth_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.brrragent.lock")


def _read_auth_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"OpenCode auth file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenCode auth file is invalid JSON: {path}") from exc


def _write_auth_file(path: Path, auth: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(auth, indent=2), encoding="utf-8")
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o600
    tmp_path.chmod(mode)
    os.replace(tmp_path, path)


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
        return json.loads(base64.urlsafe_b64decode((payload + padding).encode()).decode())
    except Exception:
        return {}


def _validate_openai_auth(auth: dict, path: Path) -> dict:
    openai = auth.get("openai") or {}
    if openai.get("type") != "oauth" or not openai.get("refresh"):
        raise ValueError(f"OpenCode OpenAI OAuth token not found in {path}")
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
        raise ValueError(f"Codex OAuth refresh failed: HTTP {exc.code}; {detail}") from exc
    except Exception as exc:
        raise ValueError(f"Codex OAuth refresh failed: {exc}") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Codex OAuth refresh returned no access_token")

    try:
        expires_ms = int(time.time() * 1000) + int(payload.get("expires_in") or 0) * 1000
    except (TypeError, ValueError):
        expires_ms = int(time.time() * 1000) + 600_000
    claims = _decode_jwt_payload(access_token)
    account_id = (
        openai_auth.get("accountId")
        or (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    )
    refreshed_auth = {
        **openai_auth,
        "access": access_token,
        "refresh": payload.get("refresh_token") or openai_auth["refresh"],
        "expires": expires_ms,
    }
    if account_id:
        refreshed_auth["accountId"] = account_id
    return refreshed_auth, access_token, account_id


def _get_codex_access_token() -> tuple[str, str | None]:
    path = _auth_path()
    lock_path = _auth_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        auth = _read_auth_file(path)
        openai_auth = _validate_openai_auth(auth, path)
        stored = _valid_stored_access(openai_auth)
        if stored:
            return stored

        refreshed_auth, access_token, account_id = _refresh_access_token(openai_auth)
        auth["openai"] = refreshed_auth
        _write_auth_file(path, auth)
        return access_token, account_id


def _codex_headers(access_token: str, account_id: str | None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "originator": "opencode",
        "User-Agent": "opencode/brrragent",
        "session_id": f"brrragent-{uuid.uuid4()}",
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


def _stream_codex_response(payload: dict, headers: dict[str, str], timeout: int = 900) -> dict:
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
    }


def _build_payload(
    *,
    model: str,
    instructions: str | None,
    input_items: list[dict],
    tools: list[dict],
    response_schema: dict | None,
    previous_response_id: str | None = None,
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
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
) -> str:
    """Run the agent against ChatGPT/Codex Responses using OpenCode OAuth."""
    del temperature, max_tokens

    access_token, account_id = _get_codex_access_token()
    headers = _codex_headers(access_token, account_id)

    tools = mcp.get_openai_tools()
    if extra_tools:
        tools.extend(extra_tools)

    input_items = [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}]

    for turn in range(max_turns):
        bare_model, _ = parse_codex_model(model)
        logger.info("[agent] Turn %d/%d - calling Codex OAuth %s", turn + 1, max_turns, bare_model)

        payload = _build_payload(
            model=model,
            instructions=system_prompt,
            input_items=input_items,
            tools=tools,
            response_schema=response_schema,
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = _stream_codex_response(payload, headers, timeout=_codex_timeout_seconds())
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
            raise ValueError(f"Codex OAuth failed after {max_retries} retries: {last_err}")

        function_calls = result.get("function_calls") or []
        if not function_calls:
            final_text = result.get("text", "")
            logger.info("[agent] Final response after %d turn(s) (%d chars)", turn + 1, len(final_text))
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

    logger.warning("[agent] Reached max_turns=%d without final response", max_turns)
    return "[No final response after max tool turns]"
