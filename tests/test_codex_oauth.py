import json
import subprocess

import pytest

from brrragent import PromptCacheConfig
from brrragent.codex_oauth import (
    _auth_path,
    _build_payload,
    _codex_cli_auth_path,
    _codex_cli_config_args,
    _codex_headers,
    _codex_timeout_seconds,
    _extract_function_calls,
    _extract_response_text,
    _get_codex_access_token,
    _is_codex_spark_model,
    _parse_codex_cli_jsonl,
    _run_codex_cli_spark_completion,
    _sync_codex_cli_auth,
    has_opencode_oauth,
    parse_codex_model,
    run_codex_oauth_agent,
)


def test_auth_paths_default_to_brrragent_owned_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "BRRRAGENT_AUTH_PATH",
        "BRRRAGENT_OPENCODE_AUTH_PATH",
        "BRRRAGENT_CODEX_HOME",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _auth_path() == tmp_path / ".cache/brrragent/codex-auth.json"
    assert _codex_cli_auth_path() == (
        tmp_path / ".cache/brrragent/codex-home/auth.json"
    )


def test_auth_paths_preserve_explicit_legacy_overrides(monkeypatch, tmp_path):
    direct_path = tmp_path / "profile.json"
    codex_home = tmp_path / "codex-profile"
    monkeypatch.setenv("BRRRAGENT_OPENCODE_AUTH_PATH", str(direct_path))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert _auth_path() == direct_path
    assert _codex_cli_auth_path() == codex_home / "auth.json"


def test_parse_codex_model_strips_prefix_and_reasoning_suffix():
    assert parse_codex_model("codex/gpt-5.4:xhigh") == ("gpt-5.4", "xhigh")
    assert parse_codex_model("openai/gpt-5.4:high") == ("gpt-5.4", "high")


def test_codex_spark_detection_is_codex_prefix_guarded():
    assert _is_codex_spark_model("codex/gpt-5.3-codex-spark:xhigh")
    assert not _is_codex_spark_model("openai/gpt-5.3-codex-spark:xhigh")
    assert not _is_codex_spark_model("codex/gpt-5.4:xhigh")


def test_build_payload_sets_stable_cache_key():
    payload = _build_payload(
        model="codex/gpt-5.6-luna:medium",
        instructions="stable system",
        input_items=[{"role": "user", "content": "dynamic"}],
        tools=[],
        response_schema=None,
        prompt_cache=PromptCacheConfig("workflow:v2"),
    )

    assert payload["prompt_cache_key"] == "workflow:v2"
    assert payload["instructions"] == "stable system"
    assert payload["input"] == [{"role": "user", "content": "dynamic"}]


def test_codex_cli_config_args_maps_reasoning_effort():
    assert _codex_cli_config_args("xhigh") == ["-c", 'model_reasoning_effort="xhigh"']
    assert _codex_cli_config_args(None) == []
    assert _codex_cli_config_args("none") == []


def test_parse_codex_cli_jsonl_extracts_last_agent_message():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "SPARK_OK"},
                }
            ),
            json.dumps({"type": "turn.completed"}),
        ]
    )

    assert _parse_codex_cli_jsonl(stdout) == "SPARK_OK"


def test_codex_agent_forces_final_synthesis_after_tool_turn_limit(monkeypatch):
    payloads = []

    class FakeMcp:
        def get_openai_tools(self):
            return [{"type": "function", "name": "search"}]

        def call_tool(self, name, arguments):
            assert name == "search"
            assert arguments == {"q": "example"}
            return "Tool result"

    def fake_stream(payload, _headers, timeout):
        assert timeout > 0
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "function_calls": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": '{"q":"example"}',
                    }
                ]
            }
        return {"text": "Final response", "function_calls": []}

    monkeypatch.setattr(
        "brrragent.codex_oauth._get_codex_access_token", lambda: ("access", "account")
    )
    monkeypatch.setattr("brrragent.codex_oauth._stream_codex_response", fake_stream)

    result = run_codex_oauth_agent(
        system_prompt="Follow the request accurately.",
        user_prompt="Complete the task.",
        model="codex/gpt-5.5:xhigh",
        mcp=FakeMcp(),
        extra_tools=None,
        max_turns=1,
        temperature=0.2,
        max_tokens=1000,
        max_retries=2,
        on_tool_call=None,
        response_schema=None,
    )

    assert result == "Final response"
    assert len(payloads) == 2
    assert "tools" not in payloads[1]
    assert "Stop calling tools" in str(payloads[1]["input"])
    assert "Tool result" in str(payloads[1]["input"])


def _write_codex_cli_auth(codex_home):
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": None,
                "auth_mode": "chatgpt",
                "tokens": {
                    "id_token": "id.header.signature",
                    "access_token": "access.header.signature",
                    "refresh_token": "refresh-1",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_fake_executable(path):
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)


def test_run_codex_cli_spark_completion_builds_guarded_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "SPARK_OK"},
                }
            ),
            stderr="",
        )

    codex_binary = tmp_path / "codex"
    _write_fake_executable(codex_binary)
    _write_codex_cli_auth(tmp_path / "codex-home")

    monkeypatch.setenv("BRRRAGENT_CODEX_CLI", str(codex_binary))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("BRRRAGENT_CODEX_CLI_CWD", "/tmp/codex-test")
    monkeypatch.setattr("brrragent.codex_oauth.subprocess.run", fake_run)

    result = _run_codex_cli_spark_completion(
        system_prompt="system",
        user_prompt="user",
        model="codex/gpt-5.3-codex-spark:xhigh",
        response_schema=None,
        timeout=45,
    )

    assert result == "SPARK_OK"
    assert captured["cmd"][:8] == [
        str(codex_binary),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "-s",
        "read-only",
    ]
    assert captured["cmd"][captured["cmd"].index("-m") + 1] == "gpt-5.3-codex-spark"
    assert 'model_reasoning_effort="xhigh"' in captured["cmd"]
    assert "--json" in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 45
    assert captured["kwargs"]["env"]["CODEX_HOME"] == str(tmp_path / "codex-home")


def test_run_codex_cli_spark_completion_fails_fast_when_codex_missing(
    monkeypatch, tmp_path
):
    _write_codex_cli_auth(tmp_path / "codex-home")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("BRRRAGENT_CODEX_CLI", str(tmp_path / "missing-codex"))

    with pytest.raises(ValueError, match="codex CLI not found"):
        _run_codex_cli_spark_completion(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.3-codex-spark:xhigh",
            response_schema=None,
            timeout=45,
        )


def test_run_codex_cli_spark_completion_fails_fast_when_auth_missing(
    monkeypatch, tmp_path
):
    codex_binary = tmp_path / "codex"
    _write_fake_executable(codex_binary)
    monkeypatch.setenv("BRRRAGENT_CODEX_CLI", str(codex_binary))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-home"))

    with pytest.raises(ValueError, match="Codex CLI auth file not found"):
        _run_codex_cli_spark_completion(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.3-codex-spark:xhigh",
            response_schema=None,
            timeout=45,
        )


def test_run_codex_cli_spark_completion_fails_fast_when_auth_shape_invalid(
    monkeypatch, tmp_path
):
    codex_binary = tmp_path / "codex"
    codex_home = tmp_path / "codex-home"
    _write_fake_executable(codex_binary)
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps({"openai": {"type": "oauth", "refresh": "refresh-1"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BRRRAGENT_CODEX_CLI", str(codex_binary))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(ValueError, match="missing tokens.id_token"):
        _run_codex_cli_spark_completion(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.3-codex-spark:xhigh",
            response_schema=None,
            timeout=45,
        )


def test_sync_codex_cli_auth_updates_existing_raw_auth(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": None,
                "auth_mode": "chatgpt",
                "tokens": {
                    "id_token": "id.header.signature",
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "account_id": "old-acct",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _sync_codex_cli_auth(
        {
            "access": "new-access",
            "refresh": "new-refresh",
            "accountId": "new-acct",
        }
    )

    written = json.loads(auth_path.read_text(encoding="utf-8"))
    assert written["tokens"]["id_token"] == "id.header.signature"
    assert written["tokens"]["access_token"] == "new-access"
    assert written["tokens"]["refresh_token"] == "new-refresh"
    assert written["tokens"]["account_id"] == "new-acct"


def test_codex_headers_keep_existing_opencode_envelope_for_regular_models():
    headers = _codex_headers(
        "access",
        "acct_1",
        model="codex/gpt-5.4:xhigh",
        request_identity=None,
    )

    assert headers["originator"] == "opencode"
    assert headers["User-Agent"] == "opencode/brrragent"
    assert headers["ChatGPT-Account-Id"] == "acct_1"
    assert "session_id" in headers
    assert "session-id" not in headers


def test_codex_headers_accept_stable_cache_session():
    headers = _codex_headers(
        "access",
        "account",
        model="codex/gpt-5.6-luna:medium",
        session_id="brrragent-stable",
    )

    assert headers["session_id"] == "brrragent-stable"


def test_codex_headers_match_cli_envelope_for_spark(monkeypatch):
    monkeypatch.setenv("BRRRAGENT_CODEX_USER_AGENT", "codex_cli_rs/test")
    identity = {
        "installation_id": "install_1",
        "session_id": "session_1",
        "thread_id": "thread_1",
        "window_id": "window_1",
    }

    headers = _codex_headers(
        "access",
        "acct_1",
        model="codex/gpt-5.3-codex-spark:xhigh",
        request_identity=identity,
    )

    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"] == "codex_cli_rs/test"
    assert headers["ChatGPT-Account-ID"] == "acct_1"
    assert headers["session-id"] == "session_1"
    assert headers["thread-id"] == "thread_1"
    assert headers["x-client-request-id"] == "thread_1"
    assert headers["x-codex-installation-id"] == "install_1"
    assert headers["x-codex-window-id"] == "window_1"
    assert "session_id" not in headers


def test_build_payload_matches_codex_backend_constraints():
    payload = _build_payload(
        model="codex/gpt-5.4:xhigh",
        instructions="system",
        input_items=[
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        tools=[],
        response_schema=None,
    )

    assert payload["model"] == "gpt-5.4"
    assert payload["instructions"] == "system"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload


def test_build_payload_adds_codex_metadata_for_spark():
    identity = {
        "installation_id": "install_1",
        "session_id": "session_1",
        "thread_id": "thread_1",
        "window_id": "window_1",
    }
    payload = _build_payload(
        model="codex/gpt-5.3-codex-spark:xhigh",
        instructions="system",
        input_items=[
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        tools=[],
        response_schema=None,
        request_identity=identity,
    )

    assert payload["model"] == "gpt-5.3-codex-spark"
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["prompt_cache_key"] == "brrragent-codex-spark"
    assert payload["client_metadata"] == {
        "x-codex-installation-id": "install_1",
        "session_id": "session_1",
        "thread_id": "thread_1",
        "x-codex-window-id": "window_1",
    }


def test_build_payload_ignores_previous_response_id_for_codex_backend():
    payload = _build_payload(
        model="openai/gpt-5.4-mini",
        instructions="system",
        input_items=[
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        ],
        tools=[],
        response_schema=None,
        previous_response_id="resp_1",
    )

    assert "previous_response_id" not in payload
    assert payload["instructions"] == "system"


def test_extract_response_text_and_function_calls():
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            },
            {
                "type": "function_call",
                "name": "search",
                "arguments": json.dumps({"q": "test"}),
                "call_id": "call_1",
            },
        ]
    }

    assert _extract_response_text(response) == "done"
    assert _extract_function_calls(response)[0]["name"] == "search"


def test_has_opencode_oauth_reads_configured_path(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"openai": {"type": "oauth", "refresh": "rt_test"}})
    )
    monkeypatch.setenv("BRRRAGENT_OPENCODE_AUTH_PATH", str(auth_path))

    assert has_opencode_oauth()


def test_codex_timeout_uses_env_with_floor(monkeypatch):
    monkeypatch.setenv("BRRRAGENT_CODEX_TIMEOUT_SECONDS", "12")
    assert _codex_timeout_seconds() == 30

    monkeypatch.setenv("BRRRAGENT_CODEX_TIMEOUT_SECONDS", "120")
    assert _codex_timeout_seconds() == 120


def test_get_codex_access_token_reuses_valid_access(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "openai": {
                    "type": "oauth",
                    "refresh": "rt_old",
                    "access": "access_valid",
                    "expires": 4_102_444_800_000,
                    "accountId": "acct_1",
                }
            }
        )
    )
    monkeypatch.setenv("BRRRAGENT_OPENCODE_AUTH_PATH", str(auth_path))

    def fail_refresh(*args, **kwargs):
        raise AssertionError("refresh should not be called")

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fail_refresh)

    assert _get_codex_access_token() == ("access_valid", "acct_1")


def test_get_codex_access_token_persists_rotated_refresh(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "openai": {
                    "type": "oauth",
                    "refresh": "rt_old",
                    "access": "access_old",
                    "expires": 1,
                    "accountId": "acct_1",
                }
            }
        )
    )
    monkeypatch.setenv("BRRRAGENT_OPENCODE_AUTH_PATH", str(auth_path))

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "access_token": "access_new",
                    "refresh_token": "rt_new",
                    "expires_in": 3600,
                }
            ).encode()

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        return Response()

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fake_urlopen)

    assert _get_codex_access_token() == ("access_new", "acct_1")
    assert _get_codex_access_token() == ("access_new", "acct_1")
    assert len(calls) == 1

    saved = json.loads(auth_path.read_text())
    assert saved["openai"]["access"] == "access_new"
    assert saved["openai"]["refresh"] == "rt_new"
    assert saved["openai"]["expires"] > 1
