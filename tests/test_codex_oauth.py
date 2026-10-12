import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from brrragent import ImageInput, PromptCacheConfig
from brrragent.codex_oauth import (
    CodexAuthConfig,
    _auth_path,
    _build_payload,
    _codex_cli_auth_path,
    _codex_headers,
    _codex_timeout_seconds,
    _extract_function_calls,
    _extract_response_text,
    _get_codex_access_token,
    _refresh_access_token,
    _stream_codex_response,
    _sync_codex_cli_auth,
    _write_auth_file,
    codex_auth_context,
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
        "BRRRAGENT_PROFILE",
        "BRRRAGENT_PROFILE_ROOT",
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


def test_named_profile_resolves_isolated_paths(monkeypatch, tmp_path):
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv("BRRRAGENT_PROFILE", "content-worker")
    monkeypatch.setenv("BRRRAGENT_PROFILE_ROOT", str(profile_root))
    for name in (
        "BRRRAGENT_AUTH_PATH",
        "BRRRAGENT_OPENCODE_AUTH_PATH",
        "BRRRAGENT_CODEX_HOME",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _auth_path() == profile_root / "content-worker/codex-auth.json"
    assert _codex_cli_auth_path() == (
        profile_root / "content-worker/codex-home/auth.json"
    )


def test_explicit_paths_override_named_profile(monkeypatch, tmp_path):
    direct_path = tmp_path / "direct.json"
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("BRRRAGENT_PROFILE", "ignored-profile")
    monkeypatch.setenv("BRRRAGENT_AUTH_PATH", str(direct_path))
    monkeypatch.setenv("BRRRAGENT_CODEX_HOME", str(codex_home))

    assert _auth_path() == direct_path
    assert _codex_cli_auth_path() == codex_home / "auth.json"


def test_named_profile_overrides_generic_codex_home(monkeypatch, tmp_path):
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv("BRRRAGENT_PROFILE", "content-worker")
    monkeypatch.setenv("BRRRAGENT_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "shared-codex-home"))
    monkeypatch.delenv("BRRRAGENT_CODEX_HOME", raising=False)

    assert _codex_cli_auth_path() == (
        profile_root / "content-worker/codex-home/auth.json"
    )


def test_named_profile_rejects_path_traversal(monkeypatch):
    monkeypatch.setenv("BRRRAGENT_PROFILE", "../shared")

    with pytest.raises(ValueError, match="Profile must contain"):
        _auth_path()


def test_per_call_auth_paths_override_environment(monkeypatch, tmp_path):
    auth_path = tmp_path / "request/auth.json"
    codex_home = tmp_path / "request/codex-home"
    monkeypatch.setenv("BRRRAGENT_PROFILE", "environment-profile")

    with codex_auth_context(
        CodexAuthConfig(auth_path=auth_path, codex_home=codex_home)
    ):
        assert _auth_path() == auth_path
        assert _codex_cli_auth_path() == codex_home / "auth.json"

    assert "environment-profile" in str(_auth_path())


def test_concurrent_auth_contexts_do_not_cross_accounts(tmp_path):
    configs = []
    expected = []
    for suffix in ("a", "b"):
        profile_dir = tmp_path / f"profile-{suffix}"
        profile_dir.mkdir()
        auth_path = profile_dir / "auth.json"
        _write_direct_auth(auth_path, account_id=f"acct-{suffix}")
        configs.append(
            CodexAuthConfig(
                auth_path=auth_path,
                codex_home=profile_dir / "codex-home",
            )
        )
        expected.append(("access.header.signature", f"acct-{suffix}", auth_path))

    barrier = threading.Barrier(2)

    def load_auth(config):
        with codex_auth_context(config):
            barrier.wait(timeout=10)
            result = (*_get_codex_access_token(), _auth_path())
            barrier.wait(timeout=10)
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(load_auth, configs))

    assert results == expected


def test_codex_auth_config_rejects_ambiguous_selection(tmp_path):
    with pytest.raises(ValueError, match="requires a profile"):
        CodexAuthConfig()

    with pytest.raises(ValueError, match="cannot be combined"):
        CodexAuthConfig(profile="content-worker", auth_path=tmp_path / "auth.json")

    with pytest.raises(ValueError, match="must be provided together"):
        CodexAuthConfig(auth_path=tmp_path / "auth.json")

    with pytest.raises(ValueError, match="must differ"):
        CodexAuthConfig(
            auth_path=tmp_path / "codex-home/auth.json",
            codex_home=tmp_path / "codex-home",
        )


def test_external_provider_does_not_read_or_sync_local_auth(monkeypatch):
    provider = Mock(return_value=("shared-access", None))
    local_path = Mock(side_effect=AssertionError("local auth must not be accessed"))
    cli_sync = Mock(side_effect=AssertionError("CLI auth must not be synchronized"))
    monkeypatch.setattr("brrragent.codex_oauth._auth_path", local_path)
    monkeypatch.setattr("brrragent.codex_oauth._sync_codex_cli_auth", cli_sync)

    with codex_auth_context(CodexAuthConfig(access_token_provider=provider)):
        assert has_opencode_oauth()
        provider.assert_not_called()
        assert _get_codex_access_token() == ("shared-access", None)

    provider.assert_called_once_with()
    local_path.assert_not_called()
    cli_sync.assert_not_called()


def test_external_provider_failure_propagates_without_local_fallback(
    monkeypatch, caplog
):
    failure = OSError("credential store unavailable")
    provider = Mock(side_effect=failure)
    local_path = Mock(side_effect=AssertionError("local auth must not be accessed"))
    monkeypatch.setattr("brrragent.codex_oauth._auth_path", local_path)

    with codex_auth_context(
        CodexAuthConfig(profile="content-worker", access_token_provider=provider)
    ):
        assert has_opencode_oauth()
        with pytest.raises(OSError) as captured:
            _get_codex_access_token()

    assert captured.value is failure
    local_path.assert_not_called()
    assert not caplog.records


@pytest.mark.parametrize(
    "supplied",
    [
        None,
        ["secret-access", None],
        ("secret-access",),
        ("secret-access", None, "secret-refresh"),
        (None, None),
        ("", None),
        ("secret-access\r\nInjected:secret", None),
        ("secret-access", ""),
        ("secret-access", 42),
        ("secret-access", "account\tsecret"),
        ("secret-access\x7f", None),
        ("secret-access", "account\u0080"),
    ],
)
def test_external_provider_rejects_invalid_values_without_exposing_them(
    supplied, monkeypatch, caplog
):
    local_path = Mock(side_effect=AssertionError("local auth must not be accessed"))
    monkeypatch.setattr("brrragent.codex_oauth._auth_path", local_path)
    with codex_auth_context(CodexAuthConfig(access_token_provider=lambda: supplied)):
        with pytest.raises(ValueError, match="Codex access token provider") as captured:
            _get_codex_access_token()

    assert "secret" not in str(captured.value)
    assert not caplog.records
    local_path.assert_not_called()


def test_external_provider_is_callable_and_hidden_from_config_repr():
    with pytest.raises(TypeError, match="must be callable"):
        CodexAuthConfig(access_token_provider="secret-access")

    class Provider:
        def __call__(self):
            return "secret-access", None

        def __repr__(self):
            return "secret-refresh"

    assert "secret" not in repr(CodexAuthConfig(access_token_provider=Provider()))


def test_rejected_provider_requires_external_provider_and_hides_its_repr():
    with pytest.raises(ValueError, match="requires access_token_provider"):
        CodexAuthConfig(rejected_access_token_provider=lambda _token: ("access", None))
    with pytest.raises(TypeError, match="must be callable"):
        CodexAuthConfig(
            access_token_provider=lambda: ("access", None),
            rejected_access_token_provider="secret-refresh",
        )
    provider = Mock(return_value=("access", None))
    provider.__repr__ = Mock(return_value="secret-refresh")
    assert "secret-refresh" not in repr(
        CodexAuthConfig(
            access_token_provider=provider,
            rejected_access_token_provider=provider,
        )
    )


def test_concurrent_external_providers_preserve_profiles_and_restore_context():
    barrier = threading.Barrier(2)
    configs = [
        CodexAuthConfig(
            profile=f"worker-{suffix}",
            access_token_provider=lambda suffix=suffix: (
                f"access-{suffix}",
                f"acct-{suffix}",
            ),
        )
        for suffix in ("a", "b")
    ]

    def load_auth(config):
        with codex_auth_context(config):
            barrier.wait(timeout=10)
            credentials = _get_codex_access_token()
            with codex_auth_context(
                CodexAuthConfig(access_token_provider=lambda: ("nested-access", None))
            ):
                assert _get_codex_access_token() == ("nested-access", None)
            barrier.wait(timeout=10)
            assert _get_codex_access_token() == credentials
            return credentials

    with codex_auth_context(
        CodexAuthConfig(access_token_provider=lambda: ("outer-access", "outer-account"))
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(load_auth, configs))
        assert _get_codex_access_token() == ("outer-access", "outer-account")

    assert results == [("access-a", "acct-a"), ("access-b", "acct-b")]


def test_parse_codex_model_strips_prefix_and_reasoning_suffix():
    assert parse_codex_model("codex/gpt-5.4:xhigh") == ("gpt-5.4", "xhigh")
    assert parse_codex_model("openai/gpt-5.4:high") == ("gpt-5.4", "high")


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


def test_codex_agent_sends_multiple_images_with_structured_output(monkeypatch):
    payloads = []

    class FakeMcp:
        def get_openai_tools(self):
            return []

    def fake_stream(payload, _headers, timeout):
        assert timeout > 0
        payloads.append(payload)
        return {"text": '{"answer":"mushroom"}', "function_calls": []}

    monkeypatch.setattr(
        "brrragent.codex_oauth._get_codex_access_token", lambda: ("access", "account")
    )
    monkeypatch.setattr("brrragent.codex_oauth._stream_codex_response", fake_stream)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = run_codex_oauth_agent(
        system_prompt="Analyze images.",
        user_prompt="What is shown?",
        model="codex/gpt-5.6-luna:medium",
        mcp=FakeMcp(),
        extra_tools=None,
        max_turns=1,
        temperature=0.2,
        max_tokens=1000,
        max_retries=1,
        on_tool_call=None,
        response_schema=schema,
        images=(
            ImageInput("https://example.test/one.png", detail="low"),
            ImageInput.from_bytes(b"jpeg", media_type="image/jpeg", detail="high"),
        ),
    )

    assert result == '{"answer":"mushroom"}'
    content = payloads[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "What is shown?"}
    assert content[1] == {
        "type": "input_image",
        "image_url": "https://example.test/one.png",
        "detail": "low",
    }
    assert content[2]["image_url"].startswith("data:image/jpeg;base64,")
    assert content[2]["detail"] == "high"
    assert payloads[0]["text"]["format"]["schema"] is schema


def _write_direct_auth(path, *, account_id="acct-1", id_token="id.header.signature"):
    path.write_text(
        json.dumps(
            {
                "openai": {
                    "type": "oauth",
                    "access": "access.header.signature",
                    "refresh": "refresh-1",
                    "idToken": id_token,
                    "expires": 4_102_444_800_000,
                    "accountId": account_id,
                }
            }
        ),
        encoding="utf-8",
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
            "idToken": "new.id.signature",
            "accountId": "new-acct",
        }
    )

    written = json.loads(auth_path.read_text(encoding="utf-8"))
    assert written["tokens"]["id_token"] == "new.id.signature"
    assert written["tokens"]["access_token"] == "new-access"
    assert written["tokens"]["refresh_token"] == "new-refresh"
    assert written["tokens"]["account_id"] == "new-acct"


def test_atomic_auth_writes_use_unique_thread_tempfiles(tmp_path):
    path = tmp_path / "auth.json"
    barrier = threading.Barrier(8)

    def write(value):
        barrier.wait()
        _write_auth_file(path, {"value": value})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    assert json.loads(path.read_text())["value"] in range(8)
    assert list(tmp_path.glob(".auth.json.*.tmp")) == []


@pytest.mark.parametrize("initial_mode", [None, 0o644, 0o660])
def test_auth_writes_enforce_private_permissions(tmp_path, initial_mode):
    path = tmp_path / "auth.json"
    if initial_mode is not None:
        path.write_text("{}")
        path.chmod(initial_mode)
    auth = {"openai": {"access": "fake-access", "refresh": "fake-refresh"}}

    _write_auth_file(path, auth)

    assert json.loads(path.read_text()) == auth
    assert path.stat().st_mode & 0o777 == 0o600


def test_codex_headers_keep_existing_opencode_envelope_for_regular_models():
    headers = _codex_headers(
        "access",
        "acct_1",
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
        session_id="brrragent-stable",
    )

    assert headers["session_id"] == "brrragent-stable"


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


def test_stream_codex_response_parses_sse_result(monkeypatch):
    events = [
        b"event: response.output_text.delta\n",
        b"data: {malformed\n",
        b'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
        b'data: {"type":"response.output_text.delta","delta":" world"}\n',
        b"data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "fallback text"}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 4},
                    },
                },
            }
        ).encode()
        + b"\n",
        b"data: "
        + json.dumps(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": '{"q":"test"}',
                },
            }
        ).encode()
        + b"\n",
        b"data: [DONE]\n",
        b'data: {"type":"response.output_text.delta","delta":" ignored"}\n',
    ]
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(events)

    def fake_urlopen(req, timeout):
        captured["req"] = req
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fake_urlopen)

    result = _stream_codex_response(
        {"model": "gpt-5.4", "stream": True},
        {"Authorization": "Bearer test-token"},
        timeout=47,
    )

    assert result["id"] == "resp_123"
    assert result["text"] == "Hello world"
    assert result["function_calls"] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"q":"test"}',
        }
    ]
    assert result["usage"].input_tokens == 12
    assert result["usage"].output_tokens == 3
    assert result["usage"].total_tokens == 15
    assert result["usage"].cache_read_input_tokens == 4
    assert captured["timeout"] == 47
    assert json.loads(captured["req"].data) == {
        "model": "gpt-5.4",
        "stream": True,
    }
    assert captured["req"].get_header("Authorization") == "Bearer test-token"


def _sse_response(*events):
    return io.BytesIO(
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        + b"data: [DONE]\n"
    )


@pytest.mark.parametrize("replacement", ["refreshed-access", "peer-access"])
def test_stream_recovers_pre_stream_401_with_rejected_access_provider(
    monkeypatch, replacement, caplog
):
    provider = Mock(return_value=(replacement, "account"))
    initial_provider = Mock(return_value=("stale-access", "account"))
    auth = CodexAuthConfig(
        access_token_provider=initial_provider,
        rejected_access_token_provider=provider,
    )
    body = io.BytesIO(b"private provider response")
    urlopen = Mock(
        side_effect=[
            HTTPError(
                "https://chatgpt.com/backend-api/codex/responses",
                401,
                "Unauthorized",
                None,
                body,
            ),
            _sse_response({"type": "response.output_text.delta", "delta": "recovered"}),
        ]
    )
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    headers = _codex_headers("stale-access", "account", session_id="stable-session")
    with codex_auth_context(auth):
        assert (
            _stream_codex_response({"input": []}, headers, timeout=31)["text"]
            == "recovered"
        )

    provider.assert_called_once_with("stale-access")
    initial_provider.assert_not_called()
    assert urlopen.call_count == 2
    first, second = (call.args[0] for call in urlopen.call_args_list)
    assert first.get_header("Authorization") == "Bearer stale-access"
    assert second.get_header("Authorization") == f"Bearer {replacement}"
    assert first.data == second.data
    assert second.get_header("Chatgpt-account-id") == "account"
    assert second.get_header("Session_id") == "stable-session"
    assert headers["Authorization"] == f"Bearer {replacement}"
    assert all(call.kwargs["timeout"] == 31 for call in urlopen.call_args_list)
    assert body.closed
    assert "private provider response" not in caplog.text


@pytest.mark.parametrize("same_access,expected_calls", [(False, 2), (True, 1)])
def test_pre_stream_401_recovery_does_not_repeat_rejected_credentials(
    monkeypatch, same_access, expected_calls, caplog
):
    provider = Mock(
        return_value=("stale-access" if same_access else "new-access", None)
    )

    def reject(*_args, **_kwargs):
        raise HTTPError(
            "https://chatgpt.com/backend-api/codex/responses",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"secret response"),
        )

    urlopen = Mock(side_effect=reject)
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=lambda: ("stale-access", None),
            rejected_access_token_provider=provider,
        )
    ):
        with pytest.raises(ValueError, match="Codex Responses failed: HTTP 401"):
            _stream_codex_response({}, _codex_headers("stale-access", None))

    provider.assert_called_once_with("stale-access")
    assert urlopen.call_count == expected_calls
    assert "secret" not in caplog.text


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_auth_recovery_requires_401_and_an_explicit_callback(monkeypatch, status):
    provider = Mock(return_value=("new-access", None))
    urlopen = Mock(
        side_effect=HTTPError(
            "https://chatgpt.com/backend-api/codex/responses",
            status,
            "Synthetic failure",
            None,
            io.BytesIO(b"private body"),
        )
    )
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=lambda: ("stale-access", None),
            rejected_access_token_provider=provider if status != 401 else None,
        )
    ):
        with pytest.raises(ValueError, match=f"HTTP {status}"):
            _stream_codex_response({}, _codex_headers("stale-access", None))
    provider.assert_not_called()
    assert urlopen.call_count == 1


@pytest.mark.parametrize("stage", ["enter", "iteration"])
def test_open_stream_401_never_invokes_auth_recovery(monkeypatch, stage):
    provider = Mock(return_value=("new-access", None))

    class Response:
        def __enter__(self):
            if stage == "enter":
                self.fail()
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}\n'
            self.fail()

        def fail(self):
            raise HTTPError(
                "https://chatgpt.com/backend-api/codex/responses",
                401,
                "Unauthorized",
                None,
                io.BytesIO(b"secret stream body"),
            )

    urlopen = Mock(return_value=Response())
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=lambda: ("stale-access", None),
            rejected_access_token_provider=provider,
        )
    ):
        with pytest.raises(ValueError, match="Codex Responses failed: HTTP 401"):
            _stream_codex_response({}, _codex_headers("stale-access", None))
    provider.assert_not_called()
    assert urlopen.call_count == 1


@pytest.mark.parametrize(
    "supplied",
    [
        None,
        ["secret-access", "account"],
        ("secret-access\r\n", "account"),
        ("secret-access", "wrong-secret-account"),
        ("secret-access", None),
    ],
)
def test_rejected_provider_invalid_or_changed_account_stops_before_retry(
    monkeypatch, supplied, caplog
):
    provider = Mock(return_value=supplied)
    urlopen = Mock(
        side_effect=HTTPError(
            "https://chatgpt.com/backend-api/codex/responses",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"secret response"),
        )
    )
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=lambda: ("stale-access", "account"),
            rejected_access_token_provider=provider,
        )
    ):
        with pytest.raises(
            ValueError, match="Codex rejected access token provider"
        ) as captured:
            _stream_codex_response({}, _codex_headers("stale-access", "account"))
    provider.assert_called_once_with("stale-access")
    assert urlopen.call_count == 1
    assert "secret" not in str(captured.value)
    assert "secret" not in caplog.text


def test_rejected_provider_failure_is_redacted_and_never_transient(
    monkeypatch, streamed_codex_agent, caplog
):
    provider = Mock(
        side_effect=TimeoutError("500 temporarily unavailable secret token")
    )
    urlopen = Mock(
        side_effect=HTTPError(
            "https://chatgpt.com/backend-api/codex/responses",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"secret response"),
        )
    )
    local = Mock(side_effect=AssertionError("local fallback must not run"))
    sleep = Mock()
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    monkeypatch.setattr("brrragent.codex_oauth._auth_path", local)
    monkeypatch.setattr("brrragent.codex_oauth.time.sleep", sleep)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=lambda: ("test-token", None),
            rejected_access_token_provider=provider,
        )
    ):
        with pytest.raises(
            ValueError, match="Codex rejected access token provider failed"
        ) as captured:
            streamed_codex_agent(1)
    provider.assert_called_once_with("test-token")
    assert urlopen.call_count == 1
    sleep.assert_not_called()
    local.assert_not_called()
    assert captured.value.__suppress_context__
    assert "secret" not in str(captured.value)
    assert "secret" not in caplog.text


def test_auth_recovery_preserves_new_headers_across_tool_turns(monkeypatch):
    provider = Mock(return_value=("new-access", "account"))
    initial_provider = Mock(return_value=("stale-access", "account"))
    requests = []
    responses = [
        _sse_response(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": '{"q":"example"}',
                },
            }
        ),
        _sse_response({"type": "response.output_text.delta", "delta": "finished"}),
    ]

    def open_response(req, timeout):
        requests.append(req)
        if len(requests) == 1:
            raise HTTPError(
                req.full_url, 401, "Unauthorized", None, io.BytesIO(b"secret")
            )
        return responses.pop(0)

    mcp = Mock()
    mcp.get_openai_tools.return_value = [{"type": "function", "name": "search"}]
    mcp.call_tool.return_value = "tool result"
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", open_response)
    with codex_auth_context(
        CodexAuthConfig(
            access_token_provider=initial_provider,
            rejected_access_token_provider=provider,
        )
    ):
        assert (
            run_codex_oauth_agent(
                system_prompt="system",
                user_prompt="user",
                model="codex/gpt-5.4",
                mcp=mcp,
                extra_tools=None,
                max_turns=1,
                temperature=0,
                max_tokens=100,
                max_retries=1,
                on_tool_call=None,
                response_schema=None,
            )
            == "finished"
        )
    assert [req.get_header("Authorization") for req in requests] == [
        "Bearer stale-access",
        "Bearer new-access",
        "Bearer new-access",
    ]
    assert len({req.get_header("Session_id") for req in requests}) == 1
    provider.assert_called_once_with("stale-access")
    initial_provider.assert_called_once_with()
    mcp.call_tool.assert_called_once_with("search", {"q": "example"})


def test_concurrent_rejected_providers_preserve_account_context(monkeypatch):
    barrier = threading.Barrier(2)
    providers = {
        suffix: Mock(return_value=(f"new-{suffix}", f"account-{suffix}"))
        for suffix in ("a", "b")
    }

    def open_response(req, timeout):
        authorization = req.get_header("Authorization")
        suffix = authorization.rsplit("-", 1)[1]
        assert req.get_header("Chatgpt-account-id") == f"account-{suffix}"
        if authorization == f"Bearer old-{suffix}":
            barrier.wait(timeout=10)
            raise HTTPError(req.full_url, 401, "Unauthorized", None, io.BytesIO())
        assert authorization == f"Bearer new-{suffix}"
        return _sse_response({"type": "response.output_text.delta", "delta": suffix})

    def run(suffix):
        with codex_auth_context(
            CodexAuthConfig(
                access_token_provider=lambda: (f"old-{suffix}", f"account-{suffix}"),
                rejected_access_token_provider=providers[suffix],
            )
        ):
            headers = _codex_headers(*_get_codex_access_token())
            return _stream_codex_response({}, headers)["text"]

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", open_response)
    with codex_auth_context(
        CodexAuthConfig(access_token_provider=lambda: ("outer", None))
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(run, ("a", "b"))) == ["a", "b"]
        assert _get_codex_access_token() == ("outer", None)
    for suffix, provider in providers.items():
        provider.assert_called_once_with(f"old-{suffix}")


@pytest.mark.parametrize(
    "failure",
    [
        {"type": "response.failed", "response": {"status": "failed"}},
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": "server_error", "message": "private details"},
            },
        },
        {"type": "error", "code": "invalid_prompt", "message": "private details"},
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
    ],
)
def test_stream_codex_response_rejects_terminal_failures(monkeypatch, failure):
    monkeypatch.setattr(
        "brrragent.codex_oauth.request.urlopen",
        lambda *_args, **_kwargs: _sse_response(
            {"type": "response.output_text.delta", "delta": "partial answer"},
            failure,
        ),
    )

    with pytest.raises(ValueError, match="Codex Responses stream") as exc:
        _stream_codex_response({}, {})

    assert failure["type"] in str(exc.value)
    assert "private details" not in str(exc.value)
    assert "partial answer" not in str(exc.value)


@pytest.fixture
def streamed_codex_agent(monkeypatch):
    monkeypatch.setattr(
        "brrragent.codex_oauth._get_codex_access_token", lambda: ("test-token", None)
    )
    mcp = Mock()
    mcp.get_openai_tools.return_value = []

    def run(max_turns):
        return run_codex_oauth_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4",
            mcp=mcp,
            extra_tools=None,
            max_turns=max_turns,
            temperature=0,
            max_tokens=100,
            max_retries=3,
            on_tool_call=None,
            response_schema=None,
        )

    return run


@pytest.mark.parametrize("max_turns", [0, 1])
@pytest.mark.parametrize(
    "code", ["server_error", "rate_limit_exceeded", "vector_store_timeout"]
)
def test_codex_agent_recovers_from_stream_failure(
    monkeypatch, streamed_codex_agent, max_turns, code
):
    urlopen = Mock(
        side_effect=[
            _sse_response(
                {
                    "type": "response.failed",
                    "response": {"error": {"code": code, "message": "private details"}},
                }
            ),
            _sse_response(
                {
                    "type": "response.completed",
                    "response": {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "recovered"}
                                ],
                            }
                        ]
                    },
                }
            ),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    monkeypatch.setattr("brrragent.codex_oauth.time.sleep", sleep)

    assert streamed_codex_agent(max_turns) == "recovered"
    assert urlopen.call_count == 2
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize("max_turns", [0, 1])
def test_codex_agent_retries_http_error_without_logging_provider_body(
    monkeypatch, streamed_codex_agent, max_turns, caplog
):
    secret = "private provider response"

    def fail_once(*_args, **_kwargs):
        fail_once.calls += 1
        if fail_once.calls == 1:
            raise HTTPError(
                "https://chatgpt.com/backend-api/codex/responses",
                503,
                "Service Unavailable",
                hdrs=None,
                fp=io.BytesIO(secret.encode()),
            )
        return _sse_response(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "recovered"}],
                        }
                    ]
                },
            }
        )

    fail_once.calls = 0
    sleep = Mock()
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fail_once)
    monkeypatch.setattr("brrragent.codex_oauth.time.sleep", sleep)

    assert streamed_codex_agent(max_turns) == "recovered"
    assert fail_once.calls == 2
    sleep.assert_called_once_with(2)
    assert "_CodexHTTPError (HTTP 503)" in caplog.text
    assert secret not in caplog.text


def test_refresh_access_token_omits_http_error_body(monkeypatch):
    secret = "private refresh response"

    def fail_urlopen(*_args, **_kwargs):
        raise HTTPError(
            "https://auth.openai.com/oauth/token",
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(secret.encode()),
        )

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="Codex OAuth refresh failed: HTTP 401") as exc:
        _refresh_access_token({"refresh": "test-refresh-token"})

    assert secret not in str(exc.value)


@pytest.mark.parametrize("max_turns", [0, 1])
@pytest.mark.parametrize("code,attempts", [("server_error", 3), ("invalid_prompt", 1)])
def test_codex_agent_raises_stream_failure_with_bounded_retries(
    monkeypatch, streamed_codex_agent, max_turns, code, attempts, caplog
):
    urlopen = Mock(
        side_effect=lambda *_args, **_kwargs: _sse_response(
            {"type": "error", "code": code, "message": "private details"}
        )
    )
    sleep = Mock()
    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", urlopen)
    monkeypatch.setattr("brrragent.codex_oauth.time.sleep", sleep)

    with pytest.raises(ValueError, match="Codex Responses stream"):
        streamed_codex_agent(max_turns)

    assert urlopen.call_count == attempts
    assert sleep.call_count == attempts - 1
    assert "private details" not in caplog.text


def test_stream_codex_response_reports_http_error(monkeypatch):
    secret = '{"error":"usage_limit_reached","detail":"private response"}'

    def fail_urlopen(_req, timeout):
        assert timeout == 31
        raise HTTPError(
            "https://chatgpt.com/backend-api/codex/responses",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(secret.encode()),
        )

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fail_urlopen)

    with pytest.raises(
        ValueError,
        match=r"Codex Responses failed: HTTP 429",
    ) as exc:
        _stream_codex_response({}, {}, timeout=31)

    assert secret not in str(exc.value)


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
    monkeypatch.setenv("BRRRAGENT_CODEX_HOME", str(tmp_path / "codex-home"))

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


def test_get_codex_access_token_refreshes_once_across_threads(tmp_path, monkeypatch):
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
    monkeypatch.setenv("BRRRAGENT_CODEX_HOME", str(tmp_path / "codex-home"))

    worker_count = 6
    start = threading.Barrier(worker_count)
    refresh_entered = threading.Event()
    allow_response = threading.Event()
    call_lock = threading.Lock()
    refresh_calls = 0

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

    def fake_urlopen(_req, timeout):
        nonlocal refresh_calls
        assert timeout == 60
        with call_lock:
            refresh_calls += 1
        refresh_entered.set()
        assert allow_response.wait(timeout=10)
        return Response()

    def load_auth():
        start.wait(timeout=10)
        return _get_codex_access_token()

    monkeypatch.setattr("brrragent.codex_oauth.request.urlopen", fake_urlopen)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(load_auth) for _ in range(worker_count)]
        try:
            assert refresh_entered.wait(timeout=10)
        finally:
            allow_response.set()
        results = [future.result(timeout=10) for future in futures]

    assert results == [("access_new", "acct_1")] * worker_count
    assert refresh_calls == 1
    saved = json.loads(auth_path.read_text())
    assert saved["openai"]["access"] == "access_new"
    assert saved["openai"]["refresh"] == "rt_new"
