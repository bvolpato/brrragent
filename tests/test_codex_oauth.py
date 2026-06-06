import json

from brrragent.codex_oauth import (
    _build_payload,
    _extract_function_calls,
    _extract_response_text,
    _get_codex_access_token,
    has_opencode_oauth,
    parse_codex_model,
)


def test_parse_codex_model_strips_prefix_and_reasoning_suffix():
    assert parse_codex_model("codex/gpt-5.4:xhigh") == ("gpt-5.4", "xhigh")
    assert parse_codex_model("openai/gpt-5.4:high") == ("gpt-5.4", "high")


def test_build_payload_matches_codex_backend_constraints():
    payload = _build_payload(
        model="codex/gpt-5.4:xhigh",
        instructions="system",
        input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
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
        input_items=[{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
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
    auth_path.write_text(json.dumps({"openai": {"type": "oauth", "refresh": "rt_test"}}))
    monkeypatch.setenv("BRRRAGENT_OPENCODE_AUTH_PATH", str(auth_path))

    assert has_opencode_oauth()


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
