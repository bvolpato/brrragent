import pytest

from brrragent import agent
from brrragent.codex_oauth import CodexAuthConfig
from brrragent.images import ImageInput
from brrragent.keys import KeyPool


class DummyMcp:
    def get_openai_tools(self):
        return []


def test_google_model_helpers():
    assert agent._is_google_model("google/gemini-3.1-pro-preview")
    assert agent._is_google_model("gemini-3.1-pro-preview")
    assert not agent._is_google_model("openai/gpt-5.4")
    assert agent._is_openai_model("openai/gpt-5.4")
    assert agent._is_codex_model("codex/gpt-5.4:xhigh")
    assert not agent._is_openai_model("google/gemini-3.1-pro-preview")
    assert (
        agent._get_gemini_model_name("google/gemini-3.1-pro-preview")
        == "gemini-3.1-pro-preview"
    )
    assert (
        agent._get_gemini_model_name("gemini-3.1-pro-preview")
        == "gemini-3.1-pro-preview"
    )


def test_resolve_native_provider_uses_env_key_and_strips_prefix(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")

    assert agent._resolve_native_provider(
        "fireworks/accounts/fireworks/models/test"
    ) == (
        "https://api.fireworks.ai/inference/v1",
        "fw-key",
        "fireworks",
        "accounts/fireworks/models/test",
    )


def test_resolve_native_provider_supports_openlux(monkeypatch):
    monkeypatch.setenv("OPENLUX_API_KEY", "openlux-key")
    monkeypatch.setenv("YUNWU_API_KEY", "legacy-key")

    assert agent._resolve_native_provider("openlux/gpt-5.4:high") == (
        "https://api.openlux.ai/v1",
        "openlux-key",
        "openlux",
        "gpt-5.4:high",
    )


def test_resolve_native_provider_supports_legacy_yunwu_alias(monkeypatch):
    monkeypatch.delenv("OPENLUX_API_KEY", raising=False)
    monkeypatch.setenv("YUNWU_API_KEY", "legacy-key")

    assert agent._resolve_native_provider("yunwu/gpt-5.4:high") == (
        "https://api.openlux.ai/v1",
        "legacy-key",
        "openlux",
        "gpt-5.4:high",
    )


def test_resolve_native_provider_falls_back_without_env_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    assert agent._resolve_native_provider("minimax/text-01") is None


def test_run_agent_routes_native_provider(monkeypatch):
    calls = []

    def fake_openrouter_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
    monkeypatch.setattr(
        "brrragent.openrouter.run_openrouter_agent", fake_openrouter_agent
    )

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="minimax/text-01",
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "text-01"
    assert calls[0]["api_key"] == "minimax-key"
    assert calls[0]["base_url"] == "https://api.minimax.io/v1"


@pytest.mark.parametrize(
    ("model", "env_var", "api_key"),
    [
        ("openlux/gpt-5.4:high", "OPENLUX_API_KEY", "openlux-key"),
        ("yunwu/gpt-5.4:high", "YUNWU_API_KEY", "legacy-key"),
    ],
)
def test_run_agent_routes_openlux_native_provider_aliases(
    monkeypatch, model, env_var, api_key
):
    calls = []

    def fake_openrouter_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.delenv("OPENLUX_API_KEY", raising=False)
    monkeypatch.delenv("YUNWU_API_KEY", raising=False)
    monkeypatch.setenv(env_var, api_key)
    monkeypatch.setattr(
        "brrragent.openrouter.run_openrouter_agent", fake_openrouter_agent
    )

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model=model,
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "gpt-5.4"
    assert calls[0]["api_key"] == api_key
    assert calls[0]["base_url"] == "https://api.openlux.ai/v1"
    assert calls[0]["reasoning_effort"] == "high"


def test_run_agent_routes_gemini_key_pool(monkeypatch):
    calls = []
    pool = KeyPool(["gemini-key"], name="gemini")

    def fake_gemini_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr("brrragent.gemini.run_gemini_agent", fake_gemini_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="google/gemini-3.1-pro-preview",
            gemini_key_pool=pool,
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "gemini-3.1-pro-preview"
    assert calls[0]["key_pool"] is pool


def test_run_agent_routes_openai_direct_from_env(monkeypatch):
    calls = []

    def fake_openai_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setenv("OPENAI_API_KEY_PERSONAL", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr("brrragent.openai_direct.run_openai_agent", fake_openai_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="openai/gpt-5.5:xhigh",
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "openai/gpt-5.5:xhigh"
    assert calls[0]["key_pool"].acquire() == "openai-key"


def test_run_agent_routes_codex_oauth_when_enabled(monkeypatch):
    calls = []

    def fake_codex_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setenv("BRRRAGENT_OPENAI_BACKEND", "codex_oauth")
    monkeypatch.setenv("OPENAI_API_KEY_PERSONAL", "openai-key")
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="openai/gpt-5.5:xhigh",
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "openai/gpt-5.5:xhigh"
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["user_prompt"] == "user"


def test_run_agent_routes_codex_prefix_without_env(monkeypatch):
    calls = []

    def fake_codex_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.delenv("BRRRAGENT_OPENAI_BACKEND", raising=False)
    monkeypatch.delenv("BRRRAGENT_CODEX_OAUTH", raising=False)
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4:xhigh",
            mcp=DummyMcp(),
        )
        == "ok"
    )

    assert calls[0]["model"] == "codex/gpt-5.4:xhigh"


def test_run_agent_passes_normalized_images_to_provider(monkeypatch):
    calls = []
    image = ImageInput.from_bytes(b"image", media_type="image/png", detail="low")

    def fake_codex_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.6-luna:medium",
            mcp=DummyMcp(),
            images=[image],
        )
        == "ok"
    )
    assert calls[0]["images"] == (image,)


def test_run_agent_rejects_untyped_image_values():
    with pytest.raises(TypeError, match="ImageInput"):
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.6-luna:medium",
            mcp=DummyMcp(),
            images=["data:image/png;base64,aW1hZ2U="],
        )


def test_run_agent_applies_per_call_codex_auth(monkeypatch, tmp_path):
    observed = {}

    def fake_codex_agent(**kwargs):
        from brrragent.codex_oauth import _auth_path, _codex_cli_auth_path

        observed["auth_path"] = _auth_path()
        observed["cli_auth_path"] = _codex_cli_auth_path()
        return "ok"

    auth_path = tmp_path / "auth.json"
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4:xhigh",
            mcp=DummyMcp(),
            codex_auth=CodexAuthConfig(
                auth_path=auth_path,
                codex_home=codex_home,
            ),
        )
        == "ok"
    )
    assert observed == {
        "auth_path": auth_path,
        "cli_auth_path": codex_home / "auth.json",
    }


def test_run_agent_preserves_existing_codex_auth_context(monkeypatch, tmp_path):
    from brrragent.codex_oauth import codex_auth_context

    observed = {}

    def fake_codex_agent(**kwargs):
        from brrragent.codex_oauth import _auth_path

        observed["auth_path"] = _auth_path()
        return "ok"

    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    with codex_auth_context(
        CodexAuthConfig(auth_path=auth_path, codex_home=tmp_path / "codex-home")
    ):
        assert (
            agent.run_agent(
                system_prompt="system",
                user_prompt="user",
                model="codex/gpt-5.4:xhigh",
                mcp=DummyMcp(),
            )
            == "ok"
        )
    assert observed["auth_path"] == auth_path


def test_run_agent_requires_api_key_when_no_pool(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_PERSONAL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_CORP", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("BRRRAGENT_OPENAI_BACKEND", raising=False)
    monkeypatch.delenv("BRRRAGENT_CODEX_OAUTH", raising=False)

    with pytest.raises(ValueError, match="No API key"):
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="openai/gpt-5.4",
            mcp=DummyMcp(),
        )
