import pytest

from brrragent import agent
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
    assert agent._get_gemini_model_name("google/gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
    assert agent._get_gemini_model_name("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"


def test_resolve_native_provider_uses_env_key_and_strips_prefix(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")

    assert agent._resolve_native_provider("fireworks/accounts/fireworks/models/test") == (
        "https://api.fireworks.ai/inference/v1",
        "fw-key",
        "fireworks",
        "accounts/fireworks/models/test",
    )


def test_resolve_native_provider_supports_yunwu(monkeypatch):
    monkeypatch.setenv("YUNWU_API_KEY", "yunwu-key")

    assert agent._resolve_native_provider("yunwu/gpt-5.4:high") == (
        "https://yunwu.ai/v1",
        "yunwu-key",
        "yunwu",
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
    monkeypatch.setattr("brrragent.openrouter.run_openrouter_agent", fake_openrouter_agent)

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="minimax/text-01",
        mcp=DummyMcp(),
    ) == "ok"

    assert calls[0]["model"] == "text-01"
    assert calls[0]["api_key"] == "minimax-key"
    assert calls[0]["base_url"] == "https://api.minimax.io/v1"


def test_run_agent_routes_yunwu_native_provider(monkeypatch):
    calls = []

    def fake_openrouter_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setenv("YUNWU_API_KEY", "yunwu-key")
    monkeypatch.setattr("brrragent.openrouter.run_openrouter_agent", fake_openrouter_agent)

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="yunwu/gpt-5.4:high",
        mcp=DummyMcp(),
    ) == "ok"

    assert calls[0]["model"] == "gpt-5.4"
    assert calls[0]["api_key"] == "yunwu-key"
    assert calls[0]["base_url"] == "https://yunwu.ai/v1"
    assert calls[0]["reasoning_effort"] == "high"


def test_run_agent_routes_gemini_key_pool(monkeypatch):
    calls = []
    pool = KeyPool(["gemini-key"], name="gemini")

    def fake_gemini_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr("brrragent.gemini.run_gemini_agent", fake_gemini_agent)

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="google/gemini-3.1-pro-preview",
        gemini_key_pool=pool,
        mcp=DummyMcp(),
    ) == "ok"

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

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="openai/gpt-5.5:xhigh",
        mcp=DummyMcp(),
    ) == "ok"

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

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="openai/gpt-5.5:xhigh",
        mcp=DummyMcp(),
    ) == "ok"

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

    assert agent.run_agent(
        system_prompt="system",
        user_prompt="user",
        model="codex/gpt-5.4:xhigh",
        mcp=DummyMcp(),
    ) == "ok"

    assert calls[0]["model"] == "codex/gpt-5.4:xhigh"


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
