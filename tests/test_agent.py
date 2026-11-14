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


def test_run_agent_uses_external_codex_token_provider(monkeypatch):
    observed = []

    def fake_codex_agent(**kwargs):
        from brrragent.codex_oauth import _get_codex_access_token

        observed.append(_get_codex_access_token())
        return "ok"

    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4:xhigh",
            mcp=DummyMcp(),
            codex_auth=CodexAuthConfig(
                access_token_provider=lambda: ("shared-access", "shared-account")
            ),
        )
        == "ok"
    )
    assert observed == [("shared-access", "shared-account")]


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


def test_run_agent_without_mcp_or_env_uses_noop_caller(monkeypatch):
    from brrragent import mcp_tools

    observed = {}

    def fake_codex_agent(**kwargs):
        mcp = kwargs["mcp"]
        observed["openai_tools"] = mcp.get_openai_tools()
        observed["genai_tools"] = mcp.get_genai_tools()
        observed["call_result"] = mcp.call_tool("missing", {"q": "x"})
        return "ok"

    monkeypatch.delenv("BRRRAGENT_MCP_URL", raising=False)
    monkeypatch.delenv("BRRRAGENT_MCP_URLS", raising=False)
    monkeypatch.setattr(mcp_tools, "_default_caller", None)
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4",
        )
        == "ok"
    )
    assert observed == {
        "openai_tools": [],
        "genai_tools": [],
        "call_result": "[Tool error: unknown MCP tool 'missing']",
    }


def test_run_agent_uses_default_caller_when_mcp_env_set(monkeypatch):
    from brrragent import mcp_tools

    sentinel = DummyMcp()
    observed = {}

    def fake_codex_agent(**kwargs):
        observed["mcp"] = kwargs["mcp"]
        return "ok"

    monkeypatch.setenv("BRRRAGENT_MCP_URL", "https://mcp.example.test")
    monkeypatch.setattr(mcp_tools, "_default_caller", None)
    monkeypatch.setattr(mcp_tools, "get_default_caller", lambda: sentinel)
    monkeypatch.setattr("brrragent.codex_oauth.run_codex_oauth_agent", fake_codex_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4",
        )
        == "ok"
    )
    assert observed["mcp"] is sentinel


def test_resolve_native_provider_reads_zai_base_url_at_call_time(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    assert agent._resolve_native_provider("z-ai/glm-5") == (
        "https://api.z.ai/api/coding/paas/v4",
        "zai-key",
        "z-ai",
        "glm-5",
    )

    monkeypatch.setenv("ZAI_BASE_URL", "   ")
    assert agent._resolve_native_provider("z-ai/glm-5") == (
        "https://api.z.ai/api/coding/paas/v4",
        "zai-key",
        "z-ai",
        "glm-5",
    )

    monkeypatch.setenv("ZAI_BASE_URL", "https://custom.z.ai/v4")
    assert agent._resolve_native_provider("z-ai/glm-5") == (
        "https://custom.z.ai/v4",
        "zai-key",
        "z-ai",
        "glm-5",
    )


def test_run_agent_forwards_extra_tools_to_gemini_direct(monkeypatch):
    calls = []
    extra = [{"type": "function", "function": {"name": "local_lookup"}}]

    def fake_gemini_agent(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr("brrragent.gemini.run_gemini_agent", fake_gemini_agent)

    assert (
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="google/gemini-3.1-pro-preview",
            gemini_key_pool=KeyPool(["gemini-key"], name="gemini"),
            mcp=DummyMcp(),
            extra_tools=extra,
        )
        == "ok"
    )
    assert calls[0]["extra_tools"] is extra


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_retries": 0}, "max_retries"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": False}, "max_retries"),
        ({"max_turns": -1}, "max_turns"),
        ({"max_turns": True}, "max_turns"),
    ],
)
def test_run_agent_rejects_invalid_turn_and_retry_budgets(kwargs, match):
    with pytest.raises(ValueError, match=match):
        agent.run_agent(
            system_prompt="system",
            user_prompt="user",
            model="codex/gpt-5.4",
            mcp=DummyMcp(),
            **kwargs,
        )


def test_public_api_docs_and_py_typed_marker():
    import re
    from pathlib import Path

    import brrragent

    repo_root = Path(__file__).resolve().parents[1]
    assert (Path(brrragent.__file__).resolve().parent / "py.typed").is_file()

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"## Public API\s+```python\s+from brrragent import \((.*?)\)\s+```",
        readme,
        re.DOTALL,
    )
    assert match is not None
    documented = {item.strip() for item in match.group(1).split(",") if item.strip()}
    assert documented == set(brrragent.__all__)
