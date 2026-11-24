"""
Agentic React Loop — Smart Router.

Automatically chooses between Google Gemini direct, OpenAI direct,
Codex OAuth, native provider APIs, and OpenRouter based on the model name and
available keys.

Routing logic:
  - model starts with "google/" or "gemini-" + Gemini keys available → Gemini direct
  - model starts with "fireworks/" + FIREWORKS_API_KEY → Fireworks native
  - model starts with "minimax/" + MINIMAX_API_KEY → MiniMax native
  - model starts with "openlux/" + OPENLUX_API_KEY → OpenLux native
  - legacy "yunwu/" models use the same OpenLux native route
  - model starts with "z-ai/" + ZAI_API_KEY → Z.AI native
  - model starts with "codex/" → Codex OAuth
  - model starts with "openai/" + BRRRAGENT_OPENAI_BACKEND=codex_oauth → Codex OAuth
  - model starts with "openai/" + OpenAI keys available → OpenAI direct
  - Otherwise → OpenRouter with OPENROUTER_API_KEY

Supports KeyPool for both backends: keys are acquired per-call and evicted
on 429 errors.  If no pool is provided, falls back to env-var key picking.

Usage:
    import os

    from brrragent import McpToolCaller, run_agent, KeyPool

    gemini_pool = KeyPool.from_csv("key1,key2,key3", name="gemini")
    or_pool = KeyPool.from_csv("sk-or-1,sk-or-2", name="openrouter")

    result = run_agent(
        system_prompt="You are a helpful assistant.",
        user_prompt="Complete the requested task.",
        model="google/gemini-3.1-pro-preview",
        gemini_key_pool=gemini_pool,
        openrouter_key_pool=or_pool,
        mcp=McpToolCaller(endpoint=os.environ["BRRRAGENT_MCP_URL"]),
    )
"""

import logging
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from brrragent.images import ImageInput
from brrragent.keys import KeyPool
from brrragent.mcp_tools import McpToolCaller
from brrragent.prompt_cache import AgentUsage, PromptCacheConfig

if TYPE_CHECKING:
    from brrragent.codex_oauth import CodexAuthConfig

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 12

# Native provider registry: prefix → (base_url, env_var_name)
# Models with these prefixes are routed directly to the provider's
# OpenAI-compatible API instead of going through OpenRouter.
NATIVE_PROVIDERS: dict[str, tuple[str, str]] = {
    "fireworks/": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "minimax/": ("https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    "openlux/": ("https://api.openlux.ai/v1", "OPENLUX_API_KEY"),
    "yunwu/": ("https://api.openlux.ai/v1", "OPENLUX_API_KEY"),
    "z-ai/": ("https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
}


class _NoopMcpCaller:
    """Fallback tool caller used when no MCP server is configured."""

    def get_openai_tools(self) -> list[dict]:
        return []

    def get_genai_tools(self) -> list:
        return []

    def call_tool(self, name: str, arguments: dict) -> str:
        del arguments
        return f"[Tool error: unknown MCP tool {name!r}]"


def _is_google_model(model: str) -> bool:
    """Check if a model identifier is a Google model."""
    return model.startswith(("google/", "gemini-"))


def _is_openai_model(model: str) -> bool:
    """Check if a model identifier should use OpenAI direct when keys exist."""
    return model.startswith("openai/")


def _is_codex_model(model: str) -> bool:
    return model.startswith("codex/")


def _use_codex_oauth_backend(model: str, api_key: str | None) -> bool:
    if api_key is not None:
        return False
    if _is_codex_model(model):
        return True
    if not _is_openai_model(model):
        return False
    backend = os.environ.get("BRRRAGENT_OPENAI_BACKEND", "").strip().lower()
    enabled = os.environ.get("BRRRAGENT_CODEX_OAUTH", "").strip().lower()
    return backend in {"codex", "codex_oauth"} or enabled in {"1", "true", "yes", "on"}


def _get_gemini_model_name(model: str) -> str:
    """Strip 'google/' prefix to get the bare Gemini model name."""
    if model.startswith("google/"):
        return model[len("google/") :]
    return model


def _native_provider_api_key(env_var: str) -> str:
    api_key = os.environ.get(env_var, "").strip()
    if api_key:
        return api_key
    if env_var == "OPENLUX_API_KEY":
        return os.environ.get("YUNWU_API_KEY", "").strip()
    return ""


def _resolve_native_provider(model: str) -> tuple[str, str, str, str] | None:
    """Check if the model matches a native provider prefix.

    Returns (base_url, api_key, provider_name, bare_model) if a native route
    is available, or None to fall through to OpenRouter.

    The bare_model has the provider prefix stripped — native APIs expect the
    model ID without it (e.g. "accounts/fireworks/routers/kimi-k2p5-turbo"
    not "fireworks/accounts/fireworks/routers/kimi-k2p5-turbo"). Legacy Yunwu
    routes are reported as OpenLux after alias resolution.
    """
    for prefix, (base_url, env_var) in NATIVE_PROVIDERS.items():
        if model.startswith(prefix):
            api_key = _native_provider_api_key(env_var)
            if api_key:
                if prefix == "z-ai/":
                    base_url = os.environ.get("ZAI_BASE_URL", "").strip() or base_url
                provider_name = (
                    "openlux"
                    if prefix in {"openlux/", "yunwu/"}
                    else prefix.rstrip("/")
                )
                bare_model = model[len(prefix) :]
                return base_url, api_key, provider_name, bare_model
            logger.debug(
                "[agent] Model %s matches %s but %s is not set — falling through to OpenRouter",
                model,
                prefix,
                env_var,
            )
    return None


def _keys_from_envs(*env_names: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for env_name in env_names:
        raw = os.environ.get(env_name, "")
        for key in raw.split(","):
            key = key.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def run_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "google/gemini-3.1-pro-preview",
    api_key: str | None = None,
    gemini_key_pool: KeyPool | None = None,
    openai_key_pool: KeyPool | None = None,
    openrouter_key_pool: KeyPool | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    mcp: McpToolCaller | None = None,
    extra_tools: list[dict] | None = None,
    max_turns: int = MAX_TOOL_TURNS,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
    on_tool_call: Callable | None = None,
    response_schema: dict | None = None,
    prompt_cache: PromptCacheConfig | None = None,
    on_usage: Callable[[AgentUsage], None] | None = None,
    codex_auth: "CodexAuthConfig | None" = None,
    images: Sequence[ImageInput] | None = None,
) -> str:
    """
    Run an agentic react loop with tool calling.

    Smart routing:
      - If model starts with "google/" and Gemini keys are available → Gemini direct
      - If model starts with "codex/" → Codex OAuth
      - If model starts with "openai/" and BRRRAGENT_OPENAI_BACKEND=codex_oauth → Codex OAuth
      - If model starts with "openai/" and OpenAI keys are available → OpenAI direct
      - If model starts with a native provider prefix (fireworks/, minimax/, openlux/)
        and
        the corresponding env var is set → provider's native API
      - Otherwise → OpenRouter with OPENROUTER_API_KEY

    Key pools (preferred) or env vars can provide the keys.
    On 429, pools evict the bad key and rotate to a fresh one.
    RateLimitExhausted propagates up if ALL keys are exhausted.

    Args:
        system_prompt: System instructions for the agent.
        user_prompt: The user's request / task.
        model: Model identifier (e.g. "google/gemini-3.1-pro-preview").
        api_key: Override API key (skips smart routing, legacy).
        gemini_key_pool: KeyPool for Gemini keys (preferred over env var).
        openai_key_pool: KeyPool for direct OpenAI keys (preferred over env var).
        openrouter_key_pool: KeyPool for OpenRouter keys (preferred over env var).
        base_url: API base URL (only used for OpenRouter path).
        mcp: McpToolCaller instance. Uses the default env-configured caller when
            BRRRAGENT_MCP_URL(S) is set; otherwise runs without MCP tools.
        extra_tools: Additional tool definitions (OpenAI format).
        max_turns: Maximum number of tool-calling rounds.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        max_retries: Maximum attempts per API call on transient errors. Provider
            SDK retries are disabled so this budget is authoritative.
        on_tool_call: Optional callback(tool_name, tool_args) for logging/UI.
        response_schema: JSON Schema dict for structured output (optional).
        prompt_cache: Stable cache key and requested TTL for this prompt version.
        on_usage: Optional callback invoked with provider token and cache usage.
        codex_auth: Per-call Codex profile, file paths, or access token provider.
        images: Image URLs, Base64 data URLs, bytes, or files created as ImageInput.

    Returns:
        The model's final text response.

    Raises:
        ValueError: If all retries are exhausted without a response.
        RateLimitExhausted: If all keys in a pool hit 429.
    """
    if isinstance(max_turns, bool) or max_turns < 0:
        raise ValueError("max_turns must be at least 0")
    if isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if api_key is not None and not api_key.strip():
        raise ValueError("api_key must not be empty")

    image_inputs = tuple(images or ())
    if any(not isinstance(image, ImageInput) for image in image_inputs):
        raise TypeError("images must contain ImageInput values")

    if mcp is None:
        from brrragent.mcp_tools import (
            get_default_caller,
            has_configured_default_caller,
        )

        mcp = (
            get_default_caller()
            if has_configured_default_caller()
            else _NoopMcpCaller()
        )

    # Smart routing: only resolve env pools needed for the matched route
    if api_key is None and _is_google_model(model):
        if gemini_key_pool is None:
            gemini_csv = os.environ.get("GEMINI_API_KEY", "")
            if gemini_csv.strip():
                gemini_key_pool = KeyPool.from_csv(gemini_csv, name="gemini")

        if gemini_key_pool is not None:
            from brrragent.gemini import run_gemini_agent

            bare_model = _get_gemini_model_name(model)
            logger.info(
                "[agent] Using Gemini direct (model=%s, pool=%s)",
                bare_model,
                gemini_key_pool.status(),
            )
            return run_gemini_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=bare_model,
                key_pool=gemini_key_pool,
                mcp=mcp,
                extra_tools=extra_tools,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                on_tool_call=on_tool_call,
                response_schema=response_schema,
                prompt_cache=prompt_cache,
                on_usage=on_usage,
                images=image_inputs,
            )

    if _use_codex_oauth_backend(model, api_key):
        from brrragent.codex_oauth import codex_auth_context, run_codex_oauth_agent

        logger.info("[agent] Using Codex OAuth (model=%s)", model)
        kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": model,
            "mcp": mcp,
            "extra_tools": extra_tools,
            "max_turns": max_turns,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "on_tool_call": on_tool_call,
            "response_schema": response_schema,
            "prompt_cache": prompt_cache,
            "on_usage": on_usage,
            "images": image_inputs,
        }
        if codex_auth is None:
            return run_codex_oauth_agent(**kwargs)
        with codex_auth_context(codex_auth):
            return run_codex_oauth_agent(**kwargs)

    if api_key is None and _is_openai_model(model):
        if openai_key_pool is None:
            openai_keys = _keys_from_envs(
                "OPENAI_API_KEY_PERSONAL",
                "OPENAI_API_KEY",
                "OPENAI_API_KEY_CORP",
            )
            if openai_keys:
                openai_key_pool = KeyPool(openai_keys, name="openai")

        if openai_key_pool is not None:
            from brrragent.openai_direct import run_openai_agent

            logger.info(
                "[agent] Using OpenAI direct (model=%s, pool=%s)",
                model,
                openai_key_pool.status(),
            )
            return run_openai_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                key_pool=openai_key_pool,
                mcp=mcp,
                extra_tools=extra_tools,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                on_tool_call=on_tool_call,
                response_schema=response_schema,
                prompt_cache=prompt_cache,
                on_usage=on_usage,
                images=image_inputs,
            )

    # Check for native provider routing (fireworks/, minimax/, etc.)
    native = _resolve_native_provider(model) if api_key is None else None

    if native is not None:
        from brrragent.openai_direct import parse_openai_model
        from brrragent.openrouter import run_openrouter_agent

        native_base_url, native_api_key, provider_name, bare_model = native
        reasoning_effort = None
        if provider_name in {"openlux", "z-ai"}:
            bare_model, reasoning_effort = parse_openai_model(bare_model)
        if provider_name == "z-ai" and not reasoning_effort:
            reasoning_effort = (
                os.environ.get("ZAI_REASONING_EFFORT", "xhigh").strip() or "xhigh"
            )
        logger.info("[agent] Using %s native (model=%s)", provider_name, bare_model)
        return run_openrouter_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=bare_model,
            api_key=native_api_key,
            key_pool=None,
            base_url=native_base_url,
            mcp=mcp,
            extra_tools=extra_tools,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            on_tool_call=on_tool_call,
            response_schema=response_schema,
            reasoning_effort=reasoning_effort,
            prompt_cache=None,
            on_usage=on_usage,
            images=image_inputs,
        )

    # Fallback: OpenRouter
    from brrragent.openrouter import run_openrouter_agent

    if api_key is None and openrouter_key_pool is None:
        or_csv = os.environ.get("OPENROUTER_API_KEY", "")
        if or_csv.strip():
            openrouter_key_pool = KeyPool.from_csv(or_csv, name="openrouter")

    effective_key = api_key
    effective_pool = None if api_key is not None else openrouter_key_pool

    if effective_key is None and effective_pool is None:
        raise ValueError(
            "No API key: set GEMINI_API_KEY or OPENROUTER_API_KEY, or provide a key pool"
        )

    logger.info(
        "[agent] Using OpenRouter (model=%s, pool=%s)",
        model,
        effective_pool.status() if effective_pool else "explicit-key",
    )
    return run_openrouter_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=effective_key,
        key_pool=effective_pool,
        base_url=base_url,
        mcp=mcp,
        extra_tools=extra_tools,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        on_tool_call=on_tool_call,
        response_schema=response_schema,
        prompt_cache=prompt_cache,
        on_usage=on_usage,
        images=image_inputs,
    )
