"""
Agentic React Loop — Smart Router.

Automatically chooses between Google Gemini direct and OpenRouter based on
the model name and available API keys.

Routing logic:
  - model starts with "google/" or "gemini-" + Gemini keys available → Gemini direct
  - Otherwise → OpenRouter with OPENROUTER_API_KEY

Supports KeyPool for both backends: keys are acquired per-call and evicted
on 429 errors.  If no pool is provided, falls back to env-var key picking.

Usage:
    from brrragent import McpToolCaller, run_agent, KeyPool

    gemini_pool = KeyPool.from_csv("key1,key2,key3", name="gemini")
    or_pool = KeyPool.from_csv("sk-or-1,sk-or-2", name="openrouter")

    result = run_agent(
        system_prompt="You are a helpful assistant.",
        user_prompt="Search for the latest AI news.",
        model="google/gemini-3.1-pro-preview",
        gemini_key_pool=gemini_pool,
        openrouter_key_pool=or_pool,
        mcp=McpToolCaller(),
    )
"""

import logging
import os
from typing import Callable, Optional

from brrragent.keys import pick_key, KeyPool
from brrragent.mcp_tools import McpToolCaller

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 12


def _is_google_model(model: str) -> bool:
    """Check if a model identifier is a Google model."""
    return model.startswith("google/") or model.startswith("gemini-")


def _get_gemini_model_name(model: str) -> str:
    """Strip 'google/' prefix to get the bare Gemini model name."""
    if model.startswith("google/"):
        return model[len("google/"):]
    return model


def run_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "google/gemini-3.1-pro-preview",
    api_key: str | None = None,
    gemini_key_pool: KeyPool | None = None,
    openrouter_key_pool: KeyPool | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    mcp: McpToolCaller | None = None,
    extra_tools: list[dict] | None = None,
    max_turns: int = MAX_TOOL_TURNS,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
    on_tool_call: Optional[Callable] = None,
    response_schema: dict | None = None,
) -> str:
    """
    Run an agentic react loop with tool calling.

    Smart routing:
      - If model starts with "google/" and Gemini keys are available → Gemini direct
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
        openrouter_key_pool: KeyPool for OpenRouter keys (preferred over env var).
        base_url: API base URL (only used for OpenRouter path).
        mcp: McpToolCaller instance. Defaults to the shared singleton.
        extra_tools: Additional tool definitions (OpenAI format).
        max_turns: Maximum number of tool-calling rounds.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        max_retries: Retries per API call on transient errors.
        on_tool_call: Optional callback(tool_name, tool_args) for logging/UI.
        response_schema: JSON Schema dict for structured output (optional).

    Returns:
        The model's final text response.

    Raises:
        ValueError: If all retries are exhausted without a response.
        RateLimitExhausted: If all keys in a pool hit 429.
    """
    if mcp is None:
        from brrragent.mcp_tools import get_default_caller
        mcp = get_default_caller()

    # Build pools from env vars if not explicitly provided
    if gemini_key_pool is None:
        gemini_csv = os.environ.get("GEMINI_API_KEY", "")
        if gemini_csv.strip():
            gemini_key_pool = KeyPool.from_csv(gemini_csv, name="gemini")

    if openrouter_key_pool is None:
        or_csv = os.environ.get("OPENROUTER_API_KEY", "")
        if or_csv.strip():
            openrouter_key_pool = KeyPool.from_csv(or_csv, name="openrouter")

    # Smart routing: decide which backend to use
    use_gemini_direct = (
        _is_google_model(model)
        and gemini_key_pool is not None
        and api_key is None
    )

    if use_gemini_direct:
        from brrragent.gemini import run_gemini_agent

        bare_model = _get_gemini_model_name(model)
        logger.info("[agent] Using Gemini direct (model=%s, pool=%s)", bare_model, gemini_key_pool.status())
        return run_gemini_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=bare_model,
            key_pool=gemini_key_pool,
            mcp=mcp,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            on_tool_call=on_tool_call,
            response_schema=response_schema,
        )
    else:
        from brrragent.openrouter import run_openrouter_agent

        # Determine key source for OpenRouter
        effective_key = api_key
        effective_pool = None if api_key else openrouter_key_pool

        if effective_key is None and effective_pool is None:
            raise ValueError("No API key: set GEMINI_API_KEY or OPENROUTER_API_KEY, or provide a key pool")

        logger.info(
            "[agent] Using OpenRouter (model=%s, pool=%s)",
            model, effective_pool.status() if effective_pool else "explicit-key",
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
        )
