"""
Agentic React Loop — Smart Router.

Automatically chooses between Google Gemini direct and OpenRouter based on
the model name and available API keys.

Routing logic:
  - model starts with "google/" or "gemini-" + GEMINI_API_KEY set → Gemini direct
  - Otherwise → OpenRouter with OPENROUTER_API_KEY

API keys (OPENROUTER_API_KEY, GEMINI_API_KEY) can be comma-separated.
One key is picked at random per call to spread load and leverage caching.

Usage:
    from brrragent import McpToolCaller, run_agent

    mcp = McpToolCaller()
    result = run_agent(
        system_prompt="You are a helpful assistant.",
        user_prompt="Search for the latest AI news.",
        model="google/gemini-3.1-pro-preview",
        mcp=mcp,
    )
"""

import logging
import os
from typing import Callable, Optional

from brrragent.keys import pick_key
from brrragent.mcp_tools import McpToolCaller

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 12  # Safety valve — prevent runaway loops


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
      - If model starts with "google/" and GEMINI_API_KEY is set → Gemini direct
      - Otherwise → OpenRouter with OPENROUTER_API_KEY

    API keys (OPENROUTER_API_KEY, GEMINI_API_KEY) can be comma-separated.
    One key is picked at random per call to spread load and leverage caching.

    Args:
        system_prompt: System instructions for the agent.
        user_prompt: The user's request / task.
        model: Model identifier (e.g. "google/gemini-3.1-pro-preview").
        api_key: Override API key (skips smart routing).
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
    """
    if mcp is None:
        from brrragent.mcp_tools import get_default_caller
        mcp = get_default_caller()

    # Smart routing: decide which backend to use
    gemini_keys_csv = os.environ.get("GEMINI_API_KEY", "")
    use_gemini_direct = (
        _is_google_model(model)
        and gemini_keys_csv.strip()
        and api_key is None  # If caller passed explicit key, respect it
    )

    if use_gemini_direct:
        from brrragent.gemini import run_gemini_agent

        selected_key = pick_key(gemini_keys_csv)
        bare_model = _get_gemini_model_name(model)
        logger.info("[agent] Using Gemini direct (model=%s)", bare_model)
        return run_gemini_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=bare_model,
            api_key=selected_key,
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

        if api_key is None:
            or_keys_csv = os.environ.get("OPENROUTER_API_KEY", "")
            if not or_keys_csv.strip():
                raise ValueError("No API key: set GEMINI_API_KEY or OPENROUTER_API_KEY")
            api_key = pick_key(or_keys_csv)

        logger.info("[agent] Using OpenRouter (model=%s)", model)
        return run_openrouter_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
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
