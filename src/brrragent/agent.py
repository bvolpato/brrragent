"""
Agentic React Loop.

A reusable tool-calling loop that works with:
  - OpenRouter / OpenAI-compatible APIs   (model="google/gemini-3.1-pro-preview")
  - Google Gemini direct API              (model="google/..." + GEMINI_API_KEY set)

Smart routing: if the model starts with "google/" and GEMINI_API_KEY is set,
the agent uses the Google genai SDK directly (no OpenRouter premium). Otherwise
it falls back to OpenRouter with OPENROUTER_API_KEY.

API keys can be comma-separated — one is picked at random per session to
spread load and leverage per-key caching.

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

import json
import logging
import os
import random
import time
from typing import Callable, Optional

from brrragent.mcp_tools import McpToolCaller

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 12  # Safety valve — prevent runaway loops


def _pick_key(keys_csv: str) -> str:
    """Pick a random key from a comma-separated string of keys."""
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        raise ValueError("No API keys found in provided key string")
    chosen = random.choice(keys)
    logger.debug("[agent] Picked 1 key from %d available", len(keys))
    return chosen


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
        selected_key = _pick_key(gemini_keys_csv)
        bare_model = _get_gemini_model_name(model)
        logger.info("[agent] Using Gemini direct (model=%s)", bare_model)
        return _run_gemini_agent(
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
        # OpenRouter / OpenAI path
        if api_key is None:
            or_keys_csv = os.environ.get("OPENROUTER_API_KEY", "")
            if not or_keys_csv.strip():
                raise ValueError("No API key: set GEMINI_API_KEY or OPENROUTER_API_KEY")
            api_key = _pick_key(or_keys_csv)

        logger.info("[agent] Using OpenRouter (model=%s)", model)
        return _run_openrouter_agent(
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


# ══════════════════════════════════════════════════════════════════════
# OpenRouter / OpenAI-compatible backend
# ══════════════════════════════════════════════════════════════════════

def _run_openrouter_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    mcp: McpToolCaller,
    extra_tools: list[dict] | None,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
) -> str:
    """Run the agent using an OpenAI-compatible API (OpenRouter, OpenAI, etc.)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        default_headers={"X-Title": "brrragent"},
    )

    # Build tool definitions
    tools = mcp.get_openai_tools()
    if extra_tools:
        tools.extend(extra_tools)

    # Build conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Structured output via response_format
    response_format = None
    if response_schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_response",
                "strict": True,
                "schema": response_schema,
            },
        }

    for turn in range(max_turns):
        logger.info("[agent] Turn %d/%d — calling %s (openrouter)", turn + 1, max_turns, model)

        response = _or_call_with_retry(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            response_format=response_format,
        )

        choice = response.choices[0]
        message = choice.message

        # Add assistant message to history
        messages.append(message.model_dump())

        # Check if the model wants to call tools
        if message.tool_calls:
            logger.info("[agent] Turn %d: %d tool call(s)", turn + 1, len(message.tool_calls))

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("[agent] Tool: %s(%s)", fn_name, list(fn_args.keys()))
                if on_tool_call:
                    on_tool_call(fn_name, fn_args)

                result_text = mcp.call_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
        else:
            final_text = message.content or ""
            logger.info("[agent] Final response after %d turn(s) (%d chars)", turn + 1, len(final_text))
            return final_text

        if choice.finish_reason == "stop":
            return message.content or ""

    logger.warning("[agent] Reached max_turns=%d without final response", max_turns)
    last = messages[-1]
    if isinstance(last, dict):
        return last.get("content", "[No final response after max tool turns]")
    return getattr(last, "content", "[No final response after max tool turns]") or ""


def _or_call_with_retry(
    *,
    client,
    model: str,
    messages: list,
    tools: list,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    response_format: dict | None = None,
):
    """Call the OpenAI-compatible API with exponential backoff on transient errors."""
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={"X-Title": "brrragent"},
            )
            if response_format:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_transient = any(
                p in err_str
                for p in ("500", "503", "429", "unavailable", "overloaded", "rate")
            )

            logger.warning("[agent] API call attempt %d/%d failed: %s", attempt, max_retries, str(e)[:200])

            if is_transient and attempt < max_retries:
                wait = min(2 ** attempt, 30)
                logger.info("[agent] Transient error, retrying in %ds...", wait)
                time.sleep(wait)
            elif not is_transient:
                raise

    raise ValueError(f"API call failed after {max_retries} retries: {last_err}")


# ══════════════════════════════════════════════════════════════════════
# Google Gemini direct backend
# ══════════════════════════════════════════════════════════════════════

def _run_gemini_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    mcp: McpToolCaller,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
) -> str:
    """Run the agent using the Google genai SDK directly (no OpenRouter)."""
    try:
        from google import genai
        from google.genai import types as gt
    except ImportError as e:
        raise ImportError("google-genai required for Gemini direct: pip install google-genai") from e

    client = genai.Client(api_key=api_key)
    tools = mcp.get_genai_tools()

    # Build conversation
    contents: list[gt.Content] = [
        gt.Content(role="user", parts=[gt.Part(text=user_prompt)])
    ]

    # Config
    config_kwargs = dict(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=tools,
    )

    # Structured output via response_schema
    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = gt.GenerateContentConfig(**config_kwargs)

    for turn in range(max_turns):
        logger.info("[agent] Turn %d/%d — calling %s (gemini direct)", turn + 1, max_turns, model)

        response = _gemini_call_with_retry(
            client=client,
            model=model,
            contents=contents,
            config=config,
            max_retries=max_retries,
        )

        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            break

        model_parts = list(candidate.content.parts or [])

        # Check for function calls
        fn_calls = [p for p in model_parts if p.function_call is not None]

        if not fn_calls:
            # Model finished — extract text
            text_parts = [p.text for p in model_parts if p.text]
            final_text = "\n".join(text_parts)
            logger.info("[agent] Final response after %d turn(s) (%d chars)", turn + 1, len(final_text))
            return final_text

        # Add model's response to history
        contents.append(gt.Content(role="model", parts=model_parts))

        # Execute each tool call and gather results
        tool_result_parts = []
        for fc in fn_calls:
            fn_name = fc.function_call.name
            fn_args = dict(fc.function_call.args or {})
            logger.info("[agent] Tool: %s(%s)", fn_name, list(fn_args.keys()))
            if on_tool_call:
                on_tool_call(fn_name, fn_args)

            result_text = mcp.call_tool(fn_name, fn_args)
            tool_result_parts.append(
                gt.Part(
                    function_response=gt.FunctionResponse(
                        name=fn_name,
                        response={"result": result_text},
                    )
                )
            )

        contents.append(gt.Content(role="user", parts=tool_result_parts))
        logger.debug("[agent] Turn %d: executed %d tool(s)", turn + 1, len(fn_calls))

    logger.warning("[agent] Reached max_turns=%d without final text", max_turns)
    return "[No final response after max tool turns]"


def _gemini_call_with_retry(
    *,
    client,
    model: str,
    contents,
    config,
    max_retries: int,
):
    """Call the Gemini API with exponential backoff on transient errors."""
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_transient = any(
                p in err_str
                for p in ("500", "503", "429", "UNAVAILABLE", "INTERNAL", "overloaded", "rate")
            )

            logger.warning("[agent] Gemini call attempt %d/%d failed: %s", attempt, max_retries, err_str[:200])

            if is_transient and attempt < max_retries:
                wait = min(2 ** attempt, 30)
                logger.info("[agent] Transient error, retrying in %ds...", wait)
                time.sleep(wait)
            elif not is_transient:
                raise

    raise ValueError(f"Gemini API call failed after {max_retries} retries: {last_err}")
