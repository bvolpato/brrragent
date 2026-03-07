"""
Agentic React Loop.

A reusable tool-calling loop that works with any OpenAI-compatible API
(OpenRouter, OpenAI, etc.) and MCP tools.

Usage:
    from brrragent import McpToolCaller, run_agent

    mcp = McpToolCaller()
    result = run_agent(
        system_prompt="You are a helpful assistant.",
        user_prompt="Search for the latest AI news.",
        model="google/gemini-3.1-pro-preview",
        api_key=os.environ["OPENROUTER_API_KEY"],
        mcp=mcp,
    )
"""

import json
import logging
import os
import time

from openai import OpenAI

from brrragent.mcp_tools import McpToolCaller

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 12  # Safety valve — prevent runaway loops


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
    on_tool_call: callable | None = None,
) -> str:
    """
    Run an agentic react loop with tool calling.

    Implements the standard OpenAI function-calling loop:
      1. Send messages to model with tool definitions.
      2. If the model returns tool_calls, execute them via MCP.
      3. Feed results back and continue until the model returns content.
      4. On transient errors (500/503), retry with backoff.

    Args:
        system_prompt: System instructions for the agent.
        user_prompt: The user's request / task.
        model: Model identifier (OpenRouter format, e.g. "google/gemini-3.1-pro-preview").
        api_key: API key for the provider. Defaults to OPENROUTER_API_KEY env var.
        base_url: API base URL. Defaults to OpenRouter.
        mcp: McpToolCaller instance. Defaults to the shared singleton.
        extra_tools: Additional OpenAI-format tool definitions to include.
        max_turns: Maximum number of tool-calling rounds.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        max_retries: Retries per API call on transient errors.
        on_tool_call: Optional callback(tool_name, tool_args) for logging/UI.

    Returns:
        The model's final text response.

    Raises:
        ValueError: If all retries are exhausted without a response.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and OPENROUTER_API_KEY not set")

    if mcp is None:
        from brrragent.mcp_tools import get_default_caller
        mcp = get_default_caller()

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

    for turn in range(max_turns):
        logger.info("[agent] Turn %d/%d — calling %s", turn + 1, max_turns, model)

        # Call the model with retry
        response = _call_with_retry(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

        choice = response.choices[0]
        message = choice.message

        # Add assistant message to history
        messages.append(message.model_dump())

        # Check if the model wants to call tools
        if message.tool_calls:
            logger.info(
                "[agent] Turn %d: %d tool call(s)",
                turn + 1,
                len(message.tool_calls),
            )

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("[agent] Tool: %s(%s)", fn_name, list(fn_args.keys()))
                if on_tool_call:
                    on_tool_call(fn_name, fn_args)

                # Execute via MCP
                result_text = mcp.call_tool(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
        else:
            # Model returned final content — we're done
            final_text = message.content or ""
            logger.info(
                "[agent] Final response after %d turn(s) (%d chars)",
                turn + 1,
                len(final_text),
            )
            return final_text

        # Check finish reason
        if choice.finish_reason == "stop":
            return message.content or ""

    logger.warning("[agent] Reached max_turns=%d without final response", max_turns)
    # Return whatever we have from the last message
    last = messages[-1]
    if isinstance(last, dict):
        return last.get("content", "[No final response after max tool turns]")
    return getattr(last, "content", "[No final response after max tool turns]") or ""


def _call_with_retry(
    *,
    client: OpenAI,
    model: str,
    messages: list,
    tools: list,
    temperature: float,
    max_tokens: int,
    max_retries: int,
):
    """Call the OpenAI-compatible API with exponential backoff on transient errors."""
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={"X-Title": "brrragent"},
            )
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_transient = any(
                p in err_str
                for p in ("500", "503", "429", "unavailable", "overloaded", "rate")
            )

            logger.warning(
                "[agent] API call attempt %d/%d failed: %s",
                attempt,
                max_retries,
                str(e)[:200],
            )

            if is_transient and attempt < max_retries:
                wait = min(2 ** attempt, 30)
                logger.info("[agent] Transient error, retrying in %ds...", wait)
                time.sleep(wait)
            elif not is_transient:
                raise

    raise ValueError(f"API call failed after {max_retries} retries: {last_err}")
