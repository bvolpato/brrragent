"""
OpenRouter / OpenAI-compatible backend for the agent loop.
"""

import json
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def run_openrouter_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    mcp,
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

    tools = mcp.get_openai_tools()
    if extra_tools:
        tools.extend(extra_tools)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

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

        response = _call_with_retry(
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
        messages.append(message.model_dump())

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


def _call_with_retry(
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
