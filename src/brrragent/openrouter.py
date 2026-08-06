"""
OpenRouter / OpenAI-compatible backend for the agent loop.

Supports KeyPool-based 429 eviction: when a key hits rate limits, the
pool evicts it and the agent retries with a fresh key.
"""

import json
import logging
import time
from collections.abc import Callable

from brrragent.keys import KeyPool
from brrragent.prompt_cache import AgentUsage, PromptCacheConfig, openai_usage

logger = logging.getLogger(__name__)


def _system_content(
    system_prompt: str,
    model: str,
    prompt_cache: PromptCacheConfig | None,
):
    if not prompt_cache or not model.startswith("anthropic/"):
        return system_prompt
    cache_control = {"type": "ephemeral"}
    if prompt_cache.ttl == "1h":
        cache_control["ttl"] = "1h"
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": cache_control,
        }
    ]


def _is_rate_limit_error(err: Exception) -> bool:
    """Check if an error is a 429 / rate-limit error."""
    err_str = str(err).lower()
    return any(p in err_str for p in ("429", "rate", "quota"))


def run_openrouter_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None = None,
    key_pool: KeyPool | None = None,
    base_url: str,
    mcp,
    extra_tools: list[dict] | None,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Callable | None,
    response_schema: dict | None,
    reasoning_effort: str | None = None,
    prompt_cache: PromptCacheConfig | None = None,
    on_usage: Callable[[AgentUsage], None] | None = None,
) -> str:
    """Run the agent using an OpenAI-compatible API (OpenRouter, OpenAI, etc.).

    Key selection priority:
      1. key_pool — acquires from pool, evicts on 429, retries with fresh key
      2. api_key — single explicit key (legacy)
    """
    from openai import OpenAI

    if key_pool is None and api_key is None:
        raise ValueError("Either api_key or key_pool must be provided")

    selected_key = api_key or key_pool.acquire()

    client = OpenAI(
        api_key=selected_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
        default_headers={"X-Title": "brrragent"},
    )

    tools = mcp.get_openai_tools()
    if extra_tools:
        tools.extend(extra_tools)

    messages = [
        {
            "role": "system",
            "content": _system_content(system_prompt, model, prompt_cache),
        },
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
        logger.info(
            "[agent] Turn %d/%d — calling %s (%s)", turn + 1, max_turns, model, base_url
        )

        result = _call_with_retry(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            response_format=response_format,
            key_pool=key_pool,
            current_key=selected_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            prompt_cache=prompt_cache,
        )

        # Unpack pool rotation if applicable
        if isinstance(result, tuple):
            response, new_key = result
            if new_key != selected_key:
                selected_key = new_key
                client = OpenAI(
                    api_key=selected_key,
                    base_url=base_url,
                    timeout=120.0,
                    max_retries=0,
                    default_headers={"X-Title": "brrragent"},
                )
        else:
            response = result

        if on_usage and getattr(response, "usage", None):
            on_usage(openai_usage(response.usage))

        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump())

        if message.tool_calls:
            logger.info(
                "[agent] Turn %d: %d tool call(s)", turn + 1, len(message.tool_calls)
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

                result_text = mcp.call_tool(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
        else:
            final_text = message.content or ""
            logger.info(
                "[agent] Final response after %d turn(s) (%d chars)",
                turn + 1,
                len(final_text),
            )
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
    key_pool: KeyPool | None = None,
    current_key: str = "",
    base_url: str = "",
    reasoning_effort: str | None = None,
    prompt_cache: PromptCacheConfig | None = None,
):
    """Call the OpenAI-compatible API with exponential backoff on transient errors.

    On 429 errors with a key_pool:
      - Evicts the current key from the pool
      - Acquires a fresh key and rebuilds the client
      - Returns (response, new_key) tuple so caller can track the key change
    """
    from openai import OpenAI

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                extra_headers={"X-Title": "brrragent"},
            )
            if reasoning_effort is not None:
                kwargs["max_completion_tokens"] = max_tokens
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
            if prompt_cache and base_url.rstrip("/") == "https://openrouter.ai/api/v1":
                kwargs["extra_body"] = {
                    "prompt_cache_key": prompt_cache.key,
                    "session_id": prompt_cache.key,
                }
            resp = client.chat.completions.create(**kwargs)
            if key_pool:
                return (resp, current_key)
            return resp
        except Exception as e:
            last_err = e
            err_str = str(e).lower()

            if _is_rate_limit_error(e) and key_pool:
                logger.warning(
                    "[agent] OpenRouter 429 on key ...%s (attempt %d/%d): %s",
                    current_key[-6:],
                    attempt,
                    max_retries,
                    str(e)[:200],
                )
                key_pool.report_rate_limit(current_key)
                current_key = key_pool.acquire()
                client = OpenAI(
                    api_key=current_key,
                    base_url=base_url,
                    timeout=120.0,
                    max_retries=0,
                    default_headers={"X-Title": "brrragent"},
                )
                logger.info(
                    "[agent] Rotated to key ...%s, retrying immediately",
                    current_key[-6:],
                )
                continue

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
                wait = min(2**attempt, 30)
                logger.info("[agent] Transient error, retrying in %ds...", wait)
                time.sleep(wait)
            elif not is_transient:
                raise

    raise ValueError(f"API call failed after {max_retries} retries: {last_err}")
