"""
Direct OpenAI backend for the agent loop.

Handles OpenAI reasoning-effort suffixes such as ``openai/gpt-5.5:xhigh``.
"""

import json
import logging
import time
from typing import Callable, Optional

from brrragent.keys import KeyPool
from brrragent.prompt_cache import AgentUsage, PromptCacheConfig, openai_usage

logger = logging.getLogger(__name__)

_REASONING_SUFFIXES = (":none", ":low", ":medium", ":high", ":xhigh")


def parse_openai_model(model: str) -> tuple[str, str | None]:
    """Return bare OpenAI model plus optional reasoning effort."""
    bare = model[len("openai/") :] if model.startswith("openai/") else model
    for suffix in _REASONING_SUFFIXES:
        if bare.endswith(suffix):
            return bare[: -len(suffix)], suffix[1:]
    return bare, None


def _is_reasoning_model(model: str, reasoning_effort: str | None) -> bool:
    if reasoning_effort is not None:
        return True
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _build_chat_kwargs(
    *,
    model: str,
    messages: list,
    tools: list,
    temperature: float,
    max_tokens: int,
    response_format: dict | None,
    prompt_cache: PromptCacheConfig | None = None,
) -> dict:
    bare_model, reasoning_effort = parse_openai_model(model)
    kwargs = {
        "model": bare_model,
        "messages": messages,
        "extra_headers": {"X-Title": "brrragent"},
    }
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format
    if prompt_cache:
        kwargs["prompt_cache_key"] = prompt_cache.key

    if _is_reasoning_model(bare_model, reasoning_effort):
        kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    return kwargs


def _to_responses_tool(tool: dict) -> dict:
    if tool.get("type") != "function":
        return tool
    fn = tool.get("function", {})
    return {
        "type": "function",
        "name": fn.get("name"),
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        "strict": False,
    }


def _build_responses_kwargs(
    *,
    model: str,
    input_items,
    instructions: str | None,
    tools: list,
    temperature: float,
    max_tokens: int,
    response_schema: dict | None,
    previous_response_id: str | None = None,
    prompt_cache: PromptCacheConfig | None = None,
) -> dict:
    bare_model, reasoning_effort = parse_openai_model(model)
    explicit_cache = (
        prompt_cache is not None
        and prompt_cache.mode == "explicit"
        and bare_model.startswith("gpt-5.6")
        and instructions
        and previous_response_id is None
    )
    if explicit_cache:
        input_items = [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": instructions,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            *input_items,
        ]
        instructions = None
    kwargs = {
        "model": bare_model,
        "input": input_items,
        "max_output_tokens": max_tokens,
        "store": False,
        "extra_headers": {"X-Title": "brrragent"},
    }
    if instructions and previous_response_id is None:
        kwargs["instructions"] = instructions
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    if tools:
        kwargs["tools"] = [_to_responses_tool(tool) for tool in tools]
    if response_schema:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "agent_response",
                "strict": True,
                "schema": response_schema,
            }
        }
    if prompt_cache:
        kwargs["prompt_cache_key"] = prompt_cache.key
        if explicit_cache:
            kwargs["prompt_cache_options"] = {"mode": "explicit"}

    if _is_reasoning_model(bare_model, reasoning_effort):
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        kwargs["temperature"] = temperature
    return kwargs


def _should_use_responses(
    model: str, tools: list, response_schema: dict | None
) -> bool:
    bare_model, reasoning_effort = parse_openai_model(model)
    return _is_reasoning_model(bare_model, reasoning_effort) and (
        bool(tools) or bool(response_schema)
    )


def _response_function_calls(response) -> list:
    return [
        item
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "function_call"
    ]


def _response_final_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    parts: list[str] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", None)
            if content_type in {"output_text", "text"}:
                value = getattr(content, "text", "")
                if value:
                    parts.append(value)
    return "\n".join(parts)


def _is_rate_limit_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return any(p in err_str for p in ("429", "rate", "quota"))


def _is_transient_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return any(
        p in err_str
        for p in (
            "500",
            "502",
            "503",
            "504",
            "429",
            "unavailable",
            "overloaded",
            "rate",
        )
    )


def run_openai_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None = None,
    key_pool: KeyPool | None = None,
    mcp,
    extra_tools: list[dict] | None,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
    prompt_cache: PromptCacheConfig | None = None,
    on_usage: Callable[[AgentUsage], None] | None = None,
) -> str:
    """Run the agent using OpenAI's native API."""
    from openai import OpenAI

    if key_pool is None and api_key is None:
        raise ValueError("Either api_key or key_pool must be provided")

    selected_key = api_key or key_pool.acquire()
    client = OpenAI(
        api_key=selected_key,
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

    if _should_use_responses(model, tools, response_schema):
        return _run_openai_responses_agent(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            tools=tools,
            mcp=mcp,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            max_turns=max_turns,
            on_tool_call=on_tool_call,
            response_schema=response_schema,
            key_pool=key_pool,
            selected_key=selected_key,
            prompt_cache=prompt_cache,
            on_usage=on_usage,
        )

    for turn in range(max_turns):
        bare_model, _ = parse_openai_model(model)
        logger.info(
            "[agent] Turn %d/%d - calling OpenAI %s", turn + 1, max_turns, bare_model
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
            prompt_cache=prompt_cache,
        )

        if isinstance(result, tuple):
            response, new_key = result
            if new_key != selected_key:
                selected_key = new_key
                client = OpenAI(
                    api_key=selected_key,
                    timeout=120.0,
                    default_headers={"X-Title": "brrragent"},
                )
        else:
            response = result

        if on_usage and getattr(response, "usage", None):
            on_usage(openai_usage(response.usage))

        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_none=True))

        if message.tool_calls:
            logger.info(
                "[agent] Turn %d: %d tool call(s)", turn + 1, len(message.tool_calls)
            )

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments or "{}")
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


def _run_openai_responses_agent(
    *,
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    tools: list,
    mcp,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    max_turns: int,
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
    key_pool: KeyPool | None,
    selected_key: str,
    prompt_cache: PromptCacheConfig | None,
    on_usage: Callable[[AgentUsage], None] | None,
) -> str:
    from openai import OpenAI

    previous_response_id = None
    input_items = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        bare_model, _ = parse_openai_model(model)
        logger.info(
            "[agent] Turn %d/%d - calling OpenAI Responses %s",
            turn + 1,
            max_turns,
            bare_model,
        )

        result = _call_responses_with_retry(
            client=client,
            model=model,
            input_items=input_items,
            instructions=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            response_schema=response_schema,
            previous_response_id=previous_response_id,
            key_pool=key_pool,
            current_key=selected_key,
            prompt_cache=prompt_cache,
        )

        if isinstance(result, tuple):
            response, new_key = result
            if new_key != selected_key:
                selected_key = new_key
                client = OpenAI(
                    api_key=selected_key,
                    timeout=120.0,
                    default_headers={"X-Title": "brrragent"},
                )
        else:
            response = result

        if on_usage and getattr(response, "usage", None):
            on_usage(openai_usage(response.usage))

        previous_response_id = response.id
        function_calls = _response_function_calls(response)
        if not function_calls:
            final_text = _response_final_text(response)
            logger.info(
                "[agent] Final response after %d turn(s) (%d chars)",
                turn + 1,
                len(final_text),
            )
            return final_text

        logger.info("[agent] Turn %d: %d tool call(s)", turn + 1, len(function_calls))
        input_items = []
        for tool_call in function_calls:
            fn_name = tool_call.name
            try:
                fn_args = json.loads(tool_call.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[agent] Tool: %s(%s)", fn_name, list(fn_args.keys()))
            if on_tool_call:
                on_tool_call(fn_name, fn_args)

            result_text = mcp.call_tool(fn_name, fn_args)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result_text,
                }
            )

    logger.warning(
        "[agent] Reached max_turns=%d without final response; requesting final answer without tools",
        max_turns,
    )
    result = _call_responses_with_retry(
        client=client,
        model=model,
        input_items=[
            *input_items,
            {
                "role": "user",
                "content": "Stop calling tools. Provide the final answer using the evidence already gathered.",
            },
        ],
        instructions=system_prompt,
        tools=[],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        response_schema=response_schema,
        previous_response_id=previous_response_id,
        key_pool=key_pool,
        current_key=selected_key,
        prompt_cache=prompt_cache,
    )
    response = result[0] if isinstance(result, tuple) else result
    if on_usage and getattr(response, "usage", None):
        on_usage(openai_usage(response.usage))
    final_text = _response_final_text(response)
    logger.info(
        "[agent] Final no-tool response after max_turns (%d chars)", len(final_text)
    )
    return final_text or "[No final response after max tool turns]"


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
    prompt_cache: PromptCacheConfig | None = None,
):
    from openai import OpenAI

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = _build_chat_kwargs(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                prompt_cache=prompt_cache,
            )
            resp = client.chat.completions.create(**kwargs)
            if key_pool:
                return (resp, current_key)
            return resp
        except Exception as e:
            last_err = e

            if _is_rate_limit_error(e) and key_pool:
                logger.warning(
                    "[agent] OpenAI 429 on key ...%s (attempt %d/%d): %s",
                    current_key[-6:],
                    attempt,
                    max_retries,
                    str(e)[:200],
                )
                key_pool.report_rate_limit(current_key)
                current_key = key_pool.acquire()
                client = OpenAI(
                    api_key=current_key,
                    timeout=120.0,
                    default_headers={"X-Title": "brrragent"},
                )
                logger.info("[agent] Rotated to OpenAI key ...%s", current_key[-6:])
                continue

            logger.warning(
                "[agent] OpenAI call attempt %d/%d failed: %s",
                attempt,
                max_retries,
                str(e)[:200],
            )

            if _is_transient_error(e) and attempt < max_retries:
                wait = min(2**attempt, 30)
                logger.info("[agent] Transient OpenAI error, retrying in %ds", wait)
                time.sleep(wait)
            elif not _is_transient_error(e):
                raise

    raise ValueError(f"OpenAI call failed after {max_retries} retries: {last_err}")


def _call_responses_with_retry(
    *,
    client,
    model: str,
    input_items,
    instructions: str | None,
    tools: list,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    response_schema: dict | None = None,
    previous_response_id: str | None = None,
    key_pool: KeyPool | None = None,
    current_key: str = "",
    prompt_cache: PromptCacheConfig | None = None,
):
    from openai import OpenAI

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = _build_responses_kwargs(
                model=model,
                input_items=input_items,
                instructions=instructions,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                response_schema=response_schema,
                previous_response_id=previous_response_id,
                prompt_cache=prompt_cache,
            )
            resp = client.responses.create(**kwargs)
            if key_pool:
                return (resp, current_key)
            return resp
        except Exception as e:
            last_err = e

            if _is_rate_limit_error(e) and key_pool:
                logger.warning(
                    "[agent] OpenAI 429 on key ...%s (attempt %d/%d): %s",
                    current_key[-6:],
                    attempt,
                    max_retries,
                    str(e)[:200],
                )
                key_pool.report_rate_limit(current_key)
                current_key = key_pool.acquire()
                client = OpenAI(
                    api_key=current_key,
                    timeout=120.0,
                    default_headers={"X-Title": "brrragent"},
                )
                logger.info("[agent] Rotated to OpenAI key ...%s", current_key[-6:])
                continue

            logger.warning(
                "[agent] OpenAI Responses attempt %d/%d failed: %s",
                attempt,
                max_retries,
                str(e)[:200],
            )

            if _is_transient_error(e) and attempt < max_retries:
                wait = min(2**attempt, 30)
                logger.info("[agent] Transient OpenAI error, retrying in %ds", wait)
                time.sleep(wait)
            elif not _is_transient_error(e):
                raise

    raise ValueError(
        f"OpenAI Responses call failed after {max_retries} retries: {last_err}"
    )
