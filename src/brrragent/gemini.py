"""
Google Gemini direct backend for the agent loop.

Uses the google-genai SDK to call Gemini models directly without going
through OpenRouter — no premium, same functionality.

Supports KeyPool-based 429 eviction: when a key hits rate limits, the
pool evicts it and the agent retries with a fresh key.
"""

import logging
import time
from typing import Callable, Optional

from brrragent.keys import KeyPool

logger = logging.getLogger(__name__)


def _is_rate_limit_error(err: Exception) -> bool:
    """Check if an error is a 429 / quota / rate-limit error."""
    err_str = str(err)
    return any(p in err_str for p in ("429", "RESOURCE_EXHAUSTED", "quota", "rate"))


def run_gemini_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str | None = None,
    key_pool: KeyPool | None = None,
    mcp,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    on_tool_call: Optional[Callable],
    response_schema: dict | None,
) -> str:
    """Run the agent using the Google genai SDK directly (no OpenRouter).

    Key selection priority:
      1. key_pool — acquires from pool, evicts on 429, retries with fresh key
      2. api_key — single explicit key (legacy)
    """
    try:
        from google import genai
        from google.genai import types as gt
    except ImportError as e:
        raise ImportError("google-genai required for Gemini direct: pip install google-genai") from e

    if key_pool is None and api_key is None:
        raise ValueError("Either api_key or key_pool must be provided")

    selected_key = api_key or key_pool.acquire()
    client = genai.Client(api_key=selected_key)
    tools = mcp.get_genai_tools()

    contents: list[gt.Content] = [
        gt.Content(role="user", parts=[gt.Part(text=user_prompt)])
    ]

    config_kwargs = dict(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
        tools=tools,
    )

    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = gt.GenerateContentConfig(**config_kwargs)

    for turn in range(max_turns):
        logger.info("[agent] Turn %d/%d — calling %s (gemini direct)", turn + 1, max_turns, model)

        response = _call_with_retry(
            client=client,
            model=model,
            contents=contents,
            config=config,
            max_retries=max_retries,
            key_pool=key_pool,
            current_key=selected_key,
        )

        # If we got back a new key (pool rotation happened), rebuild the client
        if isinstance(response, tuple):
            response, new_key = response
            if new_key != selected_key:
                selected_key = new_key
                client = genai.Client(api_key=selected_key)

        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            break

        model_parts = list(candidate.content.parts or [])

        fn_calls = [p for p in model_parts if p.function_call is not None]

        if not fn_calls:
            text_parts = [p.text for p in model_parts if p.text]
            final_text = "\n".join(text_parts)
            logger.info("[agent] Final response after %d turn(s) (%d chars)", turn + 1, len(final_text))
            return final_text

        contents.append(gt.Content(role="model", parts=model_parts))

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


def _call_with_retry(
    *,
    client,
    model: str,
    contents,
    config,
    max_retries: int,
    key_pool: KeyPool | None = None,
    current_key: str = "",
):
    """Call the Gemini API with exponential backoff on transient errors.

    On 429 errors with a key_pool:
      - Evicts the current key from the pool
      - Acquires a fresh key and rebuilds the client
      - Returns (response, new_key) tuple so caller can track the key change

    Without a key_pool, 429s are retried with backoff like any transient error.
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if key_pool:
                return (resp, current_key)
            return resp
        except Exception as e:
            last_err = e
            err_str = str(e)

            if _is_rate_limit_error(e) and key_pool:
                logger.warning(
                    "[agent] Gemini 429 on key ...%s (attempt %d/%d): %s",
                    current_key[-6:], attempt, max_retries, err_str[:200],
                )
                # Evict and get a new key — may raise RateLimitExhausted
                key_pool.report_rate_limit(current_key)
                current_key = key_pool.acquire()

                from google import genai
                client = genai.Client(api_key=current_key)
                logger.info("[agent] Rotated to key ...%s, retrying immediately", current_key[-6:])
                continue

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
