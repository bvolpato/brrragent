from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class PromptCacheConfig:
    """Stable cache identity shared across requests for one prompt version."""

    key: str
    ttl: Literal["5m", "1h"] = "1h"
    mode: Literal["auto", "explicit"] = "explicit"


@dataclass(frozen=True)
class AgentUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0


def value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def openai_usage(usage: Any) -> AgentUsage:
    input_tokens = int(
        value(usage, "input_tokens", value(usage, "prompt_tokens", 0)) or 0
    )
    output_tokens = int(
        value(usage, "output_tokens", value(usage, "completion_tokens", 0)) or 0
    )
    total_tokens = int(value(usage, "total_tokens", input_tokens + output_tokens) or 0)
    details = value(usage, "input_tokens_details") or value(
        usage, "prompt_tokens_details"
    )
    cached_tokens = int(value(details, "cached_tokens", 0) or 0)
    cache_write_tokens = int(
        value(
            usage,
            "cache_write_tokens",
            value(
                usage,
                "cache_write_input_tokens",
                value(
                    details,
                    "cache_write_tokens",
                    value(usage, "cache_creation_input_tokens", 0),
                ),
            ),
        )
        or 0
    )
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_input_tokens=cached_tokens,
        cache_write_input_tokens=cache_write_tokens,
    )


def gemini_usage(usage: Any) -> AgentUsage:
    input_tokens = int(value(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(value(usage, "candidates_token_count", 0) or 0)
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            value(usage, "total_token_count", input_tokens + output_tokens) or 0
        ),
        cache_read_input_tokens=int(value(usage, "cached_content_token_count", 0) or 0),
    )
