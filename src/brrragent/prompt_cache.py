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


def _first_present(*values: Any, default: Any = 0) -> Any:
    for item in values:
        if item is not None:
            return item
    return default


def openai_usage(usage: Any) -> AgentUsage:
    input_tokens = int(
        _first_present(
            value(usage, "input_tokens"),
            value(usage, "prompt_tokens"),
            default=0,
        )
    )
    output_tokens = int(
        _first_present(
            value(usage, "output_tokens"),
            value(usage, "completion_tokens"),
            default=0,
        )
    )
    total_tokens = int(
        _first_present(
            value(usage, "total_tokens"),
            default=input_tokens + output_tokens,
        )
    )
    details = value(usage, "input_tokens_details") or value(
        usage, "prompt_tokens_details"
    )
    cached_tokens = int(_first_present(value(details, "cached_tokens"), default=0))
    cache_write_tokens = int(
        _first_present(
            value(usage, "cache_write_tokens"),
            value(usage, "cache_write_input_tokens"),
            value(details, "cache_write_tokens"),
            value(usage, "cache_creation_input_tokens"),
            default=0,
        )
    )
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_input_tokens=cached_tokens,
        cache_write_input_tokens=cache_write_tokens,
    )


def gemini_usage(usage: Any) -> AgentUsage:
    input_tokens = int(_first_present(value(usage, "prompt_token_count"), default=0))
    output_tokens = int(
        _first_present(value(usage, "candidates_token_count"), default=0)
    )
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            _first_present(
                value(usage, "total_token_count"),
                default=input_tokens + output_tokens,
            )
        ),
        cache_read_input_tokens=int(
            _first_present(value(usage, "cached_content_token_count"), default=0)
        ),
    )
