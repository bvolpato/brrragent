"""Shared validation and execution for model-requested tool calls."""

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

INVALID_TOOL_ARGUMENTS = "[Tool error: invalid tool arguments]"


def execute_tool_call(
    mcp: Any,
    name: str,
    arguments: str | None,
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
    """Validate and execute a tool call, returning invalid arguments to the model."""
    try:
        parsed_arguments = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[agent] Invalid tool arguments for %s", name)
        return INVALID_TOOL_ARGUMENTS

    if not isinstance(parsed_arguments, dict):
        logger.warning("[agent] Invalid tool arguments for %s", name)
        return INVALID_TOOL_ARGUMENTS

    logger.info("[agent] Tool: %s(%s)", name, list(parsed_arguments.keys()))
    if on_tool_call:
        on_tool_call(name, parsed_arguments)

    return mcp.call_tool(name, parsed_arguments)
