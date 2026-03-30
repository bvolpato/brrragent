"""
MCP Tool Bridge.

Converts any MCP JSON-RPC endpoint into tool definitions for LLM function
calling.  Supports both OpenAI-compatible (OpenRouter, OpenAI) and
Google genai (Gemini) formats.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Default set of tools to expose (can be overridden)
DEFAULT_RESEARCH_TOOLS = {"web_search", "news_search", "read_url", "youtube_search"}


class McpToolCaller:
    """
    HTTP client for an MCP JSON-RPC endpoint.

    Discovers tool schemas, converts them to function-call definitions
    (OpenAI or Google genai format), and executes tool calls via JSON-RPC.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        allowed_tools: set[str] | None = None,
        max_response_len: int = 15_000,
    ):
        self.endpoint = endpoint or os.getenv(
            "MCP_CRAWLERPY_URL",
            "https://crawlerpy.brunovolpato.com/mcp",
        )
        self.allowed_tools = allowed_tools or DEFAULT_RESEARCH_TOOLS
        self.max_response_len = max_response_len
        self._tool_schemas: list[dict] | None = None

    # ── Schema discovery ──────────────────────────────────────────────────

    _HEADERS: ClassVar[dict[str, str]] = {
        "Content-Type": "application/json",
        "User-Agent": "brrragent/1.0",
    }

    def _post(self, payload: dict, timeout: int = 30) -> dict:
        """POST JSON-RPC payload to the MCP endpoint and return parsed response."""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=self._HEADERS,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _fetch_tool_schemas(self) -> list[dict]:
        """Fetch tool list from MCP endpoint (cached per instance)."""
        if self._tool_schemas is not None:
            return self._tool_schemas

        data = self._post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        self._tool_schemas = data.get("result", {}).get("tools", [])
        logger.info("MCP: discovered %d tools from %s", len(self._tool_schemas), self.endpoint)
        return self._tool_schemas

    def get_filtered_schemas(self) -> list[dict]:
        """Return only the schemas for allowed tools."""
        return [t for t in self._fetch_tool_schemas() if t["name"] in self.allowed_tools]

    # ── OpenAI-style function definitions ────────────────────────────────

    def get_openai_tools(self) -> list[dict]:
        """
        Build OpenAI-compatible tool definitions for function calling.

        Returns a list of dicts in the format:
        [{"type": "function", "function": {"name", "description", "parameters"}}]
        """
        tools = []
        for schema in self.get_filtered_schemas():
            input_schema = schema.get("inputSchema", {})
            params = _clean_schema(input_schema)
            tools.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": (schema.get("description", "")[:500]),
                    "parameters": params,
                },
            })
        return tools

    # ── Google genai FunctionDeclarations ─────────────────────────────────

    def get_genai_tools(self):
        """
        Build google.genai Tool objects from MCP schemas.

        Returns a list containing a single Tool with all FunctionDeclarations.
        Requires google-genai to be installed (lazy import).
        """
        try:
            from google.genai import types as gt
        except ImportError as e:
            raise ImportError("google-genai required: pip install google-genai") from e

        declarations = []
        for schema in self.get_filtered_schemas():
            input_schema = schema.get("inputSchema", {})
            params = _clean_schema(input_schema)
            decl = gt.FunctionDeclaration(
                name=schema["name"],
                description=schema.get("description", "")[:500],
                parameters=params,
            )
            declarations.append(decl)

        return [gt.Tool(function_declarations=declarations)]

    # ── Tool execution ────────────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Execute a tool via the MCP JSON-RPC endpoint.

        Returns the tool result as a string.
        """
        logger.info("MCP call: %s(%s)", name, list(arguments.keys()))

        try:
            data = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
                timeout=90,
            )
        except urllib.error.URLError as e:
            logger.error("MCP tool %s failed: %s", name, e)
            return f"[Tool error: {e}]"

        if "error" in data:
            err = data["error"]
            logger.warning("MCP tool %s error: %s", name, err)
            return f"[Tool error: {err.get('message', str(err))}]"

        # Extract text content from MCP result
        result = data.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            text = "\n".join(parts)
        elif isinstance(content, str):
            text = content
        else:
            text = json.dumps(result)

        # Truncate very large responses to avoid context overflow
        if len(text) > self.max_response_len:
            text = text[: self.max_response_len] + "\n... [truncated]"

        return text


def _clean_schema(schema: Any) -> Any:
    """
    Clean a JSON Schema dict for OpenAI function calling.

    OpenAI rejects keys like 'format', 'example', 'minimum', 'maximum' etc.
    """
    SUPPORTED_KEYS = {"type", "description", "properties", "required", "items", "enum"}

    if not isinstance(schema, dict):
        return schema
    cleaned = {k: v for k, v in schema.items() if k in SUPPORTED_KEYS}
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _clean_schema(v) for k, v in cleaned["properties"].items()
        }
    if "items" in cleaned:
        cleaned["items"] = _clean_schema(cleaned["items"])
    return cleaned


# ── Singleton ──────────────────────────────────────────────────────────────

_default_caller: McpToolCaller | None = None


def get_default_caller() -> McpToolCaller:
    """Return the singleton MCP caller (lazy-initialized)."""
    global _default_caller
    if _default_caller is None:
        _default_caller = McpToolCaller()
    return _default_caller
