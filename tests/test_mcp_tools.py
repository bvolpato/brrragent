import urllib.error

import pytest

from brrragent.mcp_tools import McpToolCaller, _clean_schema


def test_clean_schema_removes_unsupported_keys_recursively():
    schema = {
        "type": "object",
        "format": "json",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "minimum": 3,
            },
            "limit": {
                "type": "integer",
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    assert _clean_schema(schema) == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "limit": {
                "type": "integer",
            },
        },
        "required": ["query"],
    }


class FakeMcpToolCaller(McpToolCaller):
    def __init__(self, responses, **kwargs):
        super().__init__(endpoint="https://mcp.example.test", **kwargs)
        self.responses = list(responses)

    def _post(self, payload, timeout=30):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_endpoint_is_required_without_environment(monkeypatch):
    monkeypatch.delenv("BRRRAGENT_MCP_URL", raising=False)

    with pytest.raises(ValueError, match="MCP endpoint required"):
        McpToolCaller()


def test_endpoint_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("BRRRAGENT_MCP_URL", "https://mcp.example.test")

    assert McpToolCaller().endpoint == "https://mcp.example.test"


def test_default_discovers_all_tools():
    caller = FakeMcpToolCaller(
        [
            {
                "result": {
                    "tools": [
                        {"name": "first_tool"},
                        {"name": "another_tool"},
                    ]
                }
            }
        ]
    )

    assert [tool["name"] for tool in caller.get_filtered_schemas()] == [
        "first_tool",
        "another_tool",
    ]


def test_tool_patterns_support_wildcards_and_exclusions():
    caller = FakeMcpToolCaller(
        [
            {
                "result": {
                    "tools": [
                        {"name": "web_search"},
                        {"name": "news_search"},
                        {"name": "private_search"},
                        {"name": "read_url"},
                    ]
                }
            }
        ],
        allowed_tools="*_search",
        excluded_tools={"private_*", "news_search"},
    )

    assert [tool["name"] for tool in caller.get_filtered_schemas()] == ["web_search"]


def test_empty_allowed_tools_skips_discovery():
    caller = FakeMcpToolCaller([], allowed_tools=set())

    assert caller.get_filtered_schemas() == []


def test_get_openai_tools_filters_allowed_schemas():
    caller = FakeMcpToolCaller(
        [
            {
                "result": {
                    "tools": [
                        {
                            "name": "web_search",
                            "description": "Search",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"q": {"type": "string", "format": "text"}},
                            },
                        },
                        {
                            "name": "unused_tool",
                            "description": "Ignore",
                            "inputSchema": {"type": "object"},
                        },
                    ]
                }
            }
        ],
        allowed_tools={"web_search"},
    )

    assert caller.get_openai_tools() == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]


def test_call_tool_returns_text_and_truncates_large_responses():
    caller = FakeMcpToolCaller(
        [
            {
                "result": {
                    "content": [
                        {"type": "text", "text": "abc"},
                        {"type": "text", "text": "def"},
                    ]
                }
            }
        ],
        max_response_len=5,
    )

    assert caller.call_tool("web_search", {"q": "test"}) == "abc\nd\n... [truncated]"


def test_call_tool_returns_error_string_for_url_errors():
    caller = FakeMcpToolCaller(
        [urllib.error.URLError("offline")],
    )

    assert caller.call_tool("web_search", {"q": "test"}) == "[Tool error: <urlopen error offline>]"
