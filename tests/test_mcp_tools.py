import urllib.error

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
