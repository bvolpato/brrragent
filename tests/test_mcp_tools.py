import asyncio
import socket
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import Future
from contextlib import asynccontextmanager
from threading import Event, Thread
from types import SimpleNamespace

import httpx2
import pytest

from brrragent import mcp_tools
from brrragent.mcp_tools import (
    McpServerConfig,
    McpToolCaller,
    _clean_schema,
    _WorkerCommand,
)


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
        super().__init__(
            endpoint="https://mcp.example.test",
            transport="legacy_json",
            **kwargs,
        )
        self.responses = list(responses)

    def _legacy_request(self, server, payload, timeout):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_endpoint_is_required_without_environment(monkeypatch):
    monkeypatch.delenv("BRRRAGENT_MCP_URL", raising=False)
    monkeypatch.delenv("BRRRAGENT_MCP_URLS", raising=False)

    with pytest.raises(ValueError, match="MCP server required"):
        McpToolCaller()


def test_endpoint_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("BRRRAGENT_MCP_URL", "https://mcp.example.test")

    assert McpToolCaller().endpoint == "https://mcp.example.test"


def test_multiple_endpoints_can_come_from_environment(monkeypatch):
    monkeypatch.setenv(
        "BRRRAGENT_MCP_URLS",
        "https://mcp.example.test/one, https://mcp.example.test/two",
    )

    caller = McpToolCaller()

    assert caller.endpoints == (
        "https://mcp.example.test/one",
        "https://mcp.example.test/two",
    )


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
                                "properties": {
                                    "q": {"type": "string", "format": "text"}
                                },
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
                    "tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]
                }
            },
            {
                "result": {
                    "content": [
                        {"type": "text", "text": "abc"},
                        {"type": "text", "text": "def"},
                    ]
                }
            },
        ],
        max_response_len=5,
    )

    assert caller.call_tool("web_search", {"q": "test"}) == "abc\nd\n... [truncated]"


def test_call_tool_returns_error_string_for_url_errors():
    caller = FakeMcpToolCaller(
        [urllib.error.URLError("offline")],
    )

    assert (
        caller.call_tool("web_search", {"q": "test"})
        == "[Tool error: <urlopen error offline>]"
    )


def test_excluded_tool_cannot_be_called_directly():
    caller = FakeMcpToolCaller([], excluded_tools={"private_*"})

    assert caller.call_tool("private_search", {}) == (
        "[Tool error: tool 'private_search' is not allowed]"
    )
    assert caller.responses == []


class MultiServerFakeCaller(McpToolCaller):
    def __init__(self):
        super().__init__(
            servers=[
                McpServerConfig(
                    name="search",
                    transport="legacy_json",
                    url="https://mcp.example.test/search",
                    tool_prefix="search_",
                ),
                McpServerConfig(
                    name="data",
                    transport="legacy_json",
                    url="https://mcp.example.test/data",
                    tool_prefix="data_",
                ),
            ]
        )
        self.calls = []

    def _legacy_request(self, server, payload, timeout):
        if payload["method"] == "tools/list":
            return {
                "result": {
                    "tools": [
                        {
                            "name": "lookup",
                            "description": server.name,
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            }
        self.calls.append((server.name, payload["params"]))
        return {
            "result": {"content": [{"type": "text", "text": f"from {server.name}"}]}
        }


def test_multiple_servers_namespace_and_route_tools():
    with MultiServerFakeCaller() as caller:
        assert [tool["name"] for tool in caller.get_filtered_schemas()] == [
            "search_lookup",
            "data_lookup",
        ]
        assert caller.call_tool("data_lookup", {"id": 7}) == "from data"

    assert caller.calls == [("data", {"name": "lookup", "arguments": {"id": 7}})]


def test_duplicate_tool_names_require_prefixes():
    class DuplicateFakeCaller(MultiServerFakeCaller):
        def __init__(self):
            McpToolCaller.__init__(
                self,
                servers=[
                    McpServerConfig(
                        name="one",
                        transport="legacy_json",
                        url="https://mcp.example.test/one",
                    ),
                    McpServerConfig(
                        name="two",
                        transport="legacy_json",
                        url="https://mcp.example.test/two",
                    ),
                ],
            )
            self.calls = []

    with (
        DuplicateFakeCaller() as caller,
        pytest.raises(ValueError, match="Duplicate MCP tool 'lookup'"),
    ):
        caller.get_filtered_schemas()


def test_streamable_http_initializes_once_and_reuses_session(monkeypatch):
    clients = []
    sessions = []

    @asynccontextmanager
    async def fake_streamable_http_client(url, **kwargs):
        assert url == "https://mcp.example.test"
        http_client = kwargs["http_client"]
        assert isinstance(http_client, httpx2.AsyncClient)
        assert http_client.headers["User-Agent"] == "brrragent/1.0"
        assert http_client.timeout.connect == 7
        assert http_client.timeout.read == 42
        clients.append(http_client)
        yield object(), object()

    class FakeSession:
        def __init__(self, *args, **kwargs):
            assert kwargs["read_timeout_seconds"] == 42
            self.initialize_count = 0
            self.call_count = 0
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            self.initialize_count += 1
            return SimpleNamespace(protocolVersion="2025-11-25")

        async def list_tools(self, *, params=None):
            assert params is None
            return SimpleNamespace(
                tools=[
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "inputSchema": {"type": "object"},
                    }
                ],
                next_cursor=None,
            )

        async def call_tool(self, name, arguments, **kwargs):
            assert kwargs["read_timeout_seconds"] == 42
            self.call_count += 1
            return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(
        mcp_tools,
        "streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr(mcp_tools, "ClientSession", FakeSession)

    with McpToolCaller(
        servers=[
            McpServerConfig(
                name="streamable-test",
                url="https://mcp.example.test",
                transport="streamable_http",
                auth=httpx2.BasicAuth("user", "password"),
                timeout=7,
                read_timeout=42,
            )
        ]
    ) as caller:
        assert [tool["name"] for tool in caller.get_filtered_schemas()] == ["lookup"]
        assert caller.call_tool("lookup", {}) == "ok"

    assert clients[0].is_closed
    assert len(sessions) == 1
    assert sessions[0].initialize_count == 1
    assert sessions[0].call_count == 1


def test_auto_transport_falls_back_to_sse(monkeypatch):
    attempts = []

    @asynccontextmanager
    async def failing_streamable_http_client(url, **kwargs):
        attempts.append("streamable_http")
        raise RuntimeError("not a Streamable HTTP server")
        yield

    @asynccontextmanager
    async def fake_sse_client(url, **kwargs):
        attempts.append("sse")
        yield object(), object()

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            return SimpleNamespace(protocolVersion="2025-11-25")

        async def list_tools(self, *, params=None):
            return SimpleNamespace(tools=[{"name": "lookup"}], next_cursor=None)

    monkeypatch.setattr(
        mcp_tools,
        "streamable_http_client",
        failing_streamable_http_client,
    )
    monkeypatch.setattr(mcp_tools, "sse_client", fake_sse_client)
    monkeypatch.setattr(mcp_tools, "ClientSession", FakeSession)

    with McpToolCaller(endpoint="https://mcp.example.test") as caller:
        assert [tool["name"] for tool in caller.get_filtered_schemas()] == ["lookup"]

    assert attempts == ["streamable_http", "sse"]


def test_list_all_tools_uses_v2_pagination_params():
    cursors = []

    class FakeSession:
        async def list_tools(self, *, params=None):
            cursor = params.cursor if params is not None else None
            cursors.append(cursor)
            if cursor is None:
                return SimpleNamespace(tools=[{"name": "first"}], next_cursor="next")
            return SimpleNamespace(tools=[{"name": "second"}], next_cursor=None)

    tools = asyncio.run(mcp_tools._list_all_tools(FakeSession()))

    assert tools == [{"name": "first"}, {"name": "second"}]
    assert cursors == [None, "next"]


def test_stdio_server_initializes_lists_and_calls_tool():
    server_code = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("stdio-test")

@server.tool()
def echo(value: str) -> str:
    return value

server.run(transport="stdio")
"""
    with McpToolCaller(
        servers=[
            McpServerConfig(
                name="stdio-test",
                transport="stdio",
                command=sys.executable,
                args=("-c", server_code),
            )
        ]
    ) as caller:
        assert [tool["name"] for tool in caller.get_filtered_schemas()] == ["echo"]
        assert caller.call_tool("echo", {"value": "hello"}) == "hello"


def _wait_for_server(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail(f"MCP server exited during startup:\n{stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail("MCP server did not start within 10 seconds")


@pytest.mark.parametrize(
    ("server_transport", "client_transport", "path"),
    [
        ("streamable-http", "streamable_http", "/mcp"),
        ("sse", "sse", "/sse"),
    ],
)
def test_http_server_initializes_lists_and_calls_tool(
    server_transport,
    client_transport,
    path,
):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]

    server_code = """
import sys
from mcp.server.mcpserver import MCPServer

server = MCPServer("http-test", log_level="ERROR")

@server.tool()
def echo(value: str) -> str:
    return value

server.run(
    transport=sys.argv[1],
    host="127.0.0.1",
    port=int(sys.argv[2]),
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", server_code, server_transport, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_server(process, port)
        with McpToolCaller(
            servers=[
                McpServerConfig(
                    name="http-test",
                    transport=client_transport,
                    url=f"http://127.0.0.1:{port}{path}",
                    timeout=5,
                    read_timeout=5,
                )
            ]
        ) as caller:
            assert [tool["name"] for tool in caller.get_filtered_schemas()] == ["echo"]
            assert caller.call_tool("echo", {"value": "hello"}) == "hello"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_concurrent_close_does_not_leave_a_caller_blocked():
    request_started = Event()
    release_request = Event()

    class BlockingCaller(FakeMcpToolCaller):
        def _legacy_request(self, server, payload, timeout):
            request_started.set()
            assert release_request.wait(timeout=2)
            return {"result": {"tools": []}}

    caller = BlockingCaller([])
    request_thread = Thread(target=caller.get_filtered_schemas)
    request_thread.start()
    assert request_started.wait(timeout=2)

    close_threads = [Thread(target=caller.close) for _ in range(2)]
    for thread in close_threads:
        thread.start()

    release_request.set()
    request_thread.join(timeout=2)
    for thread in close_threads:
        thread.join(timeout=2)

    assert not request_thread.is_alive()
    assert all(not thread.is_alive() for thread in close_threads)


def test_transport_cancellation_resolves_request_and_keeps_worker_alive():
    class CancelledCaller(FakeMcpToolCaller):
        async def _call(self, connections, *arguments):
            raise asyncio.CancelledError("transport failed")

    caller = CancelledCaller([])
    call_future = Future()
    close_future = Future()
    caller._commands.put(_WorkerCommand("call", ("server", "tool", {}), call_future))
    caller._commands.put(_WorkerCommand("close", (), close_future))

    asyncio.run(caller._worker())

    with pytest.raises(ConnectionError, match="cancelled by its transport"):
        call_future.result()
    assert close_future.result() is None
