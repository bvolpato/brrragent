"""MCP client and LLM tool-schema bridge."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import urllib.request
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from queue import Empty, Queue
from threading import Lock, RLock, Thread
from typing import Any, ClassVar, Literal, Self, cast

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams

logger = logging.getLogger(__name__)

McpTransport = Literal[
    "auto",
    "streamable_http",
    "sse",
    "stdio",
    "legacy_json",
]

_TRANSPORT_ALIASES = {
    "streamable-http": "streamable_http",
    "legacy-json": "legacy_json",
    "http": "streamable_http",
}
_VALID_TRANSPORTS = {"auto", "streamable_http", "sse", "stdio", "legacy_json"}
_VALID_PREFIX = re.compile(r"^[A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class McpServerConfig:
    """Connection details for one MCP server."""

    name: str
    transport: McpTransport | str = "auto"
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    auth: httpx2.Auth | None = None
    tool_prefix: str = ""
    timeout: float = 30
    read_timeout: float = 300

    def __post_init__(self) -> None:
        name = self.name.strip()
        transport = _TRANSPORT_ALIASES.get(str(self.transport), str(self.transport))
        if not name:
            raise ValueError("MCP server name cannot be empty")
        if transport not in _VALID_TRANSPORTS:
            raise ValueError(f"Unsupported MCP transport: {self.transport}")
        if transport == "stdio":
            if not self.command:
                raise ValueError(f"MCP stdio server {name!r} requires command")
            if self.headers or self.auth is not None:
                raise ValueError("MCP stdio servers use env, not HTTP headers or auth")
        elif not self.url:
            raise ValueError(f"MCP {transport} server {name!r} requires url")
        if not _VALID_PREFIX.fullmatch(self.tool_prefix):
            raise ValueError(
                "MCP tool_prefix may contain only letters, digits, _ and -"
            )
        if self.timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("MCP timeouts must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "headers", dict(self.headers))
        if self.env is not None:
            object.__setattr__(self, "env", dict(self.env))


@dataclass
class _Connection:
    transport: McpTransport
    session: ClientSession | None = None
    stack: AsyncExitStack | None = None


@dataclass
class _WorkerCommand:
    operation: str
    arguments: tuple[Any, ...]
    future: Future[Any]


class McpToolCaller:
    """Connect one or more MCP servers to LLM function calling.

    Standard transports use the official MCP SDK and retain initialized sessions
    for the caller lifetime. ``auto`` tries Streamable HTTP, legacy HTTP+SSE, then
    the pre-lifecycle raw JSON-RPC compatibility transport.
    """

    _HEADERS: ClassVar[dict[str, str]] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "brrragent/1.0",
    }

    def __init__(
        self,
        endpoint: str | None = None,
        allowed_tools: str | Iterable[str] | None = None,
        excluded_tools: str | Iterable[str] | None = None,
        max_response_len: int = 15_000,
        *,
        endpoints: Iterable[str] | None = None,
        servers: Iterable[McpServerConfig] | None = None,
        transport: McpTransport | str = "auto",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.servers = _resolve_servers(
            endpoint=endpoint,
            endpoints=endpoints,
            servers=servers,
            transport=transport,
            headers=headers,
        )
        self.endpoint = self.servers[0].url or ""
        self.endpoints = tuple(server.url for server in self.servers if server.url)
        self.allowed_tools = _normalize_patterns(allowed_tools, default={"*"})
        self.excluded_tools = _normalize_patterns(excluded_tools)
        self.max_response_len = max_response_len

        self._tool_schemas: list[dict[str, Any]] | None = None
        self._tool_routes: dict[str, tuple[str, str]] = {}
        self._cache_lock = RLock()
        self._worker_lock = Lock()
        self._commands: Queue[_WorkerCommand] = Queue()
        self._worker_thread: Thread | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close initialized MCP sessions and their transports."""
        with self._worker_lock:
            thread = self._worker_thread
            if thread is None:
                return
            future: Future[Any] = Future()
            self._commands.put(_WorkerCommand("close", (), future))
            future.result()
            thread.join()
            if self._worker_thread is thread:
                self._worker_thread = None

    def refresh_tools(self) -> list[dict[str, Any]]:
        """Refresh tool catalogs from every configured server."""
        with self._cache_lock:
            self._tool_schemas = None
            self._tool_routes = {}
        return self._fetch_tool_schemas()

    def _start_worker_locked(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_thread = Thread(
            target=lambda: asyncio.run(self._worker()),
            name="brrragent-mcp",
            daemon=True,
        )
        self._worker_thread.start()

    def _request(self, operation: str, *arguments: Any) -> Any:
        future: Future[Any] = Future()
        with self._worker_lock:
            self._start_worker_locked()
            self._commands.put(_WorkerCommand(operation, arguments, future))
        return future.result()

    async def _worker(self) -> None:
        connections: dict[str, _Connection] = {}
        try:
            while True:
                try:
                    command = self._commands.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    if command.operation == "close":
                        command.future.set_result(None)
                        break
                    if command.operation == "discover":
                        result = await self._discover(connections)
                    elif command.operation == "call":
                        result = await self._call(connections, *command.arguments)
                    else:
                        raise ValueError(
                            f"Unknown MCP worker operation: {command.operation}"
                        )
                except asyncio.CancelledError as exc:
                    command.future.set_exception(
                        ConnectionError("MCP operation was cancelled by its transport")
                    )
                    logger.debug("MCP transport cancelled an operation", exc_info=exc)
                except Exception as exc:  # noqa: BLE001 - propagate through Future
                    command.future.set_exception(exc)
                else:
                    command.future.set_result(result)
        finally:
            for connection in reversed(connections.values()):
                if connection.stack is not None:
                    try:
                        await connection.stack.aclose()
                    except BaseException:
                        logger.debug("MCP session close failed", exc_info=True)

    async def _discover(
        self,
        connections: dict[str, _Connection],
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
        schemas: list[dict[str, Any]] = []
        routes: dict[str, tuple[str, str]] = {}
        owners: dict[str, str] = {}

        for server in self.servers:
            connection = await self._ensure_connection(server, connections)
            if connection.session is None:
                tools = self._legacy_list_tools(server)
            else:
                tools = await _list_all_tools(connection.session)

            for tool in tools:
                schema = _model_dump(tool)
                remote_name = schema["name"]
                visible_name = f"{server.tool_prefix}{remote_name}"
                if visible_name in routes:
                    owner = owners[visible_name]
                    raise ValueError(
                        f"Duplicate MCP tool {visible_name!r} from {owner!r} and "
                        f"{server.name!r}; configure tool_prefix"
                    )
                schema["name"] = visible_name
                schemas.append(schema)
                routes[visible_name] = (server.name, remote_name)
                owners[visible_name] = server.name

            logger.info("MCP: discovered %d tools from %s", len(tools), server.name)

        return schemas, routes

    async def _call(
        self,
        connections: dict[str, _Connection],
        server_name: str,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        server = next(server for server in self.servers if server.name == server_name)
        connection = await self._ensure_connection(server, connections)
        if connection.session is None:
            return self._legacy_call_tool(server, remote_name, arguments)
        try:
            return await connection.session.call_tool(
                remote_name,
                arguments,
                read_timeout_seconds=server.read_timeout,
            )
        except BaseException:
            connections.pop(server.name, None)
            if connection.stack is not None:
                try:
                    await connection.stack.aclose()
                except BaseException:
                    logger.debug("MCP failed session cleanup failed", exc_info=True)
            raise

    async def _ensure_connection(
        self,
        server: McpServerConfig,
        connections: dict[str, _Connection],
    ) -> _Connection:
        if server.name in connections:
            return connections[server.name]

        transports = (
            ("streamable_http", "sse", "legacy_json")
            if server.transport == "auto"
            else (server.transport,)
        )
        errors: list[str] = []
        for transport in transports:
            try:
                connection = await self._open_connection(server, transport)
            except BaseException as exc:
                errors.append(f"{transport}: {exc}")
                logger.debug(
                    "MCP %s transport %s failed",
                    server.name,
                    transport,
                    exc_info=True,
                )
                continue
            connections[server.name] = connection
            logger.info("MCP: connected to %s via %s", server.name, transport)
            return connection

        raise ConnectionError(
            f"Could not connect to MCP server {server.name!r}: {'; '.join(errors)}"
        )

    async def _open_connection(
        self,
        server: McpServerConfig,
        transport: str,
    ) -> _Connection:
        if transport == "legacy_json":
            return _Connection(transport="legacy_json")

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if transport == "stdio":
                params = StdioServerParameters(
                    command=server.command or "",
                    args=list(server.args),
                    env=dict(server.env) if server.env is not None else None,
                    cwd=server.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params)
                )
            elif transport == "sse":
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(
                        server.url or "",
                        headers=dict(server.headers),
                        timeout=min(server.timeout, 5),
                        sse_read_timeout=server.read_timeout,
                        auth=server.auth,
                    )
                )
            else:
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers={"User-Agent": "brrragent/1.0", **server.headers},
                        auth=server.auth,
                        follow_redirects=True,
                        timeout=httpx2.Timeout(
                            server.timeout,
                            read=server.read_timeout,
                        ),
                    )
                )
                read_stream, write_stream = await stack.enter_async_context(
                    streamable_http_client(
                        server.url or "",
                        http_client=http_client,
                        terminate_on_close=True,
                    )
                )

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=server.read_timeout,
                )
            )
            await session.initialize()
            return _Connection(
                transport=cast(McpTransport, transport),
                session=session,
                stack=stack,
            )
        except BaseException:
            await stack.aclose()
            raise

    def _legacy_request(
        self,
        server: McpServerConfig,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        if server.auth is not None:
            raise ValueError(
                "MCP auth objects require streamable_http or sse transport"
            )
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            server.url or "",
            data=data,
            headers={**self._HEADERS, **server.headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def _legacy_list_tools(self, server: McpServerConfig) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            data = self._legacy_request(
                server,
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": params},
                server.timeout,
            )
            _raise_rpc_error(data)
            result = data.get("result", {})
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def _legacy_call_tool(
        self,
        server: McpServerConfig,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        data = self._legacy_request(
            server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            server.read_timeout,
        )
        _raise_rpc_error(data)
        return data.get("result", {})

    def _fetch_tool_schemas(self) -> list[dict[str, Any]]:
        with self._cache_lock:
            if self._tool_schemas is None:
                schemas, routes = self._request("discover")
                self._tool_schemas = schemas
                self._tool_routes = routes
            return self._tool_schemas

    def get_filtered_schemas(self) -> list[dict[str, Any]]:
        """Return discovered schemas matching allowed patterns minus exclusions."""
        if not self.allowed_tools:
            return []
        return [
            tool
            for tool in self._fetch_tool_schemas()
            if self.is_tool_allowed(tool["name"])
        ]

    def is_tool_allowed(self, name: str) -> bool:
        """Return whether a tool name matches inclusion and exclusion patterns."""
        return _matches_any(name, self.allowed_tools) and not self.is_tool_excluded(
            name
        )

    def is_tool_excluded(self, name: str) -> bool:
        """Return whether a tool name matches an exclusion pattern."""
        return _matches_any(name, self.excluded_tools)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Build OpenAI-compatible function definitions."""
        tools = []
        for schema in self.get_filtered_schemas():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema.get("description", "")[:500],
                        "parameters": _clean_schema(schema.get("inputSchema", {})),
                    },
                }
            )
        return tools

    def get_genai_tools(self):
        """Build google.genai Tool objects from MCP schemas."""
        try:
            from google.genai import types as gt
        except ImportError as exc:
            raise ImportError(
                "google-genai required: pip install google-genai"
            ) from exc

        declarations = []
        for schema in self.get_filtered_schemas():
            declarations.append(
                gt.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", "")[:500],
                    parameters=_clean_schema(schema.get("inputSchema", {})),
                )
            )
        return [gt.Tool(function_declarations=declarations)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute an allowed tool on its owning MCP server."""
        if not self.is_tool_allowed(name):
            return f"[Tool error: tool {name!r} is not allowed]"

        logger.info("MCP call: %s(%s)", name, list(arguments.keys()))
        try:
            self._fetch_tool_schemas()
            route = self._tool_routes.get(name)
            if route is None:
                return f"[Tool error: unknown MCP tool {name!r}]"
            result = self._request("call", *route, arguments)
            text = _result_to_text(result)
        except Exception as exc:  # noqa: BLE001 - tool errors are model-visible results
            logger.error("MCP tool %s failed: %s", name, exc)
            return f"[Tool error: {exc}]"

        if len(text) > self.max_response_len:
            text = text[: self.max_response_len] + "\n... [truncated]"
        return text


async def _list_all_tools(session: ClientSession) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    while True:
        params = PaginatedRequestParams(cursor=cursor) if cursor else None
        result = await session.list_tools(params=params)
        tools.extend(result.tools)
        cursor = result.next_cursor
        if not cursor:
            return tools


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return value.model_dump(by_alias=True, exclude_none=True)


def _raise_rpc_error(data: dict[str, Any]) -> None:
    if "error" not in data:
        return
    error = data["error"]
    raise RuntimeError(error.get("message", str(error)))


def _result_to_text(result: Any) -> str:
    result_data = _model_dump(result)
    content = result_data.get("content", [])
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            block_data = _model_dump(block)
            if block_data.get("type") == "text":
                parts.append(block_data.get("text", ""))
            else:
                parts.append(json.dumps(block_data, default=str))
    elif isinstance(content, str):
        parts.append(content)

    if parts:
        return "\n".join(parts)
    structured = result_data.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, default=str)
    return json.dumps(result_data, default=str)


def _clean_schema(schema: Any) -> Any:
    """Remove JSON Schema fields unsupported by LLM function declarations."""
    supported_keys = {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
    }
    if not isinstance(schema, dict):
        return schema
    cleaned = {key: value for key, value in schema.items() if key in supported_keys}
    if "properties" in cleaned:
        cleaned["properties"] = {
            key: _clean_schema(value) for key, value in cleaned["properties"].items()
        }
    if "items" in cleaned:
        cleaned["items"] = _clean_schema(cleaned["items"])
    return cleaned


def _resolve_servers(
    *,
    endpoint: str | None,
    endpoints: Iterable[str] | None,
    servers: Iterable[McpServerConfig] | None,
    transport: McpTransport | str,
    headers: Mapping[str, str] | None,
) -> tuple[McpServerConfig, ...]:
    supplied = sum(value is not None for value in (endpoint, endpoints, servers))
    if supplied > 1:
        raise ValueError("Pass only one of endpoint, endpoints, or servers")

    if servers is not None:
        resolved = tuple(servers)
    else:
        urls: list[str]
        if endpoint is not None:
            urls = [endpoint]
        elif endpoints is not None:
            urls = list(endpoints)
        else:
            urls_env = os.getenv("BRRRAGENT_MCP_URLS", "")
            if urls_env.strip():
                urls = urls_env.split(",")
            else:
                urls = [os.getenv("BRRRAGENT_MCP_URL", "")]
        urls = [url.strip() for url in urls if url.strip()]
        resolved = tuple(
            McpServerConfig(
                name=f"server_{index}",
                transport=transport,
                url=url,
                headers=headers or {},
            )
            for index, url in enumerate(urls, start=1)
        )

    if not resolved:
        raise ValueError(
            "MCP server required: pass endpoint, endpoints, or servers, or set "
            "BRRRAGENT_MCP_URL(S)"
        )
    names = [server.name for server in resolved]
    if len(names) != len(set(names)):
        raise ValueError("MCP server names must be unique")
    return resolved


def _normalize_patterns(
    patterns: str | Iterable[str] | None,
    *,
    default: set[str] | None = None,
) -> set[str]:
    if patterns is None:
        return set(default or ())
    if isinstance(patterns, str):
        return {patterns}
    return set(patterns)


def _matches_any(name: str, patterns: set[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


_default_caller: McpToolCaller | None = None


def get_default_caller() -> McpToolCaller:
    """Return the singleton MCP caller, initialized from environment URLs."""
    global _default_caller
    if _default_caller is None:
        _default_caller = McpToolCaller()
    return _default_caller


def _close_default_caller() -> None:
    if _default_caller is not None:
        _default_caller.close()


atexit.register(_close_default_caller)
