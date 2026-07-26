# brrragent

`brrragent` is a Python package for running a synchronous model/tool loop. It
provides a common entry point for several model APIs and converts MCP tools into
the function-calling formats expected by those APIs.

## Overview

Main components:

- `run_agent`: executes model responses and tool calls until completion.
- `McpToolCaller`: discovers, filters, routes, and executes tools across one or
  more MCP servers.
- `McpServerConfig`: configures Streamable HTTP, legacy HTTP+SSE, stdio, or raw
  JSON-RPC compatibility connections.
- `KeyPool`: selects API keys and removes rate-limited keys from rotation.

Model adapters cover OpenAI, OpenAI-compatible APIs, Google Gemini, OpenRouter,
and the optional Codex OAuth backend. Structured outputs, retry handling, tool
result truncation, and provider routing are handled by the package.

## Installation

From a checkout:

```bash
uv pip install /path/to/brrragent
```

Gemini support uses an optional dependency:

```bash
uv pip install "/path/to/brrragent[gemini]"
```

## Basic use

```python
from brrragent import McpToolCaller, run_agent

with McpToolCaller(endpoint="https://mcp.example.com/mcp") as mcp:
    result = run_agent(
        system_prompt="Follow the request and use available tools when needed.",
        user_prompt="Return a short response.",
        model="openai/gpt-5.5",
        mcp=mcp,
    )

print(result)
```

Provider credentials are read from the caller's environment. Common variables
are `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and `GEMINI_API_KEY`.

## MCP servers

One caller can combine multiple endpoints:

```python
from brrragent import McpToolCaller

mcp = McpToolCaller(
    endpoints=[
        "https://mcp.example.com/one",
        "https://mcp.example.com/two",
    ],
    allowed_tools={"*"},
    excluded_tools={"internal_*"},
)
```

Use explicit server configuration to mix transports or prefix tool names:

```python
from brrragent import McpServerConfig, McpToolCaller

mcp = McpToolCaller(
    servers=[
        McpServerConfig(
            name="remote",
            url="https://mcp.example.com/mcp",
            tool_prefix="remote_",
        ),
        McpServerConfig(
            name="local",
            transport="stdio",
            command="uvx",
            args=("example-mcp-server",),
            tool_prefix="local_",
        ),
    ]
)
```

`transport="auto"` tries Streamable HTTP, legacy HTTP+SSE, then raw JSON-RPC
POST compatibility. Standard transports use MCP initialization, version
negotiation, persistent sessions, and clean shutdown. Duplicate visible tool
names require `tool_prefix` configuration.

`BRRRAGENT_MCP_URL` configures one default endpoint. `BRRRAGENT_MCP_URLS`
accepts comma-separated endpoints.

Tool filters accept exact names and case-sensitive shell patterns such as
`*_search`. Exclusions apply during discovery and execution.

## Public API

```python
from brrragent import (
    KeyPool,
    McpServerConfig,
    McpToolCaller,
    RateLimitExhausted,
    get_default_caller,
    pick_env_key,
    pick_key,
    run_agent,
)
```

## Requirements

- Python 3.12 or newer
- `mcp>=1.28.1,<2`
- `openai>=1.66.0`
- `google-genai>=1.0.0` for the `gemini` extra

## License

MIT
