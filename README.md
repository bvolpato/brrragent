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

Install from PyPI:

```bash
uv add brrragent
```

Gemini support uses an optional dependency:

```bash
uv add "brrragent[gemini]"
```

Use `uv pip install` instead when installing into an existing environment.

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

## Image inputs

`run_agent` accepts one or more images with Codex OAuth, OpenAI, OpenRouter,
OpenAI-compatible providers, and Gemini:

```python
from brrragent import ImageInput, run_agent

result = run_agent(
    system_prompt="Return a concise image description.",
    user_prompt="What is shown?",
    model="codex/gpt-5.6-luna:medium",
    images=[ImageInput.from_file("photo.jpg", detail="low")],
)
```

Use `ImageInput(url)` for remote images, `ImageInput.from_base64(...)` for
existing Base64 data, or `ImageInput.from_bytes(...)` for in-memory content.
Multiple images preserve their input order. Detail can be `auto`, `low`,
`high`, or `original`; provider and model support still applies. Codex Spark
does not accept image input.

Provider credentials are read from the caller's environment. Common variables
are `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENLUX_API_KEY`, and
`GEMINI_API_KEY`.

Use `openlux/<model>` for OpenLux-native routing. Legacy `yunwu/<model>`
aliases remain accepted; `YUNWU_API_KEY` is used only when
`OPENLUX_API_KEY` is unset.

## Codex OAuth

Create brrragent-owned Codex credentials with device login:

```bash
uv run brrragent-auth login
```

This writes direct OAuth credentials to `~/.cache/brrragent/codex-auth.json`
and Codex CLI credentials to `~/.cache/brrragent/codex-home/auth.json`, both
with `0600` permissions. Default routing does not read shared OpenCode auth or
`~/.codex/auth.json`.

Override locations with `BRRRAGENT_AUTH_PATH`, `BRRRAGENT_CODEX_HOME`, or
command flags. Existing callers can still use `BRRRAGENT_OPENCODE_AUTH_PATH`
and explicit `CODEX_HOME` values.

Use one independently issued profile per application or worker group to avoid
refresh-token rotation conflicts:

```bash
uv run brrragent-auth login --profile content-worker
BRRRAGENT_PROFILE=content-worker uv run your-application
```

Prefer per-call selection when one process handles multiple use cases:

```python
from brrragent import CodexAuthConfig, run_agent

result = run_agent(
    system_prompt="Return a concise answer.",
    user_prompt="Summarize this input.",
    model="codex/gpt-5.6-luna:medium",
    codex_auth=CodexAuthConfig(profile="content-worker"),
)
```

Explicit paths are also supported with
`CodexAuthConfig(auth_path="...", codex_home="...")`. Per-call selection is
context-local, so concurrent use cases do not need to mutate process-wide
environment variables.

Named profiles live under `~/.cache/brrragent/profiles/<profile>/`. Run device
login separately for every profile; copying an auth file also copies its
rotating refresh token and does not provide isolation. `BRRRAGENT_PROFILE_ROOT`
can move the profile directory. A named profile also takes precedence over a
generic `CODEX_HOME` value.

## Prompt caching

Keep reusable tools and system instructions before request-specific content.
Use one versioned key per stable prompt without user IDs, timestamps, URLs, or
prompt text:

```python
from brrragent import PromptCacheConfig

cache = PromptCacheConfig(key="workflow:v2", ttl="1h")
```

Pass `prompt_cache=cache` to `run_agent`. brrragent maps it to supported cache
keys, explicit breakpoints, provider affinity, and Anthropic cache controls.
Gemini uses its stable system-first prefix for implicit caching. An optional
`on_usage` callback receives normalized token and cache usage for each request.

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

HTTP auth objects in `McpServerConfig` use `httpx2.Auth`, matching MCP SDK v2.
HTTP transports use the operating system trust store; set `SSL_CERT_FILE` or
`SSL_CERT_DIR` for custom certificate authorities.

`BRRRAGENT_MCP_URL` configures one default endpoint. `BRRRAGENT_MCP_URLS`
accepts comma-separated endpoints.

Tool filters accept exact names and case-sensitive shell patterns such as
`*_search`. Exclusions apply during discovery and execution.

## Public API

```python
from brrragent import (
    AgentUsage,
    KeyPool,
    McpServerConfig,
    McpToolCaller,
    PromptCacheConfig,
    RateLimitExhausted,
    get_default_caller,
    pick_env_key,
    pick_key,
    run_agent,
)
```

## Requirements

- Python 3.12 or newer
- `httpx>=0.27.1,<1`
- `httpx2>=2.5,<3`
- `mcp>=2.0.0,<3`
- `openai>=2.45.0,<4`
- `google-genai>=1.0.0,<3` for the `gemini` extra

## License

MIT
