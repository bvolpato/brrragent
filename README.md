# brrragent

Reusable agentic react loop with MCP tool calling. Supports **OpenRouter**, **OpenAI direct**, and **Google Gemini direct** backends with automatic smart routing.

## Features

- **Smart routing** — automatically picks the cheapest/best backend based on model name and available API keys
- **Comma-separated API keys** — spread load across multiple keys for rate-limit resilience and caching
- **MCP tool calling** — bridges any MCP JSON-RPC endpoint into LLM function-calling (both OpenAI and Gemini formats)
- **Structured output** — JSON schema enforcement on OpenRouter/OpenAI (`response_format`) and Gemini (`response_schema`)
- **Retry with backoff** — exponential backoff on transient errors (429, 503, etc.)
- **Minimal dependencies** — only `openai` is required; `google-genai` is optional

## Installation

```bash
# OpenRouter / OpenAI only
uv pip install /path/to/brrragent

# With Gemini direct support
uv pip install "/path/to/brrragent[gemini]"
```

Or add to your `pyproject.toml`:
```toml
dependencies = [
    "brrragent @ file:///path/to/brrragent",
]
```

## Quick Start

```python
from brrragent import run_agent

result = run_agent(
    system_prompt="You are a helpful research assistant.",
    user_prompt="What are the latest developments in quantum computing?",
    model="google/gemini-3.1-pro-preview",
)
print(result)
```

That's it. The agent will:
1. Detect `google/` prefix → check if `GEMINI_API_KEY` is set → use Gemini direct (no OpenRouter premium)
2. Detect `openai/` prefix → check if an OpenAI key is set → use OpenAI direct with reasoning suffix support
3. Fall back to OpenRouter if direct-provider keys are missing
4. Pick a random key if the env var contains comma-separated keys
4. Use MCP tools (web search, news, etc.) to research and answer

## Smart Routing

The routing decision is automatic based on model name and available environment variables:

| Model | Direct-provider key set? | Backend used |
|---|---|---|
| `google/gemini-3.1-pro-preview` | ✅ | **Gemini direct** (free, no OpenRouter premium) |
| `google/gemini-3.1-pro-preview` | ❌ | **OpenRouter** (uses `OPENROUTER_API_KEY`) |
| `openai/gpt-5.5:xhigh` | ✅ | **OpenAI direct** (uses `OPENAI_API_KEY_PERSONAL`, `OPENAI_API_KEY`, or `OPENAI_API_KEY_CORP`) |
| `openai/gpt-5.5:xhigh` | ❌ | **OpenRouter** (uses `OPENROUTER_API_KEY`) |
| `anthropic/claude-4.6-sonnet` | — | **OpenRouter** (always) |

You can override routing by passing `api_key` explicitly:
```python
result = run_agent(
    system_prompt="...",
    user_prompt="...",
    model="google/gemini-3.1-pro-preview",
    api_key="sk-or-v1-my-specific-key",  # forces OpenRouter
)
```

## Environment Variables

`brrragent` reads these from the **caller's environment** — set them in your app's `.env`, not in brrragent itself:

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | One of these | OpenRouter API key(s), comma-separated |
| `OPENAI_API_KEY_PERSONAL` | One of these | Preferred direct OpenAI API key(s), comma-separated |
| `OPENAI_API_KEY` | One of these | Direct OpenAI API key(s), comma-separated |
| `OPENAI_API_KEY_CORP` | One of these | Fallback direct OpenAI API key(s), comma-separated |
| `GEMINI_API_KEY` | One of these | Google Gemini API key(s), comma-separated |
| `MCP_CRAWLERPY_URL` | No | MCP endpoint (defaults to `https://crawlerpy.brunovolpato.com/mcp`) |

### Comma-Separated Keys

`OPENROUTER_API_KEY`, OpenAI key vars, and `GEMINI_API_KEY` support multiple keys separated by commas. One key is picked at random per `run_agent` call to spread load and improve rate-limit resilience:

```bash
OPENROUTER_API_KEY=sk-or-v1-key1,sk-or-v1-key2,sk-or-v1-key3
OPENAI_API_KEY_PERSONAL=sk-key1,sk-key2
GEMINI_API_KEY=AIzaSy-key1,AIzaSy-key2
```

## API Reference

### `run_agent(**kwargs) → str`

The main entrypoint. Runs an agentic react loop with tool calling.

```python
from brrragent import run_agent

result = run_agent(
    system_prompt="You are a research assistant.",
    user_prompt="Find the latest AI news and summarize.",
    model="google/gemini-3.1-pro-preview",  # default
    max_turns=12,          # max tool-calling rounds
    temperature=0.7,       # sampling temperature
    max_tokens=4096,       # max tokens per response
    max_retries=3,         # retries on transient errors
    on_tool_call=my_cb,    # optional callback(name, args)
    response_schema={...}, # optional JSON schema for structured output
)
```

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | *required* | System instructions |
| `user_prompt` | `str` | *required* | User's request |
| `model` | `str` | `google/gemini-3.1-pro-preview` | Model identifier |
| `api_key` | `str \| None` | `None` | Override API key (skips smart routing) |
| `base_url` | `str` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `mcp` | `McpToolCaller \| None` | singleton | MCP tool caller instance |
| `extra_tools` | `list[dict] \| None` | `None` | Additional OpenAI-format tool defs |
| `max_turns` | `int` | `12` | Max tool-calling rounds |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `4096` | Max tokens per response |
| `max_retries` | `int` | `3` | Retries on transient errors |
| `on_tool_call` | `Callable \| None` | `None` | Callback `(tool_name, tool_args)` |
| `response_schema` | `dict \| None` | `None` | JSON Schema for structured output |

### `McpToolCaller`

HTTP bridge to any MCP JSON-RPC endpoint. Discovers tools and converts them to function-calling definitions.

```python
from brrragent import McpToolCaller

# Default endpoint + default tools (web_search, news_search, read_url, youtube_search)
mcp = McpToolCaller()

# Custom endpoint and tools
mcp = McpToolCaller(
    endpoint="https://my-mcp-server.com/mcp",
    allowed_tools={"web_search", "my_custom_tool"},
    max_response_len=20_000,
)

# Get tool definitions for different backends
openai_tools = mcp.get_openai_tools()   # OpenAI/OpenRouter format
genai_tools = mcp.get_genai_tools()     # Google genai format

# Execute a tool
result = mcp.call_tool("web_search", {"q": "latest AI news"})
```

### `pick_key(keys_csv: str) → str`

Pick a random key from a comma-separated string:

```python
from brrragent import pick_key

key = pick_key("key1,key2,key3")  # returns one at random
```

### `pick_env_key(env_var: str) → str`

Read a comma-separated env var and pick one key at random:

```python
from brrragent import pick_env_key

key = pick_env_key("OPENROUTER_API_KEY")  # raises ValueError if not set
```

## Structured Output

Both backends support structured output via JSON Schema:

```python
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "confidence", "sources"],
}

result = run_agent(
    system_prompt="Analyze and return structured JSON.",
    user_prompt="What is the current price of Bitcoin?",
    response_schema=schema,
)

import json
data = json.loads(result)
print(data["summary"])
print(data["confidence"])
```

Under the hood:
- **OpenRouter**: uses `response_format` with `json_schema`
- **Gemini direct**: uses `response_mime_type="application/json"` + `response_schema`

## Package Structure

```
brrragent/src/brrragent/
├── __init__.py      # Public API: run_agent, McpToolCaller, pick_key, pick_env_key
├── agent.py         # Smart router — delegates to the right backend
├── keys.py          # Comma-separated API key utilities
├── gemini.py        # Google Gemini direct backend (requires google-genai)
├── openrouter.py    # OpenRouter / OpenAI-compatible backend
└── mcp_tools.py     # MCP JSON-RPC bridge (tool discovery + execution)
```

## Requirements

- Python ≥ 3.12
- `openai` ≥ 1.0.0 (always required)
- `google-genai` ≥ 1.0.0 (optional, install with `pip install brrragent[gemini]`)
