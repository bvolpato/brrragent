"""
brrragent — Reusable agentic react loop with MCP tool calling.

Provides:
  - McpToolCaller: Multi-server MCP bridge (OpenAI + Gemini formats)
  - McpServerConfig: Streamable HTTP, SSE, stdio, or legacy JSON configuration
  - run_agent: Smart-routed agentic loop (Gemini direct, OpenAI direct, Codex OAuth, OpenRouter)
  - get_default_caller: Singleton McpToolCaller instance
  - pick_key / pick_env_key: Handle comma-separated API keys
  - KeyPool: Thread-safe key pool with 429 eviction (works for any provider)
  - RateLimitExhausted: Exception when all keys are rate-limited

Backends (importable individually):
  - brrragent.openrouter: OpenRouter/OpenAI-compatible backend
  - brrragent.openai_direct: OpenAI native backend
  - brrragent.codex_oauth: ChatGPT/Codex OAuth backend via OpenCode auth
  - brrragent.gemini: Google Gemini direct backend
  - brrragent.keys: API key utilities
"""

from brrragent.agent import run_agent
from brrragent.keys import KeyPool, RateLimitExhausted, pick_env_key, pick_key
from brrragent.mcp_tools import McpServerConfig, McpToolCaller, get_default_caller
from brrragent.prompt_cache import AgentUsage, PromptCacheConfig

__all__ = [
    "AgentUsage",
    "KeyPool",
    "McpServerConfig",
    "McpToolCaller",
    "PromptCacheConfig",
    "RateLimitExhausted",
    "get_default_caller",
    "pick_env_key",
    "pick_key",
    "run_agent",
]
