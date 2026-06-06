"""
brrragent — Reusable agentic react loop with MCP tool calling.

Provides:
  - McpToolCaller: HTTP bridge to any MCP JSON-RPC endpoint (OpenAI + Gemini formats)
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

from brrragent.mcp_tools import McpToolCaller, get_default_caller
from brrragent.agent import run_agent
from brrragent.keys import pick_key, pick_env_key, KeyPool, RateLimitExhausted

__all__ = [
    "McpToolCaller",
    "get_default_caller",
    "run_agent",
    "pick_key",
    "pick_env_key",
    "KeyPool",
    "RateLimitExhausted",
]
