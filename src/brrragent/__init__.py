"""
brrragent — Reusable agentic react loop with MCP tool calling.

Provides:
  - McpToolCaller: HTTP bridge to any MCP JSON-RPC endpoint (OpenAI + Gemini formats)
  - run_agent: Smart-routed agentic loop (auto-selects Gemini direct vs OpenRouter)
  - get_default_caller: Singleton McpToolCaller instance
  - pick_key / pick_env_key: Handle comma-separated API keys

Backends (importable individually):
  - brrragent.openrouter: OpenRouter/OpenAI-compatible backend
  - brrragent.gemini: Google Gemini direct backend
  - brrragent.keys: API key utilities
"""

from brrragent.mcp_tools import McpToolCaller, get_default_caller
from brrragent.agent import run_agent
from brrragent.keys import pick_key, pick_env_key

__all__ = [
    "McpToolCaller",
    "get_default_caller",
    "run_agent",
    "pick_key",
    "pick_env_key",
]
