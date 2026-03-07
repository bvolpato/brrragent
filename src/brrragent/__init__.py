"""
brrragent — Reusable agentic react loop with MCP tool calling.

Provides:
  - McpToolCaller: HTTP bridge to any MCP JSON-RPC endpoint (OpenAI + Gemini formats)
  - run_agent: Smart-routed agentic loop (auto-selects Gemini direct vs OpenRouter)
  - get_default_caller: Singleton McpToolCaller instance
"""

from brrragent.mcp_tools import McpToolCaller, get_default_caller
from brrragent.agent import run_agent

__all__ = ["McpToolCaller", "get_default_caller", "run_agent"]
