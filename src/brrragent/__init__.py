"""
brrragent — Reusable agentic react loop with MCP tool calling.

Provides:
  - McpToolCaller: HTTP bridge to any MCP JSON-RPC endpoint
  - run_agent: OpenRouter/OpenAI-compatible agentic tool-calling loop
"""

from brrragent.mcp_tools import McpToolCaller, get_default_caller
from brrragent.agent import run_agent

__all__ = ["McpToolCaller", "get_default_caller", "run_agent"]
