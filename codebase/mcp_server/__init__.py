"""
MCP Server for Personal Knowledge AI Assistant

Exposes RAG functionality as MCP tools that Claude Desktop can use.
"""

from .server import McpServer, start_mcp_server

__all__ = ["McpServer", "start_mcp_server"]
