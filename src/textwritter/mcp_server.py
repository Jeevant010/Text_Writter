"""MCP server — exposes the handwriting agent as tools for MCP clients
(Claude Desktop, Cursor, etc.).

IMPORTANT (see docs/AGENT_SERVER_ROADMAP.md): this is OPTIONAL. Your production
website needs the FastAPI wrapper, not MCP. This server exists so an LLM client
on your own machine can drive the agent conversationally ("write my physics
notes in my handwriting, messier on page 2") during development.

Run:  python -m textwritter.mcp_server        (stdio transport)
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("textwritter")


@mcp.tool()
def enroll_style(name: str, photo_path: str) -> str:
    """Enroll a handwriting style from a photo of the user's writing.

    Returns a style_profile_id used by all later calls.
    """
    # TODO(phase-1): call enroll pipeline (preprocess -> style profile -> cache)
    raise NotImplementedError("enroll pipeline lands in Phase 1")


@mcp.tool()
def write_document(style_profile_id: str, content: str, mode: str = "verbatim") -> str:
    """Start the LangGraph satisfaction loop for a document.

    mode: 'verbatim' (transcribe as-is) or 'ai_writer' (Gemini drafts from topic).
    Returns a session_id; the graph pauses at human review.
    """
    raise NotImplementedError("wire src/textwritter/agent/loop.py here")


@mcp.tool()
def review_document(session_id: str, approved: bool, feedback: str = "") -> str:
    """Resume a paused session: approve for export, or pass tweak feedback
    ('make block 2 messier, more slant') to loop again."""
    raise NotImplementedError


@mcp.tool()
def reroll_block(session_id: str, block_id: str, seed: int | None = None) -> str:
    """Re-generate a single block with a new seed (the 'I don't like this word' button)."""
    raise NotImplementedError


if __name__ == "__main__":
    mcp.run()  # stdio
