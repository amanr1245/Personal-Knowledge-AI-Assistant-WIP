"""
MCP Server for Personal Knowledge AI Assistant

Exposes RAG functionality as MCP tools that Claude Desktop can use.
Run with: python -m mcp_server.server
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional
from contextlib import contextmanager

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# MCP imports
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.responses import PlainTextResponse
import uvicorn
import anyio

# Your existing imports
from rag_chat import ask, embed_query, get_top_k, get_demo_user_id, expand_context, filter_chunks
from reranker import rerank_chunks, RERANK_TOP_K
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("Personal-Knowledge-Assistant")

# Global state
_user_id: Optional[int] = None


def _get_user_id() -> int:
    """Get or initialize user ID."""
    global _user_id
    if _user_id is None:
        _user_id = get_demo_user_id()
    return _user_id


def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    """
    Parse boolean from environment variable.

    Accepts: "1", "true", "yes" (case-insensitive) as True.
    Returns default if env var is not set.
    """
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Reads credentials from environment with sensible defaults.
    Automatically closes cursor and connection to prevent leaks.
    """
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME", "personal_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432"))
    )
    try:
        yield conn
    finally:
        conn.close()


@mcp.tool()
async def query_knowledge_base(question: str) -> Dict[str, Any]:
    """
    Ask a question and get a RAG-powered answer from the knowledge base.

    :param question: The question to ask.
    :return: {
        "question": str,
        "answer": str,
        "sources": [{"file": str, "chunk_index": int, "similarity": float}],
        "chunk_count": int
    }
    """
    user_id = _get_user_id()

    try:
        # Get answer using existing RAG pipeline
        answer = await asyncio.to_thread(ask, question, user_id)

        # Get source information
        query_embedding = await asyncio.to_thread(embed_query, question)
        chunks, _ = await asyncio.to_thread(get_top_k, query_embedding, user_id)
        filtered, _ = await asyncio.to_thread(filter_chunks, chunks)
        reranked, _ = await asyncio.to_thread(rerank_chunks, question, filtered, top_k=RERANK_TOP_K)

        sources = []
        for chunk_tuple in reranked[:5]:  # Top 5 sources
            # chunk_tuple: (chunk, header, source, chunk_index, similarity)
            sources.append({
                "file": chunk_tuple[2],
                "chunk_index": chunk_tuple[3],
                "similarity": round(chunk_tuple[4], 3)
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunk_count": len(reranked)
        }
    except Exception as e:
        logger.exception(f"Error in query_knowledge_base: {e}")
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "chunk_count": 0
        }


@mcp.tool()
async def search_documents(question: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search for relevant documents without generating an answer.
    Returns files ranked by relevance.

    :param question: Search query.
    :param top_k: Maximum number of results (default 10).
    :return: {"results": [{"file": str, "similarity": float, "preview": str}]}
    """
    user_id = _get_user_id()

    try:
        query_embedding = await asyncio.to_thread(embed_query, question)
        chunks, total = await asyncio.to_thread(get_top_k, query_embedding, user_id)

        # Deduplicate by file, keeping highest similarity
        file_scores = {}
        file_previews = {}

        for chunk_tuple in chunks[:top_k * 2]:  # Get extra for dedup
            # chunk_tuple: (chunk, header, source, chunk_index, similarity)
            file_path = chunk_tuple[2]
            similarity = chunk_tuple[4]
            chunk_text = chunk_tuple[0]

            if file_path not in file_scores or similarity > file_scores[file_path]:
                file_scores[file_path] = similarity
                file_previews[file_path] = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text

        # Sort by score and limit
        results = [
            {
                "file": f,
                "similarity": round(s, 3),
                "preview": file_previews[f]
            }
            for f, s in sorted(file_scores.items(), key=lambda x: -x[1])[:top_k]
        ]

        return {
            "query": question,
            "results": results,
            "total_chunks_searched": total
        }
    except Exception as e:
        logger.exception(f"Error in search_documents: {e}")
        return {"query": question, "results": [], "error": str(e)}


@mcp.tool()
async def list_indexed_files() -> Dict[str, Any]:
    """
    List all files currently indexed in the knowledge base.

    :return: {"files": [str], "count": int}
    """
    user_id = _get_user_id()

    def _fetch_files():
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT source, COUNT(*) as chunk_count
                    FROM chunks
                    WHERE user_id = %s
                    GROUP BY source
                    ORDER BY source;
                """, (user_id,))
                return [{"file": row[0], "chunks": row[1]} for row in cur.fetchall()]

    try:
        files = await asyncio.to_thread(_fetch_files)

        return {
            "files": files,
            "count": len(files),
            "total_chunks": sum(f["chunks"] for f in files)
        }
    except Exception as e:
        logger.exception(f"Error in list_indexed_files: {e}")
        return {"files": [], "count": 0, "error": str(e)}


@mcp.tool()
async def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.

    :return: {"file_count": int, "chunk_count": int, "file_types": {ext: count}}
    """
    user_id = _get_user_id()

    def _fetch_stats():
        from collections import Counter

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get chunk count
                cur.execute("SELECT COUNT(*) FROM chunks WHERE user_id = %s;", (user_id,))
                chunk_count = cur.fetchone()[0]

                # Get file count and types
                cur.execute("SELECT DISTINCT source FROM chunks WHERE user_id = %s;", (user_id,))
                sources = [row[0] for row in cur.fetchall()]

        file_types = Counter()
        for source in sources:
            ext = os.path.splitext(source)[1].lower() or "unknown"
            file_types[ext] += 1

        return {
            "file_count": len(sources),
            "chunk_count": chunk_count,
            "file_types": dict(file_types)
        }

    try:
        return await asyncio.to_thread(_fetch_stats)
    except Exception as e:
        logger.exception(f"Error in get_index_stats: {e}")
        return {"error": str(e)}


class McpServer:
    """MCP Server wrapper."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8008, debug: Optional[bool] = None):
        self.host = host
        self.port = port

        # Resolve debug flag: parameter > env var > default (False)
        if debug is not None:
            resolved_debug = debug
        else:
            resolved_debug = _parse_bool_env("APP_DEBUG", default=False)

        # Get the internal MCP server
        mcp_server = mcp._mcp_server
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> PlainTextResponse:
            try:
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,
                ) as (read_stream, write_stream):
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )
            except anyio.BrokenResourceError:
                logger.info("SSE client disconnected")
            except Exception:
                logger.exception("Unhandled error in SSE handler")

            return PlainTextResponse("ok")

        self.app = Starlette(
            debug=resolved_debug,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        logger.info(f"MCP server configured on http://{self.host}:{self.port}/")

    def start(self):
        """Start the MCP server."""
        print(f"\n{'='*60}")
        print(f"Personal Knowledge AI Assistant - MCP Server")
        print(f"{'='*60}")
        print(f"Server running on: http://{self.host}:{self.port}/")
        print(f"SSE endpoint:      http://{self.host}:{self.port}/sse")
        print(f"\nAvailable tools:")
        print(f"  - query_knowledge_base(question) - Full RAG query")
        print(f"  - search_documents(question)     - Document search")
        print(f"  - list_indexed_files()           - List all files")
        print(f"  - get_index_stats()              - Index statistics")
        print(f"{'='*60}")
        print(f"\nAdd to Claude Desktop config:")
        print(f'  "personal-rag": {{"url": "http://{self.host}:{self.port}/sse"}}')
        print(f"{'='*60}\n")

        uvicorn.run(self.app, host=self.host, port=self.port)


def start_mcp_server(host: str = "127.0.0.1", port: int = 8008, debug: Optional[bool] = None):
    """Convenience function to start the MCP server."""
    server = McpServer(host=host, port=port, debug=debug)
    server.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the MCP server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    start_mcp_server(host=args.host, port=args.port, debug=args.debug if args.debug else None)
