"""
FastAPI REST API for Personal Knowledge AI Assistant

Run with: uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import time
import logging
import re
import uuid
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.schemas import (
    QueryRequest, QueryResponse, ChunkSource,
    SearchRequest, SearchResponse, SearchResult,
    DocumentInfo, StatsResponse,
    UploadResponse, ErrorResponse
)

# Import your existing RAG components
from rag_chat import ask, embed_query, get_top_k, filter_chunks, expand_context, get_demo_user_id, EXPAND_RADIUS
from reranker import rerank_chunks, RERANK_TOP_K
from loaders import load_file, is_supported, SUPPORTED_EXTENSIONS
from embed_chunks_pgvector import process_file, connect_db
import cache_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Security: Filename sanitization
def secure_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    - Strips directory components (keeps basename only)
    - Removes non-alphanumeric characters except dots, hyphens, underscores
    - Collapses multiple dots into one
    - Strips leading/trailing dots and whitespace

    Returns sanitized filename or empty string if invalid.
    """
    # Get basename to prevent path traversal
    filename = os.path.basename(filename)

    # Remove any non-alphanumeric characters except . - _
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Collapse multiple dots into one
    filename = re.sub(r'\.+', '.', filename)

    # Strip leading/trailing dots and whitespace
    filename = filename.strip('. ')

    return filename


# Helper function to parse allowed origins
def get_allowed_origins() -> List[str]:
    """
    Get allowed CORS origins from environment.

    Reads ALLOWED_ORIGINS env var as comma-separated list.
    Defaults to ["http://localhost:3000"] for development.
    """
    origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Personal Knowledge AI Assistant API...")
    app.state.user_id = get_demo_user_id()
    print(f"Using user_id: {app.state.user_id}")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Personal Knowledge AI Assistant",
    description="RAG-powered API for querying your personal knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
# Security: Uses explicit allowed origins from ALLOWED_ORIGINS env var (comma-separated).
# DO NOT use wildcard ["*"] with allow_credentials=True in production - this is invalid and unsafe.
# Set ALLOWED_ORIGINS="http://localhost:3000,https://yourdomain.com" in your environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "ok",
        "service": "Personal Knowledge AI Assistant",
        "endpoints": ["/query", "/search", "/documents", "/stats", "/upload"]
    }


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a question and get a RAG-powered answer.

    This endpoint:
    1. Embeds your question
    2. Retrieves relevant chunks from the vector database
    3. Optionally reranks chunks using LLM
    4. Optionally expands context with adjacent chunks
    5. Generates an answer using the LLM
    """
    start_time = time.time()
    user_id = app.state.user_id

    try:
        # Get full answer
        answer = ask(request.question, user_id)

        # Get source information for response
        query_embedding = embed_query(request.question)
        chunks, _ = get_top_k(query_embedding, user_id)
        filtered, _ = filter_chunks(chunks)

        if request.use_reranking:
            reranked, _ = rerank_chunks(request.question, filtered, top_k=RERANK_TOP_K)
        else:
            reranked = filtered[:RERANK_TOP_K]

        if request.expand_context:
            final_chunks = expand_context(reranked, user_id, radius=EXPAND_RADIUS)
        else:
            final_chunks = reranked

        # Build source list
        sources = []
        for chunk_tuple in reranked[:10]:  # Top 10 sources
            sources.append(ChunkSource(
                file=chunk_tuple[2],
                chunk_index=chunk_tuple[3],
                similarity=round(chunk_tuple[4], 4),
                header=chunk_tuple[1],
                preview=chunk_tuple[0][:200] + "..." if len(chunk_tuple[0]) > 200 else chunk_tuple[0]
            ))

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time_ms=processing_time,
            chunks_retrieved=len(filtered),
            chunks_used=len(final_chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents without generating an answer.
    Faster than /query - just returns ranked files.
    """
    user_id = app.state.user_id

    try:
        query_embedding = embed_query(request.query)
        chunks, total = get_top_k(query_embedding, user_id)

        # Aggregate by file
        file_data = {}
        for chunk_tuple in chunks:
            file_path = chunk_tuple[2]
            similarity = chunk_tuple[4]
            preview = chunk_tuple[0]

            if file_path not in file_data:
                file_data[file_path] = {
                    "max_similarity": similarity,
                    "preview": preview,
                    "chunk_count": 1
                }
            else:
                file_data[file_path]["chunk_count"] += 1
                if similarity > file_data[file_path]["max_similarity"]:
                    file_data[file_path]["max_similarity"] = similarity
                    file_data[file_path]["preview"] = preview

        # Sort and limit
        results = [
            SearchResult(
                file=f,
                similarity=round(d["max_similarity"], 4),
                preview=d["preview"][:200] + "..." if len(d["preview"]) > 200 else d["preview"],
                chunk_count=d["chunk_count"]
            )
            for f, d in sorted(file_data.items(), key=lambda x: -x[1]["max_similarity"])[:request.top_k]
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            total_files=len(file_data)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    user_id = app.state.user_id

    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        cur.execute("""
            SELECT source, COUNT(*) as chunks,
                   COALESCE(file_type, 'unknown') as file_type,
                   MAX(file_mtime) as indexed_at
            FROM chunks
            WHERE user_id = %s
            GROUP BY source, file_type
            ORDER BY source;
        """, (user_id,))

        documents = [
            DocumentInfo(
                file=row[0],
                chunks=row[1],
                file_type=row[2],
                indexed_at=row[3]
            )
            for row in cur.fetchall()
        ]

        return documents

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    user_id = app.state.user_id

    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Chunk count
        cur.execute("SELECT COUNT(*) FROM chunks WHERE user_id = %s;", (user_id,))
        chunk_count = cur.fetchone()[0]

        # File types
        cur.execute("""
            SELECT COALESCE(file_type, 'unknown'), COUNT(DISTINCT source)
            FROM chunks WHERE user_id = %s
            GROUP BY file_type;
        """, (user_id,))
        file_types = {row[0]: row[1] for row in cur.fetchall()}

        return StatsResponse(
            file_count=sum(file_types.values()),
            chunk_count=chunk_count,
            file_types=file_types
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    use_semantic: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Upload and index a new document.

    Supported formats: PDF, DOCX, TXT, MD, images, etc.

    The file is saved immediately and processing happens in the background.
    """
    user_id = app.state.user_id

    # Sanitize filename to prevent path traversal
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    # Check file type from sanitized filename
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    # Generate unique filename to avoid collisions
    name_without_ext = os.path.splitext(safe_filename)[0]
    unique_filename = f"{name_without_ext}_{uuid.uuid4().hex[:8]}{ext}"

    # Save uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_filename)

    try:
        # Write file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Dispatch processing to background task
        # Variables (file_path, user_id, use_semantic) are captured before adding task
        if background_tasks is not None:
            background_tasks.add_task(process_file, file_path, user_id, use_semantic=use_semantic)
            chunks_created = 0  # Background processing - count not yet available
            message = f"File {safe_filename} uploaded successfully. Processing in background."
        else:
            # Fallback to synchronous processing if BackgroundTasks not available
            chunk_count = process_file(file_path, user_id, use_semantic=use_semantic)
            # Validate and convert to int, default to 0 if None or invalid
            chunks_created = int(chunk_count) if chunk_count is not None else 0
            message = f"Successfully indexed {safe_filename}"

        return UploadResponse(
            success=True,
            file=safe_filename,
            chunks_created=chunks_created,
            message=message
        )

    except Exception as e:
        # Clean up the uploaded file if save failed
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up failed upload: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file {file_path}: {cleanup_error}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{file_path:path}")
async def delete_document(file_path: str):
    """Delete a document from the index."""
    user_id = app.state.user_id

    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        cur.execute(
            "DELETE FROM chunks WHERE user_id = %s AND source = %s;",
            (user_id, file_path)
        )
        deleted = cur.rowcount

        conn.commit()

        if deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        # Clean up physical file after successful DB deletion
        try:
            # Security: Extract only basename to prevent path traversal
            # file_path comes from the database's source field, which may contain
            # user-controlled input. Never trust it for filesystem operations.
            basename = os.path.basename(file_path)

            # Get upload directory (use app.state if configured, otherwise default)
            upload_dir = getattr(app.state, 'upload_dir', os.path.abspath('uploads'))
            upload_dir = os.path.realpath(upload_dir)  # Resolve symlinks

            # Construct path securely
            physical_path = os.path.join(upload_dir, basename)
            physical_path = os.path.realpath(physical_path)  # Resolve any symlinks/relative components

            # Verify the resolved path is within upload directory
            if not physical_path.startswith(upload_dir + os.sep):
                logger.warning(f"Security: Resolved path outside upload directory: {physical_path}")
            elif os.path.exists(physical_path):
                os.remove(physical_path)
                logger.info(f"Deleted physical file: {physical_path}")
            else:
                logger.info(f"Physical file not found (already deleted?): {physical_path}")
        except FileNotFoundError:
            # File already deleted, ignore
            pass
        except Exception as cleanup_error:
            # Log but don't fail the request - DB deletion succeeded
            logger.error(f"Failed to delete physical file for {file_path}: {cleanup_error}")

        return {"success": True, "deleted_chunks": deleted}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# Cache management
@app.post("/cache/clear")
async def clear_cache():
    """Clear the AI cache."""
    count = cache_manager.clear()
    return {"cleared": count}


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return cache_manager.stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
