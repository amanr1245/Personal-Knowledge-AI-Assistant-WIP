import os
import sys
import re
from datetime import datetime
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from chunk_pdf import chunk_pdf
from semantic_chunker import chunk_semantically_from_pdf, chunk_semantically
from loaders import load_file, SUPPORTED_EXTENSIONS, is_supported
from processors import EmbedProcessor
from file_tracker import (
    compute_file_hash,
    scan_directory,
    categorize_files,
    get_files_to_process,
    print_file_status_summary
)
import cache_manager
from retry_manager import with_retry
import time

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# OpenAI client
# ----------------------------
# Validate OPENAI_API_KEY is present
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required but not set. "
        "Please set it in your .env file or environment."
    )

openai_client = OpenAI(api_key=openai_api_key)

PDF_PATH = os.getenv("PDF_PATH")
BATCH_SIZE = 2048  # OpenAI allows up to 2048 embeddings per request

# Parallel embedding settings
PARALLEL_EMBED = os.getenv("EMBED_PARALLEL", "true").lower() == "true"
PARALLEL_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "8"))
PARALLEL_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))


# ----------------------------
# Connect to PostgreSQL
# ----------------------------
def connect_db():
    """
    Connect to PostgreSQL using environment variables.

    Environment variables:
    - DB_NAME: Database name (default: personal_rag)
    - DB_USER: Database user (default: postgres)
    - DB_PASSWORD: Database password (default: postgres)
    - DB_HOST: Database host (default: localhost)
    - DB_PORT: Database port (default: 5432)
    """
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "personal_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432"))
    )


# ----------------------------
# Get or create demo user
# ----------------------------
def get_or_create_demo_user():
    """
    Get the demo user's ID, creating the user if needed.

    Uses upsert pattern to avoid race conditions between SELECT and INSERT.
    Requires unique constraint on users.api_key column.
    """
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Try to insert, ignore if exists (requires unique constraint on api_key)
        cur.execute("""
            INSERT INTO users (name, api_key)
            VALUES ('Demo User', 'demo-api-key')
            ON CONFLICT (api_key) DO NOTHING
            RETURNING id;
        """)
        result = cur.fetchone()

        if result:
            # Successfully inserted new user
            user_id = result[0]
            conn.commit()
        else:
            # Conflict occurred (user already exists), fetch existing id
            cur.execute("SELECT id FROM users WHERE api_key = 'demo-api-key';")
            user_id = cur.fetchone()[0]

        return user_id
    finally:
        cur.close()
        conn.close()


# ----------------------------
# Get existing file hashes from database
# ----------------------------
def get_existing_files(user_id: int) -> dict:
    """
    Get existing file hashes from database for a user.

    Args:
        user_id: User ID to query

    Returns:
        Dict mapping source path -> content_hash
    """
    conn = None
    cur = None

    try:
        conn = connect_db()
        cur = conn.cursor()

        cur.execute("""
            SELECT DISTINCT source, content_hash
            FROM chunks
            WHERE user_id = %s AND content_hash IS NOT NULL;
        """, (user_id,))

        result = {row[0]: row[1] for row in cur.fetchall()}
        return result
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# ----------------------------
# Delete chunks by source file
# ----------------------------
def delete_chunks_by_source(source: str, user_id: int) -> int:
    """
    Delete all chunks for a source file.

    Args:
        source: Source file path
        user_id: User ID

    Returns:
        Number of chunks deleted
    """
    conn = None
    cur = None

    try:
        conn = connect_db()
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM chunks
            WHERE source = %s AND user_id = %s;
        """, (source, user_id))

        deleted = cur.rowcount
        conn.commit()

        return deleted
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# ----------------------------
# Batch embed using OpenAI API (with caching and parallel support)
# ----------------------------
def embed_batch(texts, parallel=None):
    """
    Embed a batch of texts using OpenAI API with caching.

    Args:
        texts: List of texts to embed
        parallel: Use parallel processing (default from env EMBED_PARALLEL)

    Returns:
        List of embedding vectors
    """
    use_parallel = parallel if parallel is not None else PARALLEL_EMBED

    if use_parallel:
        return embed_batch_parallel(texts)
    else:
        return embed_batch_sequential(texts)


def embed_batch_parallel(texts):
    """Embed texts using parallel processing with EmbedProcessor."""
    print(f"Embedding {len(texts)} texts in parallel...")

    processor = EmbedProcessor(
        max_workers=PARALLEL_WORKERS,
        batch_size=PARALLEL_BATCH_SIZE
    )

    embeddings = processor.embed_batch_parallel(texts)
    return embeddings


def embed_batch_sequential(texts):
    """Embed a batch of texts sequentially using OpenAI API with caching."""
    all_embeddings = []
    texts_to_embed = []
    text_indices = []  # Track which indices need embedding

    # Check cache for each text
    for i, text in enumerate(texts):
        cached = cache_manager.get_cached_embedding(text)
        if cached is not None:
            all_embeddings.append((i, cached))
        else:
            texts_to_embed.append(text)
            text_indices.append(i)

    cache_hits = len(texts) - len(texts_to_embed)
    if cache_hits > 0:
        print(f"Cache hits: {cache_hits}/{len(texts)} embeddings")

    # Embed uncached texts
    if texts_to_embed:
        new_embeddings = []
        for i in range(0, len(texts_to_embed), BATCH_SIZE):
            batch = texts_to_embed[i:i+BATCH_SIZE]
            print(f"Embedding batch {i//BATCH_SIZE + 1}/{(len(texts_to_embed)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)...")

            response = with_retry(
                openai_client.embeddings.create,
                model="text-embedding-3-small",
                input=batch
            )

            embeddings = [item.embedding for item in response.data]

            # Validate embedding count matches batch size
            if len(embeddings) != len(batch):
                error_msg = (
                    f"Embedding count mismatch: expected {len(batch)} embeddings, "
                    f"but received {len(embeddings)} from API"
                )
                print(f"ERROR: {error_msg}")
                raise ValueError(error_msg)

            new_embeddings.extend(embeddings)

        # Validate total embeddings match texts to embed
        if len(new_embeddings) != len(texts_to_embed):
            error_msg = (
                f"Total embedding count mismatch: expected {len(texts_to_embed)} embeddings, "
                f"but got {len(new_embeddings)} after processing all batches"
            )
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        # Cache new embeddings and add to results
        for idx, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
            cache_manager.cache_embedding(text, embedding)
            all_embeddings.append((text_indices[idx], embedding))

    # Sort by original index and extract just embeddings
    all_embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in all_embeddings]


# ----------------------------
# Chunk text content
# ----------------------------
def chunk_text(text, max_len=500, overlap=100):
    """
    Chunk text into segments using sentence-based splitting.

    Args:
        text: Text content to chunk
        max_len: Maximum chunk length in characters
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_len and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ----------------------------
# Process any supported file: load, chunk, embed, insert
# ----------------------------
def process_file(path, user_id, use_semantic=False, content_hash=None):
    """
    Process any supported file: load, chunk, embed, and store in database.

    Args:
        path: Path to file (PDF, image, text, or document)
        user_id: User ID to associate chunks with
        use_semantic: If True, use AI-powered semantic chunking with headers
        content_hash: Pre-computed content hash (computed if None)

    Returns:
        int: Number of chunks created and stored
    """
    ext = os.path.splitext(path)[1].lower()

    # Compute content hash if not provided
    if content_hash is None:
        content_hash = compute_file_hash(path)

    # Get file modification time
    file_mtime = datetime.fromtimestamp(os.path.getmtime(path))

    # For PDFs, use existing optimized pipeline
    if ext == '.pdf':
        if use_semantic:
            print(f"Semantic chunking {path}...")
            chunk_data = chunk_semantically_from_pdf(path)
            headers = [header for header, _ in chunk_data]
            chunks = [text for _, text in chunk_data]
            texts_to_embed = [f"{header}\n\n{text}" for header, text in chunk_data]
        else:
            print(f"Chunking {path}...")
            chunks = chunk_pdf(path)
            headers = [None] * len(chunks)
            texts_to_embed = chunks
    else:
        # For all other formats, use unified loader
        print(f"Loading {path}...")
        text = load_file(path)
        print(f"Extracted {len(text)} characters")

        if use_semantic:
            print(f"Semantic chunking...")
            chunk_data = chunk_semantically(text)
            headers = [header for header, _ in chunk_data]
            chunks = [text for _, text in chunk_data]
            texts_to_embed = [f"{header}\n\n{text}" for header, text in chunk_data]
        else:
            print(f"Chunking...")
            chunks = chunk_text(text)
            headers = [None] * len(chunks)
            texts_to_embed = chunks

    print(f"Created {len(chunks)} chunks")

    if len(chunks) == 0:
        print(f"Warning: No chunks created from {path}")
        return 0

    # 2. Embed all chunks using OpenAI batch API
    print(f"\nEmbedding {len(chunks)} chunks with OpenAI...")
    t0 = time.time()
    embeddings = embed_batch(texts_to_embed)
    print(f"Embedding took: {time.time() - t0:.2f}s")

    # 3. Insert into database
    print(f"\nInserting into database for user_id={user_id}...")
    conn = None
    cur = None

    try:
        conn = connect_db()
        cur = conn.cursor()

        # Prepare batch data for executemany()
        batch_data = []
        for i, (chunk, header, embedding) in enumerate(zip(chunks, headers, embeddings)):
            # Convert to pgvector format: [0.1,0.2,0.3,...]
            emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
            batch_data.append((user_id, chunk, header, emb_str, path, i, ext, content_hash, file_mtime))

        # Use executemany for efficient batch insert
        cur.executemany(
            """
            INSERT INTO chunks (user_id, chunk, header, embedding, source, chunk_index, file_type, content_hash, file_mtime)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            batch_data
        )

        conn.commit()
        print(f"All {len(chunks)} chunks embedded and stored!")
        return len(chunks)
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# ----------------------------
# Print processing summary with error details
# ----------------------------
def print_processing_summary(results: dict):
    """
    Print a summary of processing results with error details.

    Args:
        results: Dict with 'success' and 'failed' lists
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"  Success: {len(results['success'])} files")
    print(f"  Failed:  {len(results['failed'])} files")

    if results['failed']:
        print(f"\nFailed files:")
        for path, error in results['failed']:
            print(f"  - {os.path.basename(path)}: {error}")
    print(f"{'='*60}")


# ----------------------------
# Process directory: scan and process all supported files
# ----------------------------
def process_directory(dir_path, user_id, use_semantic=False):
    """
    Scan a directory and process all supported files.

    Args:
        dir_path: Path to directory
        user_id: User ID to associate chunks with
        use_semantic: If True, use AI-powered semantic chunking
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Track results
    results = {'success': [], 'failed': []}

    supported_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_supported(file_path):
                supported_files.append(file_path)

    print(f"Found {len(supported_files)} supported files in {dir_path}")
    print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS.keys())}")

    for i, file_path in enumerate(supported_files):
        print(f"\n[{i+1}/{len(supported_files)}] Processing: {file_path}")
        try:
            process_file(file_path, user_id, use_semantic)
            results['success'].append(file_path)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"ERROR: {error_msg}")
            results['failed'].append((file_path, error_msg))
            continue

    print_processing_summary(results)


# ----------------------------
# Process directory incrementally: only new/modified files
# ----------------------------
def process_directory_incremental(dir_path, user_id, use_semantic=False):
    """
    Scan a directory and process only new or modified files.

    - Skips unchanged files
    - Re-processes modified files (deletes old chunks first)
    - Deletes chunks for removed files

    Args:
        dir_path: Path to directory
        user_id: User ID to associate chunks with
        use_semantic: If True, use AI-powered semantic chunking
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    print(f"\n{'='*60}")
    print(f"INCREMENTAL UPDATE: {dir_path}")
    print(f"{'='*60}")

    # Get existing files from database
    existing_files = get_existing_files(user_id)
    print(f"Found {len(existing_files)} existing files in database")

    # Scan current files on disk
    extensions = set(SUPPORTED_EXTENSIONS.keys())
    current_files = scan_directory(dir_path, extensions)
    print(f"Found {len(current_files)} supported files on disk")

    # Categorize files
    categorized = categorize_files(current_files, existing_files)
    print_file_status_summary(categorized)

    # Get files to process and delete
    to_process, to_delete, to_delete_first = get_files_to_process(current_files, existing_files)

    # Phase 1: Delete chunks for removed files
    if to_delete:
        print(f"\n--- Deleting chunks for {len(to_delete)} removed files ---")
        for path in to_delete:
            deleted = delete_chunks_by_source(path, user_id)
            print(f"  Deleted {deleted} chunks: {path}")

    # Track results
    results = {'success': [], 'failed': [], 'modified': []}

    # Convert to_delete_first to set for quick lookup
    modified_files = set(to_delete_first)

    # Phase 2: Process new and modified files (delete old chunks AFTER successful processing)
    if to_process:
        print(f"\n--- Processing {len(to_process)} files ---")
        for i, file_path in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}] Processing: {file_path}")
            is_modified = file_path in modified_files

            try:
                # Compute hash for storage
                content_hash = compute_file_hash(file_path)

                # Process the file first (inserts new chunks)
                process_file(file_path, user_id, use_semantic, content_hash=content_hash)

                # Only delete old chunks AFTER successful processing
                if is_modified:
                    # Delete old chunks (those with different content_hash)
                    # This is safe because new chunks are already inserted
                    conn = None
                    cur = None
                    try:
                        conn = connect_db()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            DELETE FROM chunks
                            WHERE user_id = %s AND source = %s AND content_hash != %s
                            """,
                            (user_id, file_path, content_hash)
                        )
                        deleted_count = cur.rowcount
                        conn.commit()
                        print(f"  Updated existing file (removed {deleted_count} old chunks)")
                    finally:
                        if cur is not None:
                            cur.close()
                        if conn is not None:
                            conn.close()

                    results['modified'].append(file_path)
                else:
                    print(f"  Added new file")

                results['success'].append(file_path)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"ERROR: {error_msg}")
                results['failed'].append((file_path, error_msg))
                # If processing failed, old chunks are preserved (not deleted)
                if is_modified:
                    print(f"  Kept old chunks due to processing failure")
                continue
    else:
        print("\nNo files need processing - everything is up to date!")

    # Summary
    new_files = len(results['success']) - len(results['modified'])
    print(f"\n{'='*60}")
    print(f"INCREMENTAL UPDATE COMPLETE")
    print(f"  New:      {new_files} files")
    print(f"  Modified: {len(results['modified'])} files")
    print(f"  Failed:   {len(results['failed'])} files")
    print(f"  Deleted:  {len(to_delete)} files")
    print(f"  Unchanged: {len(categorized['unchanged'])} files")

    if results['failed']:
        print(f"\nFailed files:")
        for path, error in results['failed']:
            print(f"  - {os.path.basename(path)}: {error}")
    print(f"{'='*60}")


# Legacy alias for backwards compatibility
def process_pdf(path, user_id, use_semantic=False):
    """Legacy function - use process_file instead."""
    return process_file(path, user_id, use_semantic)


if __name__ == "__main__":
    # Get or create demo user for CLI usage
    demo_user_id = get_or_create_demo_user()
    print(f"Using demo user (id={demo_user_id})")

    # Check for --nocache flag
    if "--nocache" in sys.argv:
        cache_manager.set_cache_enabled(False)
        print("Cache disabled")
    else:
        print(f"Cache enabled at {cache_manager.CACHE_DIR}")

    # Check for --semantic flag
    use_semantic = "--semantic" in sys.argv

    if use_semantic:
        print("Using AI-powered semantic chunking...")
    else:
        print("Using traditional regex chunking (use --semantic for AI chunking)")

    # Check for --update flag (incremental mode)
    incremental_mode = "--update" in sys.argv

    if incremental_mode:
        print("Incremental update mode enabled (only process new/modified files)")

    # Get path from command line or environment
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else PDF_PATH

    if not path:
        print("Usage: python embed_chunks_pgvector.py <file_or_directory> [options]")
        print("\nOptions:")
        print("  --semantic   Use AI-powered semantic chunking")
        print("  --update     Incremental mode: only process new/modified files")
        print("  --nocache    Disable embedding cache")
        print(f"\nSupported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}")
        sys.exit(1)

    # Process file or directory
    if os.path.isdir(path):
        print(f"\nProcessing directory: {path}")
        if incremental_mode:
            process_directory_incremental(path, demo_user_id, use_semantic=use_semantic)
        else:
            process_directory(path, demo_user_id, use_semantic=use_semantic)
    elif os.path.isfile(path):
        if not is_supported(path):
            print(f"Error: Unsupported file format")
            print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}")
            sys.exit(1)
        process_file(path, demo_user_id, use_semantic=use_semantic)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
