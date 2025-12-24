import os
import sys
import re
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from chunk_pdf import chunk_pdf
from semantic_chunker import chunk_semantically_from_pdf, chunk_semantically
from loaders import load_file, SUPPORTED_EXTENSIONS, is_supported
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
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = os.getenv("PDF_PATH")
BATCH_SIZE = 2048  # OpenAI allows up to 2048 embeddings per request


# ----------------------------
# Connect to PostgreSQL
# ----------------------------
def connect_db():
    return psycopg2.connect(
        dbname="personal_rag",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )


# ----------------------------
# Get or create demo user
# ----------------------------
def get_or_create_demo_user():
    """Get the demo user's ID, creating the user if needed."""
    conn = connect_db()
    cur = conn.cursor()

    # Try to get existing demo user
    cur.execute("SELECT id FROM users WHERE api_key = 'demo-api-key';")
    result = cur.fetchone()

    if result:
        user_id = result[0]
    else:
        # Create demo user if not exists
        cur.execute("""
            INSERT INTO users (name, api_key)
            VALUES ('Demo User', 'demo-api-key')
            RETURNING id;
        """)
        user_id = cur.fetchone()[0]
        conn.commit()

    cur.close()
    conn.close()
    return user_id


# ----------------------------
# Batch embed using OpenAI API (with caching)
# ----------------------------
def embed_batch(texts):
    """Embed a batch of texts using OpenAI API with caching (max 2048 per call)"""
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
            new_embeddings.extend(embeddings)

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
def process_file(path, user_id, use_semantic=False):
    """
    Process any supported file: load, chunk, embed, and store in database.

    Args:
        path: Path to file (PDF, image, text, or document)
        user_id: User ID to associate chunks with
        use_semantic: If True, use AI-powered semantic chunking with headers
    """
    ext = os.path.splitext(path)[1].lower()

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
        return

    # 2. Embed all chunks using OpenAI batch API
    print(f"\nEmbedding {len(chunks)} chunks with OpenAI...")
    t0 = time.time()
    embeddings = embed_batch(texts_to_embed)
    print(f"Embedding took: {time.time() - t0:.2f}s")

    # 3. Insert into database
    print(f"\nInserting into database for user_id={user_id}...")
    conn = connect_db()
    cur = conn.cursor()

    for i, (chunk, header, embedding) in enumerate(zip(chunks, headers, embeddings)):
        # Convert to pgvector format: [0.1,0.2,0.3,...]
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cur.execute(
            """
            INSERT INTO chunks (user_id, chunk, header, embedding, source, chunk_index, file_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (user_id, chunk, header, emb_str, path, i, ext)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"All {len(chunks)} chunks embedded and stored!")


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
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"\nProcessed {len(supported_files)} files")


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

    # Get path from command line or environment
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else PDF_PATH

    if not path:
        print("Usage: python embed_chunks_pgvector.py <file_or_directory> [--semantic] [--nocache]")
        print(f"\nSupported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}")
        sys.exit(1)

    # Process file or directory
    if os.path.isdir(path):
        print(f"\nProcessing directory: {path}")
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
