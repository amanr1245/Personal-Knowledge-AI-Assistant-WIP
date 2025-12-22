import os
import sys
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from chunk_pdf import chunk_pdf
from semantic_chunker import chunk_semantically_from_pdf
import cache_manager
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

            response = openai_client.embeddings.create(
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
# Process PDF: chunk, embed, insert
# ----------------------------
def process_pdf(path, user_id, use_semantic=False):
    """
    Process a PDF: chunk, embed, and store in database.

    Args:
        path: Path to PDF file
        user_id: User ID to associate chunks with
        use_semantic: If True, use AI-powered semantic chunking with headers
    """
    if use_semantic:
        # Use semantic chunking with headers
        print(f"Semantic chunking {path}...")
        chunk_data = chunk_semantically_from_pdf(path)
        headers = [header for header, _ in chunk_data]
        chunks = [text for _, text in chunk_data]
        # Embed header + chunk together for better retrieval
        texts_to_embed = [f"{header}\n\n{text}" for header, text in chunk_data]
    else:
        # Use traditional regex chunking
        print(f"Chunking {path}...")
        chunks = chunk_pdf(path)
        headers = [None] * len(chunks)
        texts_to_embed = chunks

    print(f"Created {len(chunks)} chunks")

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
            INSERT INTO chunks (user_id, chunk, header, embedding, source, chunk_index)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (user_id, chunk, header, emb_str, path, i)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"All {len(chunks)} chunks embedded and stored!")


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

    process_pdf(PDF_PATH, demo_user_id, use_semantic=use_semantic)
