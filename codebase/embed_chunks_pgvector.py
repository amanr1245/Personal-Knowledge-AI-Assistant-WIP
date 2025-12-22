import os
import sys
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from chunk_pdf import chunk_pdf
from semantic_chunker import chunk_semantically_from_pdf
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
# Batch embed using OpenAI API
# ----------------------------
def embed_batch(texts):
    """Embed a batch of texts using OpenAI API (max 2048 per call)"""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f"Embedding batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)...")

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)

    return all_embeddings


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

    # Check for --semantic flag
    use_semantic = "--semantic" in sys.argv

    if use_semantic:
        print("Using AI-powered semantic chunking...")
    else:
        print("Using traditional regex chunking (use --semantic for AI chunking)")

    process_pdf(PDF_PATH, demo_user_id, use_semantic=use_semantic)
