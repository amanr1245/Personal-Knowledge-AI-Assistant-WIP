import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from chunk_pdf import chunk_pdf
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
def process_pdf(path):
    # 1. Chunk the PDF
    print(f"Chunking {path}...")
    chunks = chunk_pdf(path)
    print(f"Created {len(chunks)} chunks")

    # 2. Embed all chunks using OpenAI batch API
    print(f"\nEmbedding {len(chunks)} chunks with OpenAI...")
    t0 = time.time()
    embeddings = embed_batch(chunks)
    print(f"Embedding took: {time.time() - t0:.2f}s")

    # 3. Insert into database
    print("\nInserting into database...")
    conn = connect_db()
    cur = conn.cursor()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Convert to pgvector format: [0.1,0.2,0.3,...]
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cur.execute(
            """
            INSERT INTO chunks (chunk, embedding, source, chunk_index)
            VALUES (%s, %s, %s, %s);
            """,
            (chunk, emb_str, path, i)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"All {len(chunks)} chunks embedded and stored!")


if __name__ == "__main__":
    process_pdf(PDF_PATH)
