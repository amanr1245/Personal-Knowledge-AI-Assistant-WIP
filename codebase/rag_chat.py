import os
import psycopg2
from psycopg2 import pool
from openai import OpenAI
from dotenv import load_dotenv
import time

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# OpenAI client
# ----------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Database connection pool
# ----------------------------
connection_pool = None

def get_connection_pool():
    global connection_pool
    if connection_pool is None:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # min and max connections
            dbname="personal_rag",
            user="postgres",
            password="postgres",
            host="localhost",
            port=5432
        )
    return connection_pool

def connect_db():
    pool = get_connection_pool()
    return pool.getconn()


# ----------------------------
# Embed the user query
# ----------------------------
def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    embedding = response.data[0].embedding
    return embedding  # OpenAI embeddings are already normalized


# ----------------------------
# Retrieve initial pool of chunks from PostgreSQL
# ----------------------------
MAX_INITIAL = 15
MAX_CORPUS_PCT = 0.25

def get_top_k(raw_query_embedding):
    # Convert embedding to pgvector format: [0.12,0.53,...]
    emb_str = "[" + ",".join(str(x) for x in raw_query_embedding) + "]"

    conn = connect_db()
    cur = conn.cursor()

    # Get total chunk count for corpus-aware cap
    cur.execute("SELECT COUNT(*) FROM chunks;")
    total_chunks = cur.fetchone()[0]

    # Cap at 25% of corpus or MAX_INITIAL, whichever is smaller
    k = min(MAX_INITIAL, int(total_chunks * MAX_CORPUS_PCT))
    k = max(k, 1)  # Always retrieve at least 1

    # <=> operator = cosine distance (0 = identical, 2 = opposite)
    # cosine similarity = 1 - (distance / 2) to get range [0, 1]
    cur.execute(
        f"""
        SELECT chunk, source, chunk_index, 1 - (embedding <=> %s) / 2 AS similarity
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT {k};
        """,
        (emb_str, emb_str)
    )

    rows = cur.fetchall()
    cur.close()

    # Return connection to pool
    pool = get_connection_pool()
    pool.putconn(conn)

    return rows, total_chunks


# ----------------------------
# Filter chunks by similarity threshold and drop-off
# ----------------------------
MIN_SIMILARITY = 0.65
MAX_DROP = 0.10

def filter_chunks(chunks):
    """
    Filter chunks using:
    1. Minimum similarity threshold (0.65)
    2. Drop-off detection (stop when similarity drops >10% between consecutive chunks)

    Returns: (filtered_chunks, low_confidence_flag)
    """
    if not chunks:
        return [], True

    # Check if best chunk is below threshold
    best_similarity = chunks[0][3]  # similarity is 4th element
    if best_similarity < MIN_SIMILARITY:
        # Return only the best chunk with low confidence flag
        return [chunks[0]], True

    filtered = [chunks[0]]
    prev_similarity = best_similarity

    for chunk in chunks[1:]:
        similarity = chunk[3]

        # Stop if below minimum threshold
        if similarity < MIN_SIMILARITY:
            break

        # Stop if drop-off is too large
        drop = prev_similarity - similarity
        if drop > MAX_DROP:
            break

        filtered.append(chunk)
        prev_similarity = similarity

    return filtered, False


# ----------------------------
# Generate final answer using RAG
# ----------------------------
def ask(query):
    # 1. Embed query
    t0 = time.time()
    query_embedding = embed_query(query)
    print(f"Embedding took: {time.time() - t0:.2f}s")

    # 2. Retrieve initial pool of chunks
    t1 = time.time()
    initial_chunks, total_chunks = get_top_k(query_embedding)
    print(f"DB query took: {time.time() - t1:.2f}s")

    # 3. Filter chunks dynamically
    filtered_chunks, low_confidence = filter_chunks(initial_chunks)
    print(f"\n--- Chunk Selection ---")
    print(f"Total corpus: {total_chunks} chunks")
    print(f"Initial pool: {len(initial_chunks)} chunks")
    print(f"After filtering: {len(filtered_chunks)} chunks")
    if low_confidence:
        print("⚠ Low confidence match")
    print("------------------------")

    # 4. Format context text and display retrieved chunks
    context_text = ""
    print("\n--- Retrieved Chunks ---")
    for i, (chunk, source, chunk_index, sim) in enumerate(filtered_chunks):
        # chunk_index is 0-based in DB, display as 1-based
        display_chunk_num = chunk_index + 1
        print(f"[Chunk {display_chunk_num}] (similarity: {sim:.3f})")
        context_text += f"[Chunk {display_chunk_num}] (source: {source})\n{chunk}\n\n"
    print("------------------------\n")

    # 5. Create RAG prompt (with low-confidence preamble if needed)
    if low_confidence:
        preamble = """IMPORTANT: The retrieved context has low relevance to the user's question.
Start your response with: "I couldn't find exactly what you were looking for, but here's something that might be relevant:"
Then provide whatever information you can from the context.
"""
    else:
        preamble = ""

    prompt = f"""
You are a helpful assistant using Retrieval-Augmented Generation.

{preamble}Use ONLY the context below to answer the user's question.
If a fact is not in the context, say you don't have enough information.

Context:
{context_text}

Question: {query}

When you reference information from the context, cite it using the chunk number shown (e.g., [Chunk 5], [Chunk 12]).
"""

    # 6. Generate answer using OpenAI
    t2 = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"LLM generation took: {time.time() - t2:.2f}s")

    return response.choices[0].message.content


#user input loop
if __name__ == "__main__":
    print("RAG Chatbot Ready! Ask something about your documents.")
    print("Type 'exit' or 'quit' to exit.\n")
    while True:
        q = input("Ask: ")
        if q.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        print("\n" + ask(q))