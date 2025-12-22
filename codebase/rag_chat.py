import os
import psycopg2
from psycopg2 import pool
from openai import OpenAI
from dotenv import load_dotenv
import time
from reranker import rerank_chunks, RERANK_CANDIDATES, RERANK_TOP_K

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
# Get demo user ID
# ----------------------------
def get_demo_user_id():
    """Get the demo user's ID from the database."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE api_key = 'demo-api-key';")
    result = cur.fetchone()
    cur.close()
    pool = get_connection_pool()
    pool.putconn(conn)

    if result:
        return result[0]
    raise Exception("Demo user not found. Run create_index.py first.")


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
MAX_INITIAL = RERANK_CANDIDATES  # Fetch more candidates for reranking (default 20)
MAX_CORPUS_PCT = 0.50  # Increased to allow more candidates for reranking

def get_top_k(raw_query_embedding, user_id):
    # Convert embedding to pgvector format: [0.12,0.53,...]
    emb_str = "[" + ",".join(str(x) for x in raw_query_embedding) + "]"

    conn = connect_db()
    cur = conn.cursor()

    # Get total chunk count for this user's corpus
    cur.execute("SELECT COUNT(*) FROM chunks WHERE user_id = %s;", (user_id,))
    total_chunks = cur.fetchone()[0]

    # Cap at 25% of corpus or MAX_INITIAL, whichever is smaller
    k = min(MAX_INITIAL, int(total_chunks * MAX_CORPUS_PCT))
    k = max(k, 1)  # Always retrieve at least 1

    # <=> operator = cosine distance (0 = identical, 2 = opposite)
    # cosine similarity = 1 - (distance / 2) to get range [0, 1]
    # Filter by user_id to only search this user's chunks
    cur.execute(
        f"""
        SELECT chunk, header, source, chunk_index, 1 - (embedding <=> %s) / 2 AS similarity
        FROM chunks
        WHERE user_id = %s
        ORDER BY embedding <=> %s
        LIMIT {k};
        """,
        (emb_str, user_id, emb_str)
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

    Chunk tuple format: (chunk, header, source, chunk_index, similarity)
    """
    if not chunks:
        return [], True

    # Check if best chunk is below threshold
    best_similarity = chunks[0][4]  # similarity is 5th element (index 4)
    if best_similarity < MIN_SIMILARITY:
        # Return only the best chunk with low confidence flag
        return [chunks[0]], True

    filtered = [chunks[0]]
    prev_similarity = best_similarity

    for chunk in chunks[1:]:
        similarity = chunk[4]

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
def ask(query, user_id):
    # 1. Embed query
    t0 = time.time()
    query_embedding = embed_query(query)
    print(f"Embedding took: {time.time() - t0:.2f}s")

    # 2. Retrieve initial pool of chunks for this user (top RERANK_CANDIDATES)
    t1 = time.time()
    initial_chunks, total_chunks = get_top_k(query_embedding, user_id)
    print(f"DB query took: {time.time() - t1:.2f}s")

    # 3. Filter chunks by similarity threshold
    filtered_chunks, low_confidence = filter_chunks(initial_chunks)

    # 4. Rerank filtered chunks using LLM (two-stage retrieval)
    t2 = time.time()
    reranked_chunks, rejected = rerank_chunks(query, filtered_chunks, top_k=RERANK_TOP_K)
    print(f"Reranking took: {time.time() - t2:.2f}s")

    # Use reranked chunks, or fall back to filtered if rejected
    if rejected or not reranked_chunks:
        final_chunks = filtered_chunks[:RERANK_TOP_K]
        rerank_status = "rejected (using vector similarity order)"
    else:
        final_chunks = reranked_chunks
        rerank_status = "applied"

    print(f"\n--- Chunk Selection ---")
    print(f"Total corpus: {total_chunks} chunks")
    print(f"Vector search: {len(initial_chunks)} candidates")
    print(f"After similarity filter: {len(filtered_chunks)} chunks")
    print(f"After reranking: {len(final_chunks)} chunks ({rerank_status})")
    if low_confidence:
        print("Low confidence match")
    print("------------------------")

    # 5. Format context text and display retrieved chunks
    # Chunk tuple format: (chunk, header, source, chunk_index, similarity)
    context_text = ""
    print("\n--- Retrieved Chunks (after reranking) ---")
    for i, (chunk, header, source, chunk_index, sim) in enumerate(final_chunks):
        # chunk_index is 0-based in DB, display as 1-based
        display_chunk_num = chunk_index + 1
        if header:
            print(f"[Chunk {display_chunk_num}] {header} (similarity: {sim:.3f})")
            context_text += f"[Chunk {display_chunk_num}] {header} (source: {source})\n{chunk}\n\n"
        else:
            print(f"[Chunk {display_chunk_num}] (similarity: {sim:.3f})")
            context_text += f"[Chunk {display_chunk_num}] (source: {source})\n{chunk}\n\n"
    print("------------------------\n")

    # 6. Create RAG prompt (with low-confidence preamble if needed)
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

    # 7. Generate answer using OpenAI
    t3 = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"LLM generation took: {time.time() - t3:.2f}s")

    return response.choices[0].message.content


#user input loop
if __name__ == "__main__":
    # Get demo user for CLI usage
    DEMO_USER_ID = get_demo_user_id()

    print("RAG Chatbot Ready! Ask something about your documents.")
    print(f"Using demo user (id={DEMO_USER_ID})")
    print("Type 'q' to exit.\n")
    while True:
        q = input("Ask: ")
        if q.lower() in ("q"):
            print("Goodbye!")
            break
        print("\n" + ask(q, DEMO_USER_ID))