import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="personal_rag",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)

cur = conn.cursor()

# Drop existing table and recreate with new dimensions
print("Dropping existing chunks table...")
cur.execute("DROP TABLE IF EXISTS chunks;")

print("Creating chunks table with 1536 dimensions (OpenAI embeddings)...")
cur.execute("""
    CREATE TABLE chunks (
        id SERIAL PRIMARY KEY,
        chunk TEXT,
        embedding vector(1536),
        source TEXT,
        chunk_index INT
    );
""")

conn.commit()
print("Table created")

print("\nCreating HNSW index on embedding column...")
print("This may take a few minutes depending on data size...")

# Create HNSW index for fast vector similarity search
cur.execute("""
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")

conn.commit()
print("Index created successfully!")

cur.close()
conn.close()
