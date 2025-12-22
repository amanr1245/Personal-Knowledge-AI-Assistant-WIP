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

# ----------------------------
# Create users table
# ----------------------------
print("Creating users table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        api_key TEXT UNIQUE NOT NULL
    );
""")
conn.commit()
print("Users table ready")

# ----------------------------
# Insert demo user if not exists
# ----------------------------
print("Creating demo user...")
cur.execute("""
    INSERT INTO users (name, api_key)
    VALUES ('Demo User', 'demo-api-key')
    ON CONFLICT (api_key) DO NOTHING;
""")
conn.commit()
print("Demo user ready")

# ----------------------------
# Drop and recreate chunks table with user_id
# ----------------------------
print("\nDropping existing chunks table...")
cur.execute("DROP TABLE IF EXISTS chunks;")

print("Creating chunks table with user_id and 1536 dimensions (OpenAI embeddings)...")
cur.execute("""
    CREATE TABLE chunks (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        chunk TEXT,
        header TEXT,
        embedding vector(1536),
        source TEXT,
        chunk_index INT
    );
""")
conn.commit()
print("Chunks table created")

# ----------------------------
# Create HNSW index for vector similarity search
# ----------------------------
print("\nCreating HNSW index on embedding column...")
print("This may take a few minutes depending on data size...")

cur.execute("""
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()
print("HNSW index created")

# ----------------------------
# Create index on user_id for faster filtering
# ----------------------------
print("Creating index on user_id...")
cur.execute("""
    CREATE INDEX IF NOT EXISTS chunks_user_id_idx
    ON chunks (user_id);
""")
conn.commit()
print("User ID index created")

print("\nAll tables and indexes ready!")

cur.close()
conn.close()
