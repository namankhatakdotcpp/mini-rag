CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    source TEXT,
    section TEXT,
    chunk_index INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_source_idx
ON documents (source);

CREATE INDEX IF NOT EXISTS documents_created_at_idx
ON documents (created_at);

CREATE INDEX IF NOT EXISTS documents_source_chunk_idx
ON documents (source, chunk_index);
