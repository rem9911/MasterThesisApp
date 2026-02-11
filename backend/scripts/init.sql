-- backend/scripts/init.sql

-- 1. Enable Extensions (CRUCIAL)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 2. Create Main Table
CREATE TABLE IF NOT EXISTS legal_articles (
    id SERIAL PRIMARY KEY,
    
 
    code_source VARCHAR(100) NOT NULL,
    article_number VARCHAR(50) NOT NULL, 
    
    content TEXT NOT NULL,
    
    metadata JSONB NOT NULL DEFAULT '{}',
    
    embedding vector(1024),
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(code_source, article_number),

    UNIQUE(code_source, article_number),

    content_search tsvector
);

-- 3. HNSW Vector Index (For RAG performance)
CREATE INDEX IF NOT EXISTS legal_articles_embedding_idx 
ON legal_articles 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Full-Text Search Index (For keyword usage)
CREATE INDEX IF NOT EXISTS legal_articles_fts_idx 
ON legal_articles 
USING GIN (content_search);

-- 5. Trigger Function to update content_search
CREATE OR REPLACE FUNCTION legal_articles_tsvector_trigger() RETURNS trigger AS $$
BEGIN
  NEW.content_search := to_tsvector('french', unaccent(NEW.content));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- 6. Trigger application
DROP TRIGGER IF EXISTS tsvectorupdate ON legal_articles;
CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON legal_articles FOR EACH ROW EXECUTE PROCEDURE legal_articles_tsvector_trigger();