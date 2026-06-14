-- Auto-runs on first container startup (only once per volume).
-- Install pgvector extension required for vector(N) columns.
CREATE EXTENSION IF NOT EXISTS vector;