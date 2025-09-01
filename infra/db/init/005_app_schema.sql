-- Create application schema and required extensions at cluster init
CREATE SCHEMA IF NOT EXISTS app;

-- Ensure least-privilege runtime role can use and create in app schema
GRANT USAGE, CREATE ON SCHEMA app TO vectralq_app;

-- Set default search_path for the runtime role so unqualified names resolve to app first
ALTER ROLE vectralq_app SET search_path = app, public;

-- Required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS vector;

-- Default privileges so future tables/sequences in schema app are accessible to runtime user
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO vectralq_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO vectralq_app;


