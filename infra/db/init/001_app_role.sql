-- Create least-privilege app role at cluster init
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vectralq_app') THEN
        CREATE ROLE vectralq_app LOGIN PASSWORD 'vectralq_app' NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END$$;

-- Grant future privileges on public schema objects
GRANT USAGE ON SCHEMA public TO vectralq_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO vectralq_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO vectralq_app;

-- Note: object-level GRANTs on existing tables are applied in migrations.

