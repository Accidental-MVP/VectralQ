from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create helper schema/function for RLS application in future migrations
    op.execute("CREATE SCHEMA IF NOT EXISTS app")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.apply_tenant_rls(table_name text, tenant_column text)
        RETURNS void AS $$
        BEGIN
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
            EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
            EXECUTE format('DROP POLICY IF EXISTS %I_isolate ON %I', table_name, table_name);
            EXECUTE format(
                'CREATE POLICY %1$I_isolate ON %1$I USING ((%2$I)::text = current_setting(''app.tenant_id'', true)) WITH CHECK ((%2$I)::text = current_setting(''app.tenant_id'', true))',
                table_name, tenant_column
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Tenants
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )

    # Users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default=sa.text("'member'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"]) 
    op.create_index("ix_users_email", "users", ["email"], unique=False)

    # Docs
    op.create_table(
        "docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_docs_tenant_id", "docs", ["tenant_id"]) 

    # Apply RLS helper to tables
    op.execute("SELECT app.apply_tenant_rls('tenants', 'id')")
    op.execute("SELECT app.apply_tenant_rls('users', 'tenant_id')")
    op.execute("SELECT app.apply_tenant_rls('docs', 'tenant_id')")

    # Create least-privilege runtime role and grant privileges
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vectralq_app') THEN
                CREATE ROLE vectralq_app LOGIN PASSWORD 'vectralq_app' NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
            END IF;
        END$$;
    """)
    op.execute("GRANT USAGE ON SCHEMA public TO vectralq_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO vectralq_app")
    op.execute("GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO vectralq_app")
    op.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO vectralq_app")
    op.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO vectralq_app")


def downgrade() -> None:
    for policy in ("docs_isolate", "users_isolate", "tenants_isolate"):
        table = policy.split("_")[0]
        op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    for table in ("docs", "users", "tenants"):
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.drop_table("docs")
    op.drop_table("users")
    op.drop_table("tenants")


