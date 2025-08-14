from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Helper function for RLS application in future migrations (schema-qualified)
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.apply_tenant_rls(schema_name text, table_name text, tenant_column text)
        RETURNS void AS $$
        BEGIN
            EXECUTE format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', schema_name, table_name);
            EXECUTE format('ALTER TABLE %I.%I FORCE ROW LEVEL SECURITY', schema_name, table_name);
            EXECUTE format('DROP POLICY IF EXISTS %I_isolate ON %I.%I', table_name, schema_name, table_name);
            EXECUTE format(
                'CREATE POLICY %2$I_isolate ON %1$I.%2$I USING ((%3$I)::text = current_setting(''app.tenant_id'', true)) WITH CHECK ((%3$I)::text = current_setting(''app.tenant_id'', true))',
                schema_name, table_name, tenant_column
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Tenants (in schema app)
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )

    # Users (in schema app)
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("app.tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default=sa.text("'member'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"], schema="app") 
    op.create_index("ix_users_email", "users", ["email"], unique=False, schema="app")

    # Docs (in schema app)
    op.create_table(
        "docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("app.tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_index("ix_docs_tenant_id", "docs", ["tenant_id"], schema="app") 

    # Apply RLS helper to tables
    op.execute("SELECT app.apply_tenant_rls('app', 'tenants', 'id')")
    op.execute("SELECT app.apply_tenant_rls('app', 'users', 'tenant_id')")
    op.execute("SELECT app.apply_tenant_rls('app', 'docs', 'tenant_id')")


def downgrade() -> None:
    for policy, table in (("docs_isolate", "docs"), ("users_isolate", "users"), ("tenants_isolate", "tenants")):
        op.execute(f"DROP POLICY IF EXISTS {policy} ON app.{table}")
    for table in ("docs", "users", "tenants"):
        op.execute(f"ALTER TABLE app.{table} DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_docs_tenant_id", table_name="docs", schema="app")
    op.drop_table("docs", schema="app")
    op.drop_index("ix_users_email", table_name="users", schema="app")
    op.drop_index("ix_users_tenant_id", table_name="users", schema="app")
    op.drop_table("users", schema="app")
    op.drop_table("tenants", schema="app")


