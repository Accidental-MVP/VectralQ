from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0005_google_drive"
down_revision = "0004_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # OAuth credentials table
    op.create_table(
        "oauth_credentials",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=False),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("expiry", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_unique_constraint(
        "uq_oauth_credentials_tenant_provider",
        "oauth_credentials",
        ["tenant_id", "provider"],
        schema="app",
    )

    # Connectors table
    op.create_table(
        "connectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_unique_constraint(
        "uq_connectors_tenant_provider",
        "connectors",
        ["tenant_id", "provider"],
        schema="app",
    )

    # Extend doc_files with external source mapping if not exists
    # Use IF NOT EXISTS variants via raw SQL for portability in Alembic
    op.execute(
        """
        ALTER TABLE app.doc_files
        ADD COLUMN IF NOT EXISTS source TEXT,
        ADD COLUMN IF NOT EXISTS external_id TEXT,
        ADD COLUMN IF NOT EXISTS mime_type TEXT;
        """
    )
    # Add checksum for whole file if missing
    try:
        op.execute("ALTER TABLE app.doc_files ADD COLUMN IF NOT EXISTS checksum_sha256 TEXT")
    except Exception:
        pass
    # Helpful index
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_files_tenant_source_ext ON app.doc_files(tenant_id, source, external_id)"
    )

    # Apply RLS to new tables
    op.execute("SELECT app.apply_tenant_rls('app','oauth_credentials','tenant_id')")
    op.execute("SELECT app.apply_tenant_rls('app','connectors','tenant_id')")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_doc_files_tenant_source_ext")
    # Columns left as-is to avoid data loss
    op.drop_constraint("uq_connectors_tenant_provider", "connectors", schema="app")
    op.drop_table("connectors", schema="app")
    op.drop_constraint("uq_oauth_credentials_tenant_provider", "oauth_credentials", schema="app")
    op.drop_table("oauth_credentials", schema="app")


