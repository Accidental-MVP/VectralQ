from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_ingest"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create doc_files
    op.create_table(
        "doc_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("'upload'")),
        sa.Column("ingest_status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_unique_constraint(
        "uq_doc_files_tenant_sha", "doc_files", ["tenant_id", "sha256"], schema="app"
    )

    # Create doc_chunks
    op.create_table(
        "doc_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("doc_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id_sha1", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("text_tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),  # placeholder; will switch to vector later
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        schema="app",
    )
    op.create_unique_constraint(
        "uq_doc_chunks_tenant_doc_chunkid", "doc_chunks", ["tenant_id", "doc_file_id", "chunk_id_sha1"], schema="app"
    )
    # FK (RLS will protect cross-tenant implicitly via policies)
    op.create_foreign_key(
        "fk_doc_chunks_doc_file", source_table="doc_chunks", referent_table="doc_files",
        local_cols=["doc_file_id"], remote_cols=["id"], source_schema="app", referent_schema="app", ondelete="CASCADE"
    )

    # Index for BM25
    op.execute("CREATE INDEX IF NOT EXISTS ix_doc_chunks_text_tsv ON app.doc_chunks USING GIN(text_tsv)")

    # Trigger to keep text_tsv in sync
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.doc_chunks_tsvector_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.text_tsv := to_tsvector('simple', COALESCE(NEW.text, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS doc_chunks_tsvector_update ON app.doc_chunks;
        CREATE TRIGGER doc_chunks_tsvector_update BEFORE INSERT OR UPDATE ON app.doc_chunks
        FOR EACH ROW EXECUTE FUNCTION app.doc_chunks_tsvector_trigger();
        """
    )

    # Apply RLS
    op.execute("SELECT app.apply_tenant_rls('app','doc_files','tenant_id')")
    op.execute("SELECT app.apply_tenant_rls('app','doc_chunks','tenant_id')")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS doc_chunks_tsvector_update ON app.doc_chunks")
    op.execute("DROP FUNCTION IF EXISTS app.doc_chunks_tsvector_trigger")
    op.drop_constraint("fk_doc_chunks_doc_file", "doc_chunks", type_="foreignkey", schema="app")
    op.drop_constraint("uq_doc_chunks_tenant_doc_chunkid", "doc_chunks", type_="unique", schema="app")
    op.drop_index("ix_doc_chunks_text_tsv", table_name="doc_chunks", schema="app")
    op.drop_table("doc_chunks", schema="app")
    op.drop_constraint("uq_doc_files_tenant_sha", "doc_files", type_="unique", schema="app")
    op.drop_table("doc_files", schema="app")


