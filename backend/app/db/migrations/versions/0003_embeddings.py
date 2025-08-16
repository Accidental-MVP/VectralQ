from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_embeddings"
down_revision = "0002_ingest"
branch_labels = None
depends_on = None


EMBED_DIM = 384


def upgrade() -> None:
    # Add checksum column if missing
    op.add_column("doc_chunks", sa.Column("text_checksum", sa.Text(), nullable=True), schema="app")

    # Ensure vector extension exists (done in init; safe to call)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Convert embedding column to vector(EMBED_DIM)
    # If the column doesn't exist yet (older state), add it; else alter type
    # Try ALTER first, fallback to ADD
    try:
        op.execute(f"ALTER TABLE app.doc_chunks ALTER COLUMN embedding TYPE vector({EMBED_DIM}) USING embedding::vector")
    except Exception:
        op.add_column("doc_chunks", sa.Column("embedding", sa.TEXT(), nullable=True), schema="app")
        op.execute(f"ALTER TABLE app.doc_chunks ALTER COLUMN embedding TYPE vector({EMBED_DIM}) USING embedding::vector")

    # Create IVFFLAT index for cosine ops
    op.execute(
        f"CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_ivfflat ON app.doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # Analyze for planner stats
    op.execute("ANALYZE app.doc_chunks")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_doc_chunks_embedding_ivfflat")
    # Revert to TEXT to avoid extension requirement
    op.execute("ALTER TABLE app.doc_chunks ALTER COLUMN embedding TYPE TEXT")
    op.drop_column("doc_chunks", "text_checksum", schema="app")


