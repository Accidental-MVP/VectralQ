from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_bm25_english_enw"
down_revision = "0006_soft_delete_and_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Add weighted english tsvector column (if missing)
    op.add_column("doc_chunks", sa.Column("text_tsv_enw", sa.TEXT(), nullable=True), schema="app")
    # 2) Backfill values
    op.execute(
        """
        UPDATE app.doc_chunks
        SET text_tsv_enw = (
          setweight(to_tsvector('english', coalesce(title,'')),   'A') ||
          setweight(to_tsvector('english', coalesce(heading,'')), 'B') ||
          setweight(to_tsvector('english', coalesce(text,'')),    'D')
        )
        """
    )
    # 3) Index
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv_enw
          ON app.doc_chunks USING gin (text_tsv_enw)
        """
    )
    # 4) Trigger to keep it fresh
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.doc_chunks_tsv_enw_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.text_tsv_enw :=
            setweight(to_tsvector('english', coalesce(NEW.title,'')),   'A') ||
            setweight(to_tsvector('english', coalesce(NEW.heading,'')), 'B') ||
            setweight(to_tsvector('english', coalesce(NEW.text,'')),    'D');
          RETURN NEW;
        END$$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_doc_chunks_tsv_enw ON app.doc_chunks")
    op.execute(
        """
        CREATE TRIGGER trg_doc_chunks_tsv_enw
        BEFORE INSERT OR UPDATE OF title, heading, text
        ON app.doc_chunks
        FOR EACH ROW EXECUTE FUNCTION app.doc_chunks_tsv_enw_trigger();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_doc_chunks_tsv_enw ON app.doc_chunks")
    op.execute("DROP FUNCTION IF EXISTS app.doc_chunks_tsv_enw_trigger()")
    op.execute("DROP INDEX IF EXISTS idx_doc_chunks_tsv_enw")
    op.drop_column("doc_chunks", "text_tsv_enw", schema="app")


