from __future__ import annotations

from alembic import op


revision = "0008_add_title_heading_enw"
down_revision = "0007_bm25_english_enw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add title and heading columns if not present; ensure text_tsv_enw exists
    op.execute(
        """
        ALTER TABLE app.doc_chunks
          ADD COLUMN IF NOT EXISTS title   TEXT,
          ADD COLUMN IF NOT EXISTS heading TEXT,
          ADD COLUMN IF NOT EXISTS text_tsv_enw tsvector;
        """
    )

    # Create or replace weighted trigger function
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.doc_chunks_tsv_enw_trigger()
        RETURNS trigger AS $$
        BEGIN
          NEW.text_tsv_enw :=
            setweight(to_tsvector('english', coalesce(NEW.title,'')),   'A') ||
            setweight(to_tsvector('english', coalesce(NEW.heading,'')), 'B') ||
            setweight(to_tsvector('english', coalesce(NEW.text,'')),    'D');
          RETURN NEW;
        END $$ LANGUAGE plpgsql;
        """
    )

    # Recreate trigger
    op.execute("DROP TRIGGER IF EXISTS trg_doc_chunks_tsv_enw ON app.doc_chunks;")
    op.execute(
        """
        CREATE TRIGGER trg_doc_chunks_tsv_enw
        BEFORE INSERT OR UPDATE OF title, heading, text
        ON app.doc_chunks
        FOR EACH ROW EXECUTE FUNCTION app.doc_chunks_tsv_enw_trigger();
        """
    )

    # Index
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_tsv_enw
          ON app.doc_chunks USING gin (text_tsv_enw);
        """
    )

    # Backfill tsvector using available columns
    op.execute(
        """
        UPDATE app.doc_chunks SET text_tsv_enw =
            setweight(to_tsvector('english', coalesce(title,'')),   'A') ||
            setweight(to_tsvector('english', coalesce(heading,'')), 'B') ||
            setweight(to_tsvector('english', coalesce(text,'')),    'D');
        """
    )


def downgrade() -> None:
    # Keep columns; just drop index and trigger to be safe on downgrade
    op.execute("DROP INDEX IF EXISTS idx_doc_chunks_tsv_enw;")
    op.execute("DROP TRIGGER IF EXISTS trg_doc_chunks_tsv_enw ON app.doc_chunks;")
    # function may be used by future migrations; do not drop


