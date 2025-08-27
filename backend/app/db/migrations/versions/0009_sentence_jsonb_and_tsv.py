from __future__ import annotations

from alembic import op


revision = "0009_sentence_jsonb_and_tsv"
down_revision = "0008_add_title_heading_enw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add sentences JSONB and sentence_tsv to doc_chunks
    op.execute(
        """
        ALTER TABLE app.doc_chunks
          ADD COLUMN IF NOT EXISTS sentences JSONB,
          ADD COLUMN IF NOT EXISTS sentence_tsv tsvector;
        """
    )

    # Trigger to rebuild sentence_tsv from sentences[].text
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app.update_sentence_tsv() RETURNS trigger AS $$
        BEGIN
          NEW.sentence_tsv := to_tsvector('english',
            COALESCE(
              array_to_string(ARRAY(
                SELECT (j->>'text') FROM jsonb_array_elements(COALESCE(NEW.sentences, '[]'::jsonb)) AS j
              ), ' '),
            '')
          );
          RETURN NEW;
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute("DROP TRIGGER IF EXISTS trg_sentence_tsv ON app.doc_chunks;")
    op.execute(
        """
        CREATE TRIGGER trg_sentence_tsv
        BEFORE INSERT OR UPDATE OF sentences
        ON app.doc_chunks
        FOR EACH ROW EXECUTE FUNCTION app.update_sentence_tsv();
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_sentence_tsv
          ON app.doc_chunks USING gin (sentence_tsv);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_doc_chunks_sentence_tsv;")
    op.execute("DROP TRIGGER IF EXISTS trg_sentence_tsv ON app.doc_chunks;")

