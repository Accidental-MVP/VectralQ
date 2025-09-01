from __future__ import annotations

from alembic import op


revision = "0010_enable_unaccent"
down_revision = "0009_sentence_jsonb_and_tsv"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")


def downgrade() -> None:
    # keep extension if in use; safe no-op
    pass


