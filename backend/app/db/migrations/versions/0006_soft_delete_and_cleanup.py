from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_soft_delete_and_cleanup"
down_revision = "0005_google_drive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Soft delete column on doc_files
    op.add_column("doc_files", sa.Column("deleted_at", sa.TIMESTAMP(timezone=True), nullable=True), schema="app")
    op.execute("CREATE INDEX IF NOT EXISTS ix_doc_files_tenant_deleted_at ON app.doc_files(tenant_id, deleted_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_doc_files_tenant_deleted_at")
    op.drop_column("doc_files", "deleted_at", schema="app")


