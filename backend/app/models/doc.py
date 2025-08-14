from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Column, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class Doc(Base):
    __tablename__ = "docs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)  # e.g., 'upload', 'gdrive', 'notion'
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))


