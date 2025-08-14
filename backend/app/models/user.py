from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Column, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(320), nullable=False)
    role = Column(String(50), nullable=False, server_default=text("'member'"))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"))


