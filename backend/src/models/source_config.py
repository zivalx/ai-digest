"""
SourceConfig model - stores source configurations (NON-SENSITIVE).
"""
from sqlalchemy import Column, String, Boolean, DateTime, Index, JSON
from sqlalchemy.sql import func
import uuid

from .base import Base, GUID


class SourceConfig(Base):
    """
    Source configuration - defines what and how to collect.

    SECURITY NOTE: Does NOT store credentials directly.
    Uses credential_ref to reference environment variable names.

    Example:
        credential_ref = "REDDIT_CLIENT_1"  # Points to env var
        collect_spec = {"subreddits": ["python"], "max_posts": 50}
    """
    __tablename__ = "source_configs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)  # User-friendly name
    source_type = Column(String(50), nullable=False)  # "reddit", "youtube", etc.

    # Credential reference (NOT the actual credential)
    credential_ref = Column(String(100), nullable=False)  # e.g., "REDDIT_CLIENT_1"

    # Collection specification (what to collect)
    collect_spec = Column(JSON, nullable=False)

    # Pipeline purpose: "general" (default digest) or "finance" (channel-intel feeder)
    purpose = Column(String(50), nullable=False, default="general")

    # Status
    enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_source_type_enabled", "source_type", "enabled"),
    )

    def __repr__(self):
        return f"<SourceConfig(id={self.id}, name={self.name}, type={self.source_type}, enabled={self.enabled})>"
