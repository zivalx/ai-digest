"""
Cycle model - represents a collection + summarization cycle.
"""
from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
import enum

from .base import Base, GUID


class CycleStatus(str, enum.Enum):
    """Cycle lifecycle states."""
    PENDING = "pending"
    COLLECTING = "collecting"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class Cycle(Base):
    """
    Collection cycle - tracks a complete run of data collection + summarization.

    Lifecycle:
        pending → collecting → summarizing → completed (or failed)
    """
    __tablename__ = "cycles"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    status = Column(SQLEnum(CycleStatus), nullable=False, default=CycleStatus.PENDING)
    config_snapshot = Column(JSON, nullable=False)  # Full config used for this cycle

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Relationships
    collected_data = relationship("CollectedData", back_populates="cycle", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="cycle", cascade="all, delete-orphan")
    extracted_mentions = relationship("ExtractedMention", back_populates="cycle", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Cycle(id={self.id}, name={self.name}, status={self.status})>"
