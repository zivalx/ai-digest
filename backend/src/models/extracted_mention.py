"""
ExtractedMention model - a ticker/asset pick extracted from collected content.

Each mention carries a pointer back to its exact source (citation_json) and
tracks whether it has been pushed to trender's discovery ingest endpoint.
"""
from sqlalchemy import Column, String, Text, Numeric, Boolean, DateTime, ForeignKey, Index, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from .base import Base, GUID


class ExtractedMention(Base):
    """
    A single ticker/asset pick extracted by the LLM from a collected item.

    Idempotency: `dedupe_key` = (symbol | citation.url | start_seconds) is unique,
    so re-runs/retries never double-insert the same pick. `pushed_to_trender`
    gates re-sending to the trender ingest endpoint.
    """
    __tablename__ = "extracted_mentions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    cycle_id = Column(GUID(), ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False)

    # Pick content
    symbol = Column(String(32), nullable=False)
    company_name = Column(String(255), nullable=True)
    thesis = Column(Text, nullable=True)
    stance = Column(String(16), nullable=True)  # bullish | bearish | neutral
    sentiment_score = Column(Numeric(4, 3), nullable=True)  # -1.000 .. 1.000
    confidence = Column(Numeric(4, 3), nullable=True)  # 0.000 .. 1.000

    # Source pointer (full citation payload sent to trender)
    citation_json = Column(JSON, nullable=False)

    # Idempotency
    dedupe_key = Column(String(512), nullable=False, unique=True)

    # Push tracking
    pushed_to_trender = Column(Boolean, nullable=False, default=False)
    trender_response = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    cycle = relationship("Cycle", back_populates="extracted_mentions")

    __table_args__ = (
        Index("idx_mention_cycle_id", "cycle_id"),
        Index("idx_mention_symbol", "symbol"),
    )

    def __repr__(self):
        return (
            f"<ExtractedMention(id={self.id}, symbol={self.symbol}, "
            f"stance={self.stance}, pushed={self.pushed_to_trender})>"
        )
