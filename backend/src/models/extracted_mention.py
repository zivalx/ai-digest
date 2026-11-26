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

    # Track-record (recommendation) fields
    is_recommendation = Column(Boolean, nullable=False, default=False)
    action = Column(String(16), nullable=True)
    horizon = Column(String(16), nullable=True)
    entry_price = Column(Numeric(18, 6), nullable=True)
    target_price = Column(Numeric(18, 6), nullable=True)
    stop_price = Column(Numeric(18, 6), nullable=True)
    trigger_price = Column(Numeric(18, 6), nullable=True)
    quote = Column(Text, nullable=True)
    said_at = Column(String(40), nullable=True)
    said_at_precision = Column(String(8), nullable=True)

    # When the source content was published (the payload's mentioned_at). Kept so a
    # push that failed can be retried later with the right date.
    mentioned_at = Column(DateTime(timezone=True), nullable=True)

    # Source pointer (full citation payload sent to trender)
    citation_json = Column(JSON, nullable=False)

    # Idempotency
    dedupe_key = Column(String(512), nullable=False, unique=True)

    # Push tracking
    pushed_to_trender = Column(Boolean, nullable=False, default=False)
    trender_response = Column(Text, nullable=True)

    # Track-record push tracking (trender POST /api/track-record/calls).
    # call_payload_json is the exact CallIn payload, kept so unpushed / rejected
    # calls can be re-sent without re-running extraction.
    call_payload_json = Column(JSON(none_as_null=True), nullable=True)
    pushed_calls_to_trender = Column(Boolean, nullable=False, default=False)
    trender_calls_response = Column(Text, nullable=True)

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
