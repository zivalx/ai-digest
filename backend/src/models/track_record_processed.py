"""
TrackRecordProcessed model - ledger for the track-record backfill.

One row per (source_type, item_id, extraction_version) that the backfill has
actually extracted, so re-runs skip finished items but a bumped
``EXTRACTION_VERSION`` (new prompt) reprocesses them.
"""
from sqlalchemy import Column, String, DateTime, UniqueConstraint
from sqlalchemy.sql import func
import uuid

from .base import Base, GUID


class TrackRecordProcessed(Base):
    __tablename__ = "track_record_processed"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "item_id", "extraction_version",
            name="uq_track_record_processed_item_version",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    source_type = Column(String(50), nullable=False)
    item_id = Column(String(255), nullable=False)
    extraction_version = Column(String(32), nullable=False)
    processed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
