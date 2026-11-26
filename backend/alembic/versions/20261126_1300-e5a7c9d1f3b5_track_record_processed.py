"""track_record_processed ledger for the backfill

Revision ID: e5a7c9d1f3b5
Revises: d4f6b8c0e2a4
Create Date: 2026-11-02 13:00:00

Idempotent: a brand-new database already has the table from the baseline's create_all.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from src.models.base import GUID


# revision identifiers, used by Alembic.
revision: str = "e5a7c9d1f3b5"
down_revision: Union[str, None] = "d4f6b8c0e2a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if "track_record_processed" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "track_record_processed",
        sa.Column("id", GUID(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("item_id", sa.String(length=255), nullable=False),
        sa.Column("extraction_version", sa.String(length=32), nullable=False),
        sa.Column(
            "processed_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_type", "item_id", "extraction_version",
            name="uq_track_record_processed_item_version",
        ),
    )


def downgrade() -> None:
    if "track_record_processed" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("track_record_processed")
