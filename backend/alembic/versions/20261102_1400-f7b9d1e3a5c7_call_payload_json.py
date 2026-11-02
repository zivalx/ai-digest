"""call_payload_json on extracted_mentions

Stores the exact track-record call payload so failed / rejected call pushes can
be re-sent later (sweep at the start of each track-record cycle, or
``backfill_track_record.py --repush``) without re-running the LLM.

Revision ID: f7b9d1e3a5c7
Revises: e5a7c9d1f3b5
Create Date: 2026-11-02 14:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f7b9d1e3a5c7"
down_revision: Union[str, None] = "e5a7c9d1f3b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("extracted_mentions") as batch_op:
        batch_op.add_column(sa.Column("call_payload_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("extracted_mentions") as batch_op:
        batch_op.drop_column("call_payload_json")
