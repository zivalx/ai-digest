"""track record fields on extracted_mentions and source_configs

Influencer track record: recommendation fields (action, horizon, prices, quote,
said_at) and call-push tracking on extracted_mentions, plus a `track_record`
flag on source_configs.

Revision ID: d4f6b8c0e2a4
Revises: c8e07325e3a6
Create Date: 2026-11-02 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4f6b8c0e2a4"
down_revision: Union[str, None] = "c8e07325e3a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("source_configs") as batch_op:
        batch_op.add_column(
            sa.Column("track_record", sa.Boolean(), nullable=False, server_default=sa.false())
        )

    with op.batch_alter_table("extracted_mentions") as batch_op:
        batch_op.add_column(
            sa.Column("is_recommendation", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(sa.Column("action", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("horizon", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("entry_price", sa.Numeric(precision=18, scale=6), nullable=True))
        batch_op.add_column(sa.Column("target_price", sa.Numeric(precision=18, scale=6), nullable=True))
        batch_op.add_column(sa.Column("stop_price", sa.Numeric(precision=18, scale=6), nullable=True))
        batch_op.add_column(sa.Column("trigger_price", sa.Numeric(precision=18, scale=6), nullable=True))
        batch_op.add_column(sa.Column("quote", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("said_at", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("said_at_precision", sa.String(length=8), nullable=True))
        batch_op.add_column(
            sa.Column("pushed_calls_to_trender", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(sa.Column("trender_calls_response", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("extracted_mentions") as batch_op:
        for col in (
            "trender_calls_response",
            "pushed_calls_to_trender",
            "said_at_precision",
            "said_at",
            "quote",
            "trigger_price",
            "stop_price",
            "target_price",
            "entry_price",
            "horizon",
            "action",
            "is_recommendation",
        ):
            batch_op.drop_column(col)

    with op.batch_alter_table("source_configs") as batch_op:
        batch_op.drop_column("track_record")
