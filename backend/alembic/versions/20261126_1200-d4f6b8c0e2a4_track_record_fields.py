"""track record fields on extracted_mentions and source_configs

Influencer track record: recommendation fields (action, horizon, prices, quote,
said_at) and call-push tracking on extracted_mentions, plus a `track_record`
flag on source_configs.

Revision ID: d4f6b8c0e2a4
Revises: d41b7a90c2f1
Create Date: 2026-11-02 12:00:00

Chained after the deploy branch's head (d41b7a90c2f1). Idempotent: a brand-new
database already has these columns from the baseline's create_all, so each column
is added only when missing.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4f6b8c0e2a4"
down_revision: Union[str, None] = "d41b7a90c2f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


MENTION_COLUMNS = (
    ("is_recommendation", lambda: sa.Column(
        "is_recommendation", sa.Boolean(), nullable=False, server_default=sa.false())),
    ("action", lambda: sa.Column("action", sa.String(length=16), nullable=True)),
    ("horizon", lambda: sa.Column("horizon", sa.String(length=16), nullable=True)),
    ("entry_price", lambda: sa.Column("entry_price", sa.Numeric(precision=18, scale=6), nullable=True)),
    ("target_price", lambda: sa.Column("target_price", sa.Numeric(precision=18, scale=6), nullable=True)),
    ("stop_price", lambda: sa.Column("stop_price", sa.Numeric(precision=18, scale=6), nullable=True)),
    ("trigger_price", lambda: sa.Column("trigger_price", sa.Numeric(precision=18, scale=6), nullable=True)),
    ("quote", lambda: sa.Column("quote", sa.Text(), nullable=True)),
    ("said_at", lambda: sa.Column("said_at", sa.String(length=40), nullable=True)),
    ("said_at_precision", lambda: sa.Column("said_at_precision", sa.String(length=8), nullable=True)),
    ("pushed_calls_to_trender", lambda: sa.Column(
        "pushed_calls_to_trender", sa.Boolean(), nullable=False, server_default=sa.false())),
    ("trender_calls_response", lambda: sa.Column("trender_calls_response", sa.Text(), nullable=True)),
)


def _columns(table: str) -> set:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if "track_record" not in _columns("source_configs"):
        with op.batch_alter_table("source_configs") as batch_op:
            batch_op.add_column(
                sa.Column("track_record", sa.Boolean(), nullable=False, server_default=sa.false())
            )

    existing = _columns("extracted_mentions")
    missing = [make() for name, make in MENTION_COLUMNS if name not in existing]
    if missing:
        with op.batch_alter_table("extracted_mentions") as batch_op:
            for column in missing:
                batch_op.add_column(column)


def downgrade() -> None:
    existing = _columns("extracted_mentions")
    present = [name for name, _ in reversed(MENTION_COLUMNS) if name in existing]
    if present:
        with op.batch_alter_table("extracted_mentions") as batch_op:
            for name in present:
                batch_op.drop_column(name)

    if "track_record" in _columns("source_configs"):
        with op.batch_alter_table("source_configs") as batch_op:
            batch_op.drop_column("track_record")
