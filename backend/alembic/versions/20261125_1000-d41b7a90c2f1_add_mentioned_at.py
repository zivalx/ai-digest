"""add extracted_mentions.mentioned_at

Keeps the publish time of each pick so a push to trender that failed can be retried
on a later cycle with the correct ``mentioned_at``.

Revision ID: d41b7a90c2f1
Revises: c8e07325e3a6
Create Date: 2026-11-25 10:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d41b7a90c2f1"
down_revision: Union[str, None] = "c8e07325e3a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A fresh database already got the column from the previous revision's create_all.
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("extracted_mentions")}
    if "mentioned_at" not in columns:
        op.add_column(
            "extracted_mentions",
            sa.Column("mentioned_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    with op.batch_alter_table("extracted_mentions") as batch_op:
        batch_op.drop_column("mentioned_at")
