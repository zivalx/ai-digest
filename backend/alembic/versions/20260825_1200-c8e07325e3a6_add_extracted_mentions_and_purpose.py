"""add extracted_mentions table and source_configs.purpose

Channel-intelligence feeder: new ExtractedMention table (ticker picks with
citations + trender push tracking) and a `purpose` column on source_configs so
the feeder knows which configs to run ("finance").

Revision ID: c8e07325e3a6
Revises:
Create Date: 2026-08-25 12:00:00

Safe on every starting point: on an empty database it builds the whole current schema
(there is no earlier baseline revision); on an older create_all database it only adds
what is missing. The container runs ``alembic upgrade head`` on every start.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from src.models.base import GUID


# revision identifiers, used by Alembic.
revision: str = "c8e07325e3a6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())

    if "source_configs" not in tables:
        # Brand-new database: there is no baseline revision before this one, so build
        # the whole current schema here (this already includes everything below).
        from src.models import Base

        Base.metadata.create_all(bind)
        return

    # Existing database (tables made by create_all before the feeder existed).
    columns = {c["name"] for c in sa.inspect(bind).get_columns("source_configs")}
    if "purpose" not in columns:
        with op.batch_alter_table("source_configs") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "purpose",
                    sa.String(length=50),
                    nullable=False,
                    server_default="general",
                )
            )

    if "extracted_mentions" in tables:
        return

    op.create_table(
        "extracted_mentions",
        sa.Column("id", GUID(), nullable=False),
        sa.Column("cycle_id", GUID(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=True),
        sa.Column("thesis", sa.Text(), nullable=True),
        sa.Column("stance", sa.String(length=16), nullable=True),
        sa.Column("sentiment_score", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("citation_json", sa.JSON(), nullable=False),
        sa.Column("dedupe_key", sa.String(length=512), nullable=False),
        sa.Column("pushed_to_trender", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("trender_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["cycle_id"], ["cycles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedupe_key"),
    )
    op.create_index("idx_mention_cycle_id", "extracted_mentions", ["cycle_id"])
    op.create_index("idx_mention_symbol", "extracted_mentions", ["symbol"])


def downgrade() -> None:
    op.drop_index("idx_mention_symbol", table_name="extracted_mentions")
    op.drop_index("idx_mention_cycle_id", table_name="extracted_mentions")
    op.drop_table("extracted_mentions")

    with op.batch_alter_table("source_configs") as batch_op:
        batch_op.drop_column("purpose")
