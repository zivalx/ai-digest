"""
Source Configuration API Routes.

Handles CRUD operations for source configurations.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import logging

from ..database import get_db
from ..models import SourceConfig
from ..schemas import (
    SourceConfigCreate,
    SourceConfigUpdate,
    SourceConfigResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=SourceConfigResponse, status_code=201)
async def create_config(
    config_data: SourceConfigCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new source configuration."""
    logger.info(f"Creating source config: {config_data.name} ({config_data.source_type})")

    config = SourceConfig(
        name=config_data.name,
        source_type=config_data.source_type,
        credential_ref=config_data.credential_ref,
        collect_spec=config_data.collect_spec,
        enabled=config_data.enabled,
        purpose=config_data.purpose,
    )

    db.add(config)
    await db.commit()
    await db.refresh(config)

    return SourceConfigResponse.model_validate(config)


@router.get("/", response_model=List[SourceConfigResponse])
async def list_configs(
    source_type: str = None,
    enabled: bool = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all source configurations.

    Query parameters:
    - source_type: Filter by source type
    - enabled: Filter by enabled status
    """
    query = select(SourceConfig)

    if source_type:
        query = query.where(SourceConfig.source_type == source_type)

    if enabled is not None:
        query = query.where(SourceConfig.enabled == enabled)

    result = await db.execute(query)
    configs = result.scalars().all()

    return [SourceConfigResponse.model_validate(c) for c in configs]


@router.get("/{config_id}", response_model=SourceConfigResponse)
async def get_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific source configuration."""
    result = await db.execute(
        select(SourceConfig).where(SourceConfig.id == config_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    return SourceConfigResponse.model_validate(config)


@router.put("/{config_id}", response_model=SourceConfigResponse)
async def update_config(
    config_id: str,
    config_data: SourceConfigUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a source configuration."""
    result = await db.execute(
        select(SourceConfig).where(SourceConfig.id == config_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    # Update fields
    if config_data.name is not None:
        config.name = config_data.name
    if config_data.credential_ref is not None:
        config.credential_ref = config_data.credential_ref
    if config_data.collect_spec is not None:
        config.collect_spec = config_data.collect_spec
    if config_data.enabled is not None:
        config.enabled = config_data.enabled
    if config_data.purpose is not None:
        config.purpose = config_data.purpose

    await db.commit()
    await db.refresh(config)

    logger.info(f"Updated config {config_id}")

    return SourceConfigResponse.model_validate(config)


@router.delete("/{config_id}", status_code=204)
async def delete_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a source configuration."""
    result = await db.execute(
        select(SourceConfig).where(SourceConfig.id == config_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    await db.delete(config)
    await db.commit()

    logger.info(f"Deleted config {config_id}")
