"""
Daigest FastAPI Application
Orchestrates data collection from multiple sources and generates AI summaries.
"""
import logging
import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import health  # imported at start-up: its STARTED_AT is the process start time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Daigest API",
    description="Multi-source data collection and AI summarization platform",
    version="2.0.0",
)

def cors_origins() -> List[str]:
    """Allowed browser origins. Only needed for the React dev server: in production the UI
    is served by this app, so the browser talks to the same origin."""
    raw = os.getenv("CORS_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:3000", "http://localhost:3001"]


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Holds the finance-feeder scheduler instance when enabled.
feeder_scheduler = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    logger.info("Daigest API starting up...")
    from src.database import init_db
    logger.info("Initializing database tables...")
    await init_db()
    logger.info("Database initialization complete")

    # Channel-intelligence feeder scheduler (guarded so tests/local runs don't auto-start it).
    from src.config import get_settings
    settings = get_settings()
    if settings.feeder_enabled:
        global feeder_scheduler
        from src.services.scheduler_service import SchedulerService
        feeder_scheduler = SchedulerService()
        feeder_scheduler.start()
        logger.info("Finance feeder enabled (interval=%ss)", settings.feeder_interval_seconds)
    else:
        logger.info("Finance feeder disabled (set FEEDER_ENABLED=true to enable)")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background services on shutdown."""
    global feeder_scheduler
    if feeder_scheduler is not None:
        feeder_scheduler.shutdown()
        feeder_scheduler = None


async def root():
    """Status endpoint (only registered when no UI is being served at /)."""
    return {
        "status": "ok",
        "service": "Daigest API",
        "version": "2.0.0",
    }


@app.get("/health")
async def health_check():
    """Detailed health check — pings the database."""
    from sqlalchemy import text
    from src.database import AsyncSessionLocal

    database = "connected"
    status = "healthy"
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        database = f"error: {exc.__class__.__name__}"
        status = "degraded"

    return {
        "status": status,
        "database": database,
        "services": {
            "collectors": "ready",
            "summarizer": "ready",
        },
    }


@app.get("/health/deep")
async def health_deep():
    """Deep health for the zivalx-ops watchdog: database, feeder age, extraction, push backlog.

    Looked up on the module at call time so tests can replace ``src.health.deep_health``.
    """
    result = await health.deep_health()
    return JSONResponse(result, status_code=health.status_code_for(result))


# Register API routes
from .routes import cycle_router, config_router

app.include_router(cycle_router, prefix="/api/cycles", tags=["Cycles"])
app.include_router(config_router, prefix="/api/configs", tags=["Configurations"])


def mount_ui(application: FastAPI, directory: Optional[str]) -> bool:
    """Serve a built React app from ``directory`` at / (same origin as the API).

    Must run after every API route is registered: the mount is a catch-all. Returns False
    (and mounts nothing) when there is no build there.
    """
    if not directory or not os.path.isfile(os.path.join(directory, "index.html")):
        return False
    application.mount("/", StaticFiles(directory=directory, html=True), name="ui")
    return True


# The production image copies the frontend build to UI_DIR; local development has none.
if mount_ui(app, os.getenv("UI_DIR")):
    logger.info("Serving the UI from %s", os.getenv("UI_DIR"))
else:
    app.add_api_route("/", root, methods=["GET"])
