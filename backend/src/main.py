"""
Daigest FastAPI Application
Orchestrates data collection from multiple sources and generates AI summaries.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Daigest API",
    description="Multi-source data collection and AI summarization platform",
    version="2.0.0",
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # React dev server
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


@app.get("/")
async def root():
    """Health check endpoint."""
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


# Register API routes
from .routes import cycle_router, config_router

app.include_router(cycle_router, prefix="/api/cycles", tags=["Cycles"])
app.include_router(config_router, prefix="/api/configs", tags=["Configurations"])
