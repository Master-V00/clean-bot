"""
Main application entry point - Start this for production deployment
"""
import asyncio
import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger

from app.server_config import settings
from app.api.main import app as api_app
from app.workers.task_worker import worker
from app.utils.device import startup_banner as _device_banner

# Redirect torch.hub downloads to project models dir (not system cache)
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "models")
os.environ["TORCH_HOME"] = _models_dir


# Suppress uvicorn access logs (the HTTP 404 GET /portal-resolver spam)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# Suppress httpx info/debug logs (Google Drive client logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Setup logging
logger.remove()
logger.add(
    "logs/panelcleaner.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)
logger.add(
    lambda msg: print(msg, end=""),
    format="<level>{level: <8}</level> | {time:HH:mm:ss} | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("PanelCleaner Server Starting...")
    logger.info(_device_banner())

    # GPU capability benchmark
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            capability = torch.cuda.get_device_capability()
            logger.info(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f}GB | Compute: {capability[0]}.{capability[1]}")

            if gpu_mem >= 8:
                logger.info("GPU Power: HIGH — concurrent_models=8 recommended")
            elif gpu_mem >= 4:
                logger.info("GPU Power: MEDIUM — concurrent_models=4 recommended")
            else:
                logger.info("GPU Power: LOW — concurrent_models=2 recommended")
        else:
            logger.warning("No CUDA GPU detected — running on CPU (significantly slower)")
    except Exception as e:
        logger.debug(f"GPU detection skipped: {e}")

    logger.debug("Listing all registered routes:")
    for route in app.routes:
        logger.debug(f"Route: {route.path} | Methods: {getattr(route, 'methods', 'N/A')}")
    # ────────────────────────────────────────────────

    # Start background worker as a background task
    logger.info("Starting background task worker...")
    worker_task = asyncio.create_task(worker.process_queue())

    # Start cleanup scheduler as a background task (not awaited)
    logger.info("Starting cleanup scheduler...")
    cleanup_task = asyncio.create_task(
        _run_cleanup_loop(
            interval_seconds=settings.CLEANUP_INTERVAL_SECONDS,
            expiration_minutes=settings.EXPIRATION_TIME_MINUTES,
        )
    )

    logger.info("PanelCleaner Server Ready!")

    yield

    # Shutdown
    logger.info("Shutting down PanelCleaner Server...")
    worker_task.cancel()
    cleanup_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("PanelCleaner Server Stopped")


async def _run_cleanup_loop(interval_seconds: int, expiration_minutes: int):
    """Run the cleanup scheduler as a periodic background task."""
    from app.services.cleanup import CleanupScheduler
    scheduler = CleanupScheduler(interval_seconds, expiration_minutes)
    while True:
        scheduler.cleanup_expired_outputs()
        await asyncio.sleep(interval_seconds)


# Create main app
app = FastAPI(
    title="PanelCleaner Server",
    description="Background task processing server for PanelCleaner Discord Bot",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (single source - no duplicate route injection)
app.mount("/api", api_app)


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=False,
    )
