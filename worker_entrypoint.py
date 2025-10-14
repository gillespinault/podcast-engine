#!/usr/bin/env python3
"""
Podcast Engine - RQ Worker Entrypoint
Starts RQ workers to process podcast jobs from Redis Queue
"""
import sys
from loguru import logger
from rq import Worker, Queue
from app.worker import redis_conn
from app.config import settings

# Configure loguru for worker
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)

if __name__ == "__main__":
    logger.info("🚀 Starting RQ Worker for Podcast Engine")
    logger.info(f"Redis: {settings.redis_host}:{settings.redis_port}/{settings.redis_db}")
    logger.info(f"Queue: podcast_processing")

    # Listen to podcast_processing queue (RQ 2.0 doesn't need Connection context)
    import os
    worker = Worker(
        queues=['podcast_processing'],
        connection=redis_conn,
        name=f"podcast-worker-{os.getpid()}",  # Unique name per process
    )
    logger.info(f"✓ Worker registered: {worker.name}")
    logger.info("📝 Listening for jobs...")

    # Start listening (blocking)
    worker.work(with_scheduler=True, logging_level=settings.log_level)
