"""
Podcast Engine - Main Application
FastAPI service for text-to-podcast conversion
"""
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings, AUDIO_FORMATS
from app.api.models import (
    PodcastRequest,
    PodcastResponse,
    HealthResponse,
    ErrorResponse,
    ProcessingStats,
    WebhookPodcastRequest,
    WebhookPodcastResponse,
)
from app.core.chunking import TextChunker
from app.core.tts import KokoroTTSClient
from app.core.audio import AudioProcessor
import httpx
import asyncio


# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown tasks
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # Initialize global TTS client
    app.state.tts_client = KokoroTTSClient()

    # Fetch available voices dynamically
    logger.info("Fetching available voices from Kokoro TTS...")
    app.state.available_voices = await app.state.tts_client.get_available_voices()
    logger.info(f"✓ Loaded {len(app.state.available_voices)} voices")

    # Verify storage directories exist
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.final_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Storage: {settings.storage_base_path}")
    logger.info(f"✓ Kokoro TTS: {settings.kokoro_tts_url}")

    app.state.start_time = time.time()

    yield

    # Shutdown
    logger.info("🛑 Shutting down Podcast Engine")
    await app.state.tts_client.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Convert text to professional podcasts/audiobooks using Kokoro TTS",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for GUI) - only if directory exists
if settings.enable_gui:
    static_dir = Path("app/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="app/static"), name="static")
    else:
        logger.warning("Static directory 'app/static' not found - static files disabled")


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message=str(exc),
        ).model_dump()
    )


# Routes
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "gui": "/gui" if settings.enable_gui else None,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    """Health check endpoint"""
    uptime = time.time() - request.app.state.start_time

    # Check Kokoro TTS
    kokoro_status = await request.app.state.tts_client.health_check()

    # Check storage
    storage_path = Path(settings.storage_base_path)
    storage_available = storage_path.exists() and storage_path.is_dir()

    # Overall health
    all_healthy = (
        kokoro_status["status"] == "healthy" and
        storage_available
    )

    return HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        version=settings.app_version,
        uptime_seconds=uptime,
        services={
            "kokoro_tts": kokoro_status,
        },
        system={
            "storage": {
                "available": storage_available,
                "path": str(storage_path),
            }
        }
    )


@app.get("/api/v1/voices", tags=["Voices"])
async def get_voices(request: Request, language: str = None):
    """
    Get available TTS voices, optionally filtered by language

    Args:
        language: Filter by language code (en, fr, es, ja, zh, pt, it, hi)

    Returns:
        Dict of voice_id → {name, gender, language, accent}
    """
    all_voices = request.app.state.available_voices

    if language:
        # Filter by language
        filtered_voices = {
            voice_id: voice_info
            for voice_id, voice_info in all_voices.items()
            if voice_info.get("language") == language.lower()
        }
        return filtered_voices
    else:
        return all_voices


@app.post("/api/v1/create-podcast", response_model=PodcastResponse, tags=["Podcast"])
async def create_podcast(request: Request, podcast_req: PodcastRequest):
    """
    Create podcast from text

    This endpoint converts text to a professional podcast/audiobook using:
    1. Smart text chunking (sentence-aware)
    2. Parallel TTS synthesis (Kokoro)
    3. ffmpeg audio merge
    4. Metadata embedding

    Returns:
        PodcastResponse with podcast file info
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())

    logger.info(f"[{job_id}] New podcast request: {podcast_req.metadata.title}")

    try:
        # Create job directory
        job_dir = Path(settings.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Step 1: Text chunking
        logger.info(f"[{job_id}] Step 1: Chunking text ({len(podcast_req.text)} chars)")
        chunker = TextChunker(
            max_chunk_size=podcast_req.tts_options.chunk_size,
            preserve_sentence=podcast_req.tts_options.preserve_sentence,
            remove_urls=podcast_req.tts_options.remove_urls,
            remove_markdown=podcast_req.tts_options.remove_markdown,
        )

        chunks = chunker.create_chunks(
            text=podcast_req.text,
            add_chapter_markers=podcast_req.tts_options.add_chapter_markers
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated")

        logger.info(f"[{job_id}] Generated {len(chunks)} chunks")

        # Step 2: TTS synthesis (parallel)
        logger.info(f"[{job_id}] Step 2: Synthesizing audio (parallel, max={podcast_req.processing_options.max_parallel_tts})")
        tts_client = request.app.state.tts_client

        tts_results = await tts_client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=chunks_dir,
            voice=podcast_req.tts_options.voice,
            speed=podcast_req.tts_options.speed,
            max_parallel=podcast_req.processing_options.max_parallel_tts,
            pause_between=podcast_req.tts_options.pause_between_chunks,
        )

        # Check for failures
        failed_chunks = [r for r in tts_results if not r[2]]
        if failed_chunks:
            logger.warning(f"[{job_id}] {len(failed_chunks)} chunks failed TTS")
            if not podcast_req.processing_options.retry_on_error:
                raise HTTPException(status_code=500, detail=f"{len(failed_chunks)} TTS chunks failed")

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]

        if not audio_files:
            raise HTTPException(status_code=500, detail="No audio chunks generated")

        # Step 3: Merge audio
        logger.info(f"[{job_id}] Step 3: Merging {len(audio_files)} audio files")
        audio_processor = AudioProcessor()

        output_filename = f"{podcast_req.metadata.title.replace(' ', '-').lower()}-{job_id[:8]}{AUDIO_FORMATS[podcast_req.audio_options.format]['extension']}"
        output_path = Path(settings.final_dir) / output_filename

        merged_audio = await audio_processor.merge_audio_files(
            input_files=audio_files,
            output_path=output_path,
            format=podcast_req.audio_options.format,
            bitrate=podcast_req.audio_options.bitrate,
            sample_rate=podcast_req.audio_options.sample_rate,
            channels=podcast_req.audio_options.channels,
            add_silence_start=podcast_req.audio_options.add_silence_start,
            add_silence_end=podcast_req.audio_options.add_silence_end,
        )

        # Step 4: Embed metadata
        logger.info(f"[{job_id}] Step 4: Embedding metadata")

        # Download cover image if URL provided
        cover_image_path = None
        if podcast_req.metadata.cover_image_url and podcast_req.audio_options.embed_cover:
            cover_image_path = job_dir / "cover.jpg"
            cover_image_path = await audio_processor.download_cover_image(
                str(podcast_req.metadata.cover_image_url),
                cover_image_path
            )

        audio_processor.embed_metadata(
            audio_path=merged_audio,
            title=podcast_req.metadata.title,
            author=podcast_req.metadata.author,
            description=podcast_req.metadata.description,
            album=podcast_req.metadata.publisher,
            genre=podcast_req.metadata.genre,
            narrator=podcast_req.metadata.narrator,
            publisher=podcast_req.metadata.publisher,
            copyright=podcast_req.metadata.copyright,
            publication_date=podcast_req.metadata.publication_date,
            cover_image_path=cover_image_path,
        )

        # Step 5: Get audio duration
        duration = await audio_processor.get_audio_duration(merged_audio)

        # Step 6: Cleanup temporary files on success
        # Always cleanup temp files after successful processing to save disk space
        logger.info(f"[{job_id}] Cleaning up temporary files")
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)

        # Calculate stats
        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_chunks=len(chunks),
            successful_chunks=len(audio_files),
            failed_chunks=len(failed_chunks),
            total_duration_seconds=duration,
            processing_time_seconds=processing_time,
            tts_api_calls=len(chunks),
            average_chunk_time=processing_time / len(chunks) if chunks else 0,
            text_length_chars=len(podcast_req.text),
            text_length_words=len(podcast_req.text.split()),
            estimated_listening_time_minutes=duration / 60,
        )

        logger.success(f"[{job_id}] Podcast created successfully in {processing_time:.1f}s")

        # Build response
        response = PodcastResponse(
            success=True,
            job_id=job_id,
            podcast={
                "filename": output_filename,
                "file_path": str(merged_audio),
                "file_size": merged_audio.stat().st_size,
                "duration_seconds": duration,
                "format": podcast_req.audio_options.format,
            },
            processing=stats,
            message=f"Podcast created successfully in {processing_time:.1f}s"
        )

        # Return binary file if requested
        if podcast_req.processing_options.return_binary:
            return FileResponse(
                path=merged_audio,
                media_type=f"audio/{podcast_req.audio_options.format}",
                filename=output_filename,
            )
        else:
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Podcast creation failed: {e}")

        # Cleanup on error
        if podcast_req.processing_options.cleanup_on_error:
            import shutil
            job_dir = Path(settings.temp_dir) / job_id
            shutil.rmtree(job_dir, ignore_errors=True)

        raise HTTPException(status_code=500, detail=str(e))


async def _process_podcast_job_async(
    job_id: str,
    podcast_req: WebhookPodcastRequest,
    app_state,
    callback_url: str = None,
    callbacks: dict = None
):
    """
    Background task for async podcast processing

    This function processes the podcast job asynchronously and sends
    a webhook callback when complete (success or failure).

    Args:
        job_id: Unique job identifier
        podcast_req: Podcast request with all options
        app_state: FastAPI app.state (for tts_client access)
        callback_url: Webhook URL to notify on completion
        callbacks: Original callbacks (source_workflow_id, etc.)
    """
    start_time = time.time()
    logger.info(f"[{job_id}] Starting async podcast processing: {podcast_req.metadata.title}")

    try:
        # Create job directory
        job_dir = Path(settings.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Step 1: Text chunking
        logger.info(f"[{job_id}] Step 1: Chunking text ({len(podcast_req.text)} chars)")
        chunker = TextChunker(
            max_chunk_size=podcast_req.tts_options.chunk_size,
            preserve_sentence=podcast_req.tts_options.preserve_sentence,
            remove_urls=podcast_req.tts_options.remove_urls,
            remove_markdown=podcast_req.tts_options.remove_markdown,
        )

        chunks = chunker.create_chunks(
            text=podcast_req.text,
            add_chapter_markers=podcast_req.tts_options.add_chapter_markers
        )

        if not chunks:
            raise ValueError("No text chunks generated")

        logger.info(f"[{job_id}] Generated {len(chunks)} chunks")

        # Step 2: TTS synthesis (parallel)
        logger.info(f"[{job_id}] Step 2: Synthesizing audio (parallel, max={podcast_req.processing_options.max_parallel_tts})")
        tts_client = app_state.tts_client

        tts_results = await tts_client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=chunks_dir,
            voice=podcast_req.tts_options.voice,
            speed=podcast_req.tts_options.speed,
            max_parallel=podcast_req.processing_options.max_parallel_tts,
            pause_between=podcast_req.tts_options.pause_between_chunks,
        )

        # Check for failures
        failed_chunks = [r for r in tts_results if not r[2]]
        if failed_chunks:
            logger.warning(f"[{job_id}] {len(failed_chunks)} chunks failed TTS")

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]

        if not audio_files:
            raise ValueError("No audio chunks generated")

        # Step 3: Merge audio
        logger.info(f"[{job_id}] Step 3: Merging {len(audio_files)} audio files")
        audio_processor = AudioProcessor()

        output_filename = f"{podcast_req.metadata.title.replace(' ', '-').lower()}-{job_id[:8]}{AUDIO_FORMATS[podcast_req.audio_options.format]['extension']}"
        output_path = Path(settings.final_dir) / output_filename

        merged_audio = await audio_processor.merge_audio_files(
            input_files=audio_files,
            output_path=output_path,
            format=podcast_req.audio_options.format,
            bitrate=podcast_req.audio_options.bitrate,
            sample_rate=podcast_req.audio_options.sample_rate,
            channels=podcast_req.audio_options.channels,
            add_silence_start=podcast_req.audio_options.add_silence_start,
            add_silence_end=podcast_req.audio_options.add_silence_end,
        )

        # Step 4: Embed metadata
        logger.info(f"[{job_id}] Step 4: Embedding metadata")

        # Download cover image if URL provided
        cover_image_path = None
        if podcast_req.metadata.cover_image_url and podcast_req.audio_options.embed_cover:
            cover_image_path = job_dir / "cover.jpg"
            cover_image_path = await audio_processor.download_cover_image(
                str(podcast_req.metadata.cover_image_url),
                cover_image_path
            )

        audio_processor.embed_metadata(
            audio_path=merged_audio,
            title=podcast_req.metadata.title,
            author=podcast_req.metadata.author,
            description=podcast_req.metadata.description,
            album=podcast_req.metadata.publisher,
            genre=podcast_req.metadata.genre,
            narrator=podcast_req.metadata.narrator,
            publisher=podcast_req.metadata.publisher,
            copyright=podcast_req.metadata.copyright,
            publication_date=podcast_req.metadata.publication_date,
            cover_image_path=cover_image_path,
        )

        # Step 5: Get audio duration
        duration = await audio_processor.get_audio_duration(merged_audio)

        # Step 6: Cleanup temporary files
        logger.info(f"[{job_id}] Cleaning up temporary files")
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)

        # Calculate stats
        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_chunks=len(chunks),
            successful_chunks=len(audio_files),
            failed_chunks=len(failed_chunks),
            total_duration_seconds=duration,
            processing_time_seconds=processing_time,
            tts_api_calls=len(chunks),
            average_chunk_time=processing_time / len(chunks) if chunks else 0,
            text_length_chars=len(podcast_req.text),
            text_length_words=len(podcast_req.text.split()),
            estimated_listening_time_minutes=duration / 60,
        )

        logger.success(f"[{job_id}] Async podcast created successfully in {processing_time:.1f}s")

        # Build podcast data for callback
        podcast_data = {
            "filename": output_filename,
            "file_path": str(merged_audio),
            "file_size": merged_audio.stat().st_size,
            "duration_seconds": duration,
            "format": podcast_req.audio_options.format,
        }

        # Send webhook callback on success
        if callback_url:
            await send_webhook_callback(
                callback_url=str(callback_url),
                job_id=job_id,
                success=True,
                podcast_data=podcast_data,
                processing_stats=stats,
                callbacks=callbacks
            )

    except Exception as e:
        logger.error(f"[{job_id}] Async podcast creation failed: {e}")

        # Cleanup on error
        if podcast_req.processing_options.cleanup_on_error:
            import shutil
            job_dir = Path(settings.temp_dir) / job_id
            shutil.rmtree(job_dir, ignore_errors=True)

        # Send webhook callback on failure
        if callback_url:
            await send_webhook_callback(
                callback_url=str(callback_url),
                job_id=job_id,
                success=False,
                error=str(e),
                callbacks=callbacks
            )


async def send_webhook_callback(
    callback_url: str,
    job_id: str,
    success: bool,
    podcast_data: dict = None,
    processing_stats: ProcessingStats = None,
    callbacks: dict = None,
    error: str = None,
    max_retries: int = 3
):
    """
    Send webhook callback with retry logic (exponential backoff)

    Args:
        callback_url: Webhook URL to POST results
        job_id: Job identifier
        success: True if processing succeeded, False otherwise
        podcast_data: Podcast information (filename, file_path, etc.)
        processing_stats: Processing statistics
        callbacks: Original callbacks (source_workflow_id, source_item_id)
        error: Error message if success=False
        max_retries: Maximum retry attempts (default: 3)
    """
    payload = {
        "job_id": job_id,
        "success": success,
        "timestamp": time.time(),
    }

    if success:
        payload["podcast"] = podcast_data
        if processing_stats:
            payload["processing"] = processing_stats.model_dump()
        if callbacks:
            payload["callbacks"] = callbacks
    else:
        payload["error"] = error

    # Retry with exponential backoff: 1s, 2s, 4s
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{job_id}] Sending webhook callback to {callback_url} (attempt {attempt}/{max_retries})")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    callback_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code in [200, 201, 202, 204]:
                    logger.success(f"[{job_id}] Webhook callback delivered successfully (status {response.status_code})")
                    return True
                else:
                    logger.warning(f"[{job_id}] Webhook callback returned status {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.error(f"[{job_id}] Webhook callback attempt {attempt} failed: {e}")

        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            wait_time = 2 ** (attempt - 1)  # 1s, 2s, 4s
            logger.info(f"[{job_id}] Retrying webhook callback in {wait_time}s...")
            await asyncio.sleep(wait_time)

    logger.error(f"[{job_id}] Webhook callback failed after {max_retries} attempts")
    return False


@app.post("/api/v1/webhook/create-podcast", response_model=WebhookPodcastResponse, tags=["Webhook"])
async def webhook_create_podcast(
    request: Request,
    podcast_req: WebhookPodcastRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(None, alias="X-API-KEY")
):
    """
    Webhook endpoint for automated podcast creation (n8n integration)

    This endpoint is designed for n8n workflows and external automation.
    It extends the standard /api/v1/create-podcast with:
    - X-API-KEY authentication
    - Optional callbacks tracking (source_workflow_id, source_item_id)
    - Optional base64 binary response (for direct upload)
    - Optional chapters support (PDF+LLM use case)
    - Async mode support (background processing with webhook callback)

    Authentication:
        Requires X-API-KEY header matching PODCAST_ENGINE_API_KEY env var

    Async Mode:
        If processing_options.async_mode = True:
        - Returns immediately with job_id
        - Processes in background
        - Sends webhook callback to processing_options.callback_url when complete

    Returns:
        WebhookPodcastResponse with optional binary_data field
    """
    # Authentication check
    if settings.api_key:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing X-API-KEY header"
            )
        if x_api_key != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

    start_time = time.time()
    job_id = str(uuid.uuid4())

    logger.info(f"[{job_id}] Webhook podcast request: {podcast_req.metadata.title}")
    if podcast_req.callbacks:
        logger.info(f"[{job_id}] Callbacks: workflow={podcast_req.callbacks.source_workflow_id}, item={podcast_req.callbacks.source_item_id}")

    # ============================================================================
    # ASYNC MODE: Return immediately and process in background
    # ============================================================================
    if podcast_req.processing_options.async_mode:
        logger.info(f"[{job_id}] Async mode enabled - submitting job to background")

        # Validate callback_url if async mode
        callback_url = podcast_req.processing_options.callback_url
        if not callback_url:
            raise HTTPException(
                status_code=400,
                detail="callback_url is required when async_mode=true"
            )

        # Launch background task
        background_tasks.add_task(
            _process_podcast_job_async,
            job_id=job_id,
            podcast_req=podcast_req,
            app_state=request.app.state,
            callback_url=str(callback_url),
            callbacks=podcast_req.callbacks.model_dump() if podcast_req.callbacks else None
        )

        # Return immediately with job_id
        return WebhookPodcastResponse(
            success=True,
            job_id=job_id,
            message=f"Job {job_id} submitted for async processing. Callback will be sent to {callback_url} when complete.",
            callbacks=podcast_req.callbacks
        )

    # ============================================================================
    # SYNC MODE: Process synchronously (original behavior)
    # ============================================================================
    logger.info(f"[{job_id}] Sync mode - processing immediately")

    try:
        # Create job directory
        job_dir = Path(settings.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Step 1: Text chunking
        logger.info(f"[{job_id}] Step 1: Chunking text ({len(podcast_req.text)} chars)")
        chunker = TextChunker(
            max_chunk_size=podcast_req.tts_options.chunk_size,
            preserve_sentence=podcast_req.tts_options.preserve_sentence,
            remove_urls=podcast_req.tts_options.remove_urls,
            remove_markdown=podcast_req.tts_options.remove_markdown,
        )

        chunks = chunker.create_chunks(
            text=podcast_req.text,
            add_chapter_markers=podcast_req.tts_options.add_chapter_markers
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated")

        logger.info(f"[{job_id}] Generated {len(chunks)} chunks")

        # Step 2: TTS synthesis (parallel)
        logger.info(f"[{job_id}] Step 2: Synthesizing audio (parallel, max={podcast_req.processing_options.max_parallel_tts})")
        tts_client = request.app.state.tts_client

        tts_results = await tts_client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=chunks_dir,
            voice=podcast_req.tts_options.voice,
            speed=podcast_req.tts_options.speed,
            max_parallel=podcast_req.processing_options.max_parallel_tts,
            pause_between=podcast_req.tts_options.pause_between_chunks,
        )

        # Check for failures
        failed_chunks = [r for r in tts_results if not r[2]]
        if failed_chunks:
            logger.warning(f"[{job_id}] {len(failed_chunks)} chunks failed TTS")
            if not podcast_req.processing_options.retry_on_error:
                raise HTTPException(status_code=500, detail=f"{len(failed_chunks)} TTS chunks failed")

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]

        if not audio_files:
            raise HTTPException(status_code=500, detail="No audio chunks generated")

        # Step 3: Merge audio
        logger.info(f"[{job_id}] Step 3: Merging {len(audio_files)} audio files")
        audio_processor = AudioProcessor()

        output_filename = f"{podcast_req.metadata.title.replace(' ', '-').lower()}-{job_id[:8]}{AUDIO_FORMATS[podcast_req.audio_options.format]['extension']}"
        output_path = Path(settings.final_dir) / output_filename

        merged_audio = await audio_processor.merge_audio_files(
            input_files=audio_files,
            output_path=output_path,
            format=podcast_req.audio_options.format,
            bitrate=podcast_req.audio_options.bitrate,
            sample_rate=podcast_req.audio_options.sample_rate,
            channels=podcast_req.audio_options.channels,
            add_silence_start=podcast_req.audio_options.add_silence_start,
            add_silence_end=podcast_req.audio_options.add_silence_end,
        )

        # Step 4: Embed metadata
        logger.info(f"[{job_id}] Step 4: Embedding metadata")

        # Download cover image if URL provided
        cover_image_path = None
        if podcast_req.metadata.cover_image_url and podcast_req.audio_options.embed_cover:
            cover_image_path = job_dir / "cover.jpg"
            cover_image_path = await audio_processor.download_cover_image(
                str(podcast_req.metadata.cover_image_url),
                cover_image_path
            )

        audio_processor.embed_metadata(
            audio_path=merged_audio,
            title=podcast_req.metadata.title,
            author=podcast_req.metadata.author,
            description=podcast_req.metadata.description,
            album=podcast_req.metadata.publisher,
            genre=podcast_req.metadata.genre,
            narrator=podcast_req.metadata.narrator,
            publisher=podcast_req.metadata.publisher,
            copyright=podcast_req.metadata.copyright,
            publication_date=podcast_req.metadata.publication_date,
            cover_image_path=cover_image_path,
        )

        # Step 5: Get audio duration
        duration = await audio_processor.get_audio_duration(merged_audio)

        # Step 6: Encode to base64 if requested
        binary_data = None
        if podcast_req.processing_options.return_binary:
            logger.info(f"[{job_id}] Encoding audio to base64 for response")
            import base64
            with open(merged_audio, "rb") as f:
                binary_data = base64.b64encode(f.read()).decode("utf-8")

        # Step 7: Cleanup temporary files
        logger.info(f"[{job_id}] Cleaning up temporary files")
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)

        # Calculate stats
        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_chunks=len(chunks),
            successful_chunks=len(audio_files),
            failed_chunks=len(failed_chunks),
            total_duration_seconds=duration,
            processing_time_seconds=processing_time,
            tts_api_calls=len(chunks),
            average_chunk_time=processing_time / len(chunks) if chunks else 0,
            text_length_chars=len(podcast_req.text),
            text_length_words=len(podcast_req.text.split()),
            estimated_listening_time_minutes=duration / 60,
        )

        logger.success(f"[{job_id}] Webhook podcast created successfully in {processing_time:.1f}s")

        # Build response
        response = WebhookPodcastResponse(
            success=True,
            job_id=job_id,
            podcast={
                "filename": output_filename,
                "file_path": str(merged_audio),
                "file_size": merged_audio.stat().st_size,
                "duration_seconds": duration,
                "format": podcast_req.audio_options.format,
                "binary_data": binary_data,  # Include base64 if requested
            },
            processing=stats,
            callbacks=podcast_req.callbacks,  # Echo callbacks
            message=f"Podcast created successfully in {processing_time:.1f}s"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Webhook podcast creation failed: {e}")

        # Cleanup on error
        if podcast_req.processing_options.cleanup_on_error:
            import shutil
            job_dir = Path(settings.temp_dir) / job_id
            shutil.rmtree(job_dir, ignore_errors=True)

        raise HTTPException(status_code=500, detail=str(e))


# Import GUI routes if enabled
if settings.enable_gui:
    from app.gui import routes as gui_routes
    app.include_router(gui_routes.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
