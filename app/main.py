"""
Podcast Engine - Main Application
FastAPI service for text-to-podcast conversion
"""
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header
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

# Mount static files (for GUI)
if settings.enable_gui:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")


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


@app.post("/api/v1/webhook/create-podcast", response_model=WebhookPodcastResponse, tags=["Webhook"])
async def webhook_create_podcast(
    request: Request,
    podcast_req: WebhookPodcastRequest,
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

    Authentication:
        Requires X-API-KEY header matching PODCAST_ENGINE_API_KEY env var

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
