"""
Podcast Engine - Main Application
FastAPI service for text-to-podcast conversion
"""
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, File, UploadFile, Form
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
    JobStatusResponse,
    JobListResponse,
    JobProgress,
    PodcastMetadata,
    TTSOptions,
    AudioOptions,
    ProcessingOptions,
    TTSProviderInfo,
    TTSVoiceInfo,
    TTSProvidersResponse,
    TTSVoicesResponse,
)
from app.core.chunking import TextChunker
from app.core.tts_providers import TTSProviderManager
from app.core.audio import AudioProcessor
from app.core.pdf_processor import SimplePDFProcessor, validate_pdf, PDFValidationError
from app.llm.gemini import GeminiClient
from app.llm.voice_selector import select_voice
from app.worker import enqueue_podcast_job, redis_conn, podcast_queue  # RQ job queue
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
import httpx
import asyncio
import json
import tempfile


# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown tasks
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # Initialize global TTS client (multi-provider: Google Cloud TTS, Piper, Kokoro)
    app.state.tts_client = TTSProviderManager()
    await app.state.tts_client.initialize()
    logger.info(f"✓ TTS Providers initialized: {', '.join([p.name for p in app.state.tts_client.providers])}")

    # Set empty available_voices (multi-provider system uses provider-specific voices)
    app.state.available_voices = {}

    # Verify storage directories exist
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.final_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Storage: {settings.storage_base_path}")
    logger.info(f"✓ TTS Provider Priority: {settings.tts_provider}")

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


@app.get("/api/v1/tts/providers", response_model=TTSProvidersResponse, tags=["TTS"])
async def get_tts_providers(request: Request):
    """
    Get list of available TTS providers

    Returns information about all configured TTS providers:
    - Google Cloud TTS (Neural2 voices, 1M chars/month free)
    - Piper TTS (self-hosted, CPU-optimized)
    - Kokoro TTS (self-hosted, fallback)

    Returns:
        TTSProvidersResponse with list of providers and default provider
    """
    tts_manager: TTSProviderManager = request.app.state.tts_client

    # Provider metadata
    provider_metadata = {
        "google": {
            "id": "google",
            "name": "Google Cloud TTS",
            "description": "High-quality Neural2 voices with natural prosody. Free tier: 1M characters/month.",
            "languages": ["fr", "en", "es", "de", "it", "pt", "ja", "zh"]
        },
        "piper": {
            "id": "piper",
            "name": "Piper TTS",
            "description": "Self-hosted TTS engine via openedai-speech. CPU-optimized, fast inference (<1s latency).",
            "languages": ["fr", "en", "es", "de"]
        },
        "kokoro": {
            "id": "kokoro",
            "name": "Kokoro TTS",
            "description": "Self-hosted multilingual TTS. Fallback provider (known memory issues).",
            "languages": ["fr", "en", "es", "ja", "zh", "pt", "it", "hi"]
        }
    }

    # Build list of available providers based on initialized providers
    providers_list = []
    for provider in tts_manager.providers:
        provider_id = provider.name.lower().split()[0]  # Extract "google" from "Google Cloud TTS"

        if provider_id in provider_metadata:
            metadata = provider_metadata[provider_id]
            providers_list.append(TTSProviderInfo(
                id=metadata["id"],
                name=metadata["name"],
                description=metadata["description"],
                available=True,  # Provider is initialized, so it's available
                languages=metadata["languages"]
            ))

    # Get default provider (first in list)
    default_provider_id = tts_manager.current_provider.name.lower().split()[0] if tts_manager.current_provider else "google"

    return TTSProvidersResponse(
        providers=providers_list,
        default_provider=default_provider_id
    )


@app.get("/api/v1/tts/voices", response_model=TTSVoicesResponse, tags=["TTS"])
async def get_tts_voices(
    request: Request,
    provider: str = None,
    language: str = None
):
    """
    Get list of available TTS voices, optionally filtered by provider and/or language

    Args:
        provider: Filter by provider (google, piper, kokoro)
        language: Filter by language code (fr, en, es, etc.)

    Returns:
        TTSVoicesResponse with list of voices matching filters

    Examples:
        GET /api/v1/tts/voices - All voices from all providers
        GET /api/v1/tts/voices?provider=google - All Google voices
        GET /api/v1/tts/voices?language=fr - All French voices
        GET /api/v1/tts/voices?provider=google&language=fr - Google French voices
    """
    # Voice catalog (comprehensive list of voices per provider)
    voice_catalog = {
        "google": [
            TTSVoiceInfo(id="fr-FR-Neural2-A", name="French Female (Neural2-A)", language="fr", gender="female", accent="French", provider="google"),
            TTSVoiceInfo(id="fr-FR-Neural2-B", name="French Male (Neural2-B)", language="fr", gender="male", accent="French", provider="google"),
            TTSVoiceInfo(id="fr-FR-Neural2-C", name="French Female (Neural2-C)", language="fr", gender="female", accent="French", provider="google"),
            TTSVoiceInfo(id="fr-FR-Neural2-D", name="French Male (Neural2-D)", language="fr", gender="male", accent="French", provider="google"),
            TTSVoiceInfo(id="fr-FR-Neural2-E", name="French Female (Neural2-E)", language="fr", gender="female", accent="French", provider="google"),
            TTSVoiceInfo(id="en-US-Neural2-A", name="American Male (Neural2-A)", language="en", gender="male", accent="American", provider="google"),
            TTSVoiceInfo(id="en-US-Neural2-C", name="American Female (Neural2-C)", language="en", gender="female", accent="American", provider="google"),
            TTSVoiceInfo(id="en-US-Neural2-D", name="American Male (Neural2-D)", language="en", gender="male", accent="American", provider="google"),
            TTSVoiceInfo(id="en-US-Neural2-F", name="American Female (Neural2-F)", language="en", gender="female", accent="American", provider="google"),
            TTSVoiceInfo(id="en-US-Neural2-H", name="American Female (Neural2-H)", language="en", gender="female", accent="American", provider="google"),
        ],
        "piper": [
            TTSVoiceInfo(id="fr_FR-siwis-medium", name="French Siwis (Medium)", language="fr", gender="female", accent="French", provider="piper"),
            TTSVoiceInfo(id="en_US-lessac-medium", name="American Lessac (Medium)", language="en", gender="female", accent="American", provider="piper"),
        ],
        "kokoro": [
            TTSVoiceInfo(id="ff_siwis", name="French Female (Siwis)", language="fr", gender="female", accent="French", provider="kokoro"),
            TTSVoiceInfo(id="ff_bella", name="French Female (Bella)", language="fr", gender="female", accent="French", provider="kokoro"),
            TTSVoiceInfo(id="ff_sarah", name="French Female (Sarah)", language="fr", gender="female", accent="French", provider="kokoro"),
            TTSVoiceInfo(id="af_bella", name="American Female (Bella)", language="en", gender="female", accent="American", provider="kokoro"),
            TTSVoiceInfo(id="af_sarah", name="American Female (Sarah)", language="en", gender="female", accent="American", provider="kokoro"),
            TTSVoiceInfo(id="am_adam", name="American Male (Adam)", language="en", gender="male", accent="American", provider="kokoro"),
            TTSVoiceInfo(id="am_michael", name="American Male (Michael)", language="en", gender="male", accent="American", provider="kokoro"),
        ]
    }

    # Collect all voices (or filter by provider)
    all_voices = []
    if provider:
        # Single provider
        if provider.lower() in voice_catalog:
            all_voices = voice_catalog[provider.lower()]
        else:
            # Unknown provider - return empty list
            return TTSVoicesResponse(voices=[], provider=provider, language=language)
    else:
        # All providers
        for voices in voice_catalog.values():
            all_voices.extend(voices)

    # Filter by language if specified
    if language:
        all_voices = [v for v in all_voices if v.language == language.lower()]

    # Return filtered voices
    return TTSVoicesResponse(
        voices=all_voices,
        provider=provider or "all",
        language=language
    )


@app.post("/api/v1/extract-metadata", tags=["Metadata"])
async def extract_metadata(
    request: Request,
    # Text mode: JSON body with text field
    text: str = None,
    # PDF mode: multipart/form-data
    pdf_file: UploadFile = File(default=None),
    filename: str = Form(default=None),
    source_url: str = Form(default=None),
):
    """
    Extract metadata from content using Gemini LLM (autofill feature)

    This endpoint supports two input modes:

    **Text Mode** (application/json):
    - Provide JSON body with `text` field

    **PDF Mode** (multipart/form-data):
    - Upload `pdf_file` (PDF document)
    - Optionally provide `filename` and `source_url` hints

    Uses Gemini AI to extract:
    - Title (from content or filename)
    - Author
    - Language (ISO 639-1 code)
    - Description (1-2 sentence summary)
    - Genre
    - Tags (3-7 keywords)
    - Publication date (if available)
    - Voice suggestion based on detected language

    Returns:
        {
            "title": "Extracted title",
            "author": "Author Name",
            "language": "fr",
            "description": "Brief summary...",
            "genre": "Technology",
            "tags": ["AI", "machine-learning", "education"],
            "publication_date": "2025" or null,
            "voice_suggestion": "ff_siwis"
        }
    """
    try:
        # ============================================================================
        # Mode Detection: Text (JSON) vs PDF (multipart/form-data)
        # ============================================================================
        if pdf_file:
            # ========================================================================
            # PDF MODE: Extract text from PDF, then extract metadata
            # ========================================================================
            logger.info(f"Metadata extraction: PDF mode ({pdf_file.filename})")

            # Save uploaded PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf_path = Path(tmp_pdf.name)
                content = await pdf_file.read()
                tmp_pdf.write(content)
                tmp_pdf.flush()

            try:
                # Validate PDF
                validate_pdf(tmp_pdf_path)

                # Upload PDF to Gemini File API and extract metadata
                logger.info("Uploading PDF to Gemini File API for metadata extraction")
                gemini_client = GeminiClient()

                # Upload PDF file
                uploaded_file = gemini_client.upload_pdf_file(tmp_pdf_path, pdf_file.filename)

                # Extract metadata directly from PDF (without downloading text)
                metadata_result = gemini_client.extract_metadata_from_pdf(
                    pdf_file=uploaded_file,
                    filename=pdf_file.filename,
                    source_url=source_url
                )

                logger.info(
                    f"Metadata extracted from PDF via Gemini: title='{metadata_result['title']}', "
                    f"author='{metadata_result['author']}', language={metadata_result['language']}"
                )

                # Return metadata immediately (no need to extract text)
                return metadata_result

            except PDFValidationError as e:
                tmp_pdf_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                tmp_pdf_path.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")
            finally:
                # Clean up temporary PDF
                tmp_pdf_path.unlink(missing_ok=True)

        elif text:
            # ========================================================================
            # TEXT MODE: Use provided text directly
            # ========================================================================
            logger.info(f"Metadata extraction: Text mode ({len(text)} chars)")
            extracted_text = text

        else:
            # Neither PDF nor text provided
            raise HTTPException(
                status_code=400,
                detail="Either provide 'text' (JSON) OR 'pdf_file' (multipart/form-data)"
            )

        # ============================================================================
        # Extract metadata using Gemini
        # ============================================================================
        try:
            gemini_client = GeminiClient()
            metadata_result = gemini_client.extract_metadata(
                text=extracted_text,
                filename=filename,
                source_url=source_url
            )

            logger.info(
                f"Metadata extracted: title='{metadata_result['title']}', "
                f"author='{metadata_result['author']}', language={metadata_result['language']}"
            )

            return metadata_result

        except ValueError as e:
            # Gemini API key not configured
            if "GEMINI_API_KEY" in str(e):
                raise HTTPException(
                    status_code=501,
                    detail="Metadata extraction requires Gemini API key. Please configure GEMINI_API_KEY environment variable in Dokploy."
                )
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        except Exception as e:
            logger.exception("Gemini metadata extraction failed")
            raise HTTPException(status_code=500, detail=f"Gemini metadata extraction failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job Status Endpoints (for GUI progress tracking)
# ============================================================================

def _rq_job_to_status_response(job: Job) -> JobStatusResponse:
    """
    Convert RQ Job object to JobStatusResponse model

    Reads progress metadata from Redis (stored by worker.py)
    """
    # Get job metadata (stored by worker in meta field)
    meta = job.meta or {}
    title = meta.get("title")

    # Parse progress from meta
    progress = None
    if meta.get("progress"):
        progress = JobProgress(**meta["progress"])

    # Get timestamps
    created_at = job.created_at
    started_at = job.started_at
    ended_at = job.ended_at

    # Get result or error
    result_data = None
    error_msg = None

    if job.is_finished and job.result:
        result_data = job.result
    elif job.is_failed:
        error_msg = str(job.exc_info) if job.exc_info else "Unknown error"

    return JobStatusResponse(
        job_id=job.id,
        status=job.get_status().value,  # Convert JobStatus enum to string
        title=title,
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        progress=progress,
        result=result_data,
        error=error_msg,
        position_in_queue=None  # Calculated separately for queued jobs
    )


@app.get("/api/v1/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs():
    """
    List all jobs in the queue (queued, started, finished, failed)

    Returns jobs sorted by creation time (newest first)
    """
    # Get registries
    started_registry = StartedJobRegistry(queue=podcast_queue)
    finished_registry = FinishedJobRegistry(queue=podcast_queue)
    failed_registry = FailedJobRegistry(queue=podcast_queue)

    # Get job IDs from each registry
    queued_job_ids = podcast_queue.job_ids  # Jobs in queue
    started_job_ids = started_registry.get_job_ids()
    finished_job_ids = finished_registry.get_job_ids()
    failed_job_ids = failed_registry.get_job_ids()

    # Collect all jobs
    all_job_ids = list(queued_job_ids) + list(started_job_ids) + list(finished_job_ids) + list(failed_job_ids)

    jobs_list = []
    for job_id in all_job_ids:
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            job_status = _rq_job_to_status_response(job)

            # Add position in queue for queued jobs
            if job.get_status().value == "queued":
                try:
                    job_status.position_in_queue = queued_job_ids.index(job_id) + 1
                except ValueError:
                    pass

            jobs_list.append(job_status)
        except Exception as e:
            # Jobs not found are normal (expired/cleaned from Redis but still in registry)
            # Log as DEBUG instead of WARNING to reduce noise
            error_msg = str(e)
            if "No such job" in error_msg:
                logger.debug(f"Skipping expired job {job_id}: {error_msg}")
            else:
                # Other errors (validation, etc.) are actual warnings
                logger.warning(f"Failed to fetch job {job_id}: {e}")
            continue

    # Sort by creation time (newest first)
    jobs_list.sort(key=lambda x: x.created_at or time.time(), reverse=True)

    # Count by status
    queued_count = len([j for j in jobs_list if j.status == "queued"])
    started_count = len([j for j in jobs_list if j.status == "started"])
    finished_count = len([j for j in jobs_list if j.status == "finished"])
    failed_count = len([j for j in jobs_list if j.status == "failed"])

    return JobListResponse(
        jobs=jobs_list,
        total=len(jobs_list),
        queued=queued_count,
        started=started_count,
        finished=finished_count,
        failed=failed_count
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get status of a specific job

    Args:
        job_id: RQ job ID

    Returns:
        JobStatusResponse with current progress
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return _rq_job_to_status_response(job)
    except Exception as e:
        logger.error(f"Failed to fetch job {job_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.delete("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str):
    """
    Cancel a running or queued job and cleanup temporary files

    Args:
        job_id: RQ job ID to cancel

    Returns:
        Success message with cleanup details

    Raises:
        HTTPException 404: Job not found
        HTTPException 400: Job cannot be cancelled (already finished/failed)
    """
    try:
        # Fetch job from Redis
        job = Job.fetch(job_id, connection=redis_conn)
        job_status = job.get_status().value

        logger.info(f"[{job_id}] Cancel request received (current status: {job_status})")

        # Check if job can be cancelled
        if job_status not in ["queued", "started"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be cancelled (status: {job_status}). Only queued or started jobs can be cancelled."
            )

        # Cancel the job in Redis
        job.cancel()
        logger.info(f"[{job_id}] Job marked as cancelled in Redis")

        # Cleanup temporary files
        cleanup_results = {"temp_dir_deleted": False, "cover_deleted": False}

        # Delete job temp directory
        job_dir = Path(settings.temp_dir) / job_id
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
            cleanup_results["temp_dir_deleted"] = True
            logger.info(f"[{job_id}] Cleaned up temp directory: {job_dir}")

        # Delete cover image if exists
        cover_storage_dir = Path(settings.storage_base_path) / "covers"
        cover_path = cover_storage_dir / f"{job_id}_cover.jpg"
        if cover_path.exists():
            cover_path.unlink(missing_ok=True)
            cleanup_results["cover_deleted"] = True
            logger.info(f"[{job_id}] Cleaned up cover image: {cover_path}")

        logger.success(f"[{job_id}] Job cancelled successfully")

        return {
            "success": True,
            "job_id": job_id,
            "message": f"Job {job_id} cancelled successfully",
            "cleanup": cleanup_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found: {str(e)}")


@app.post("/api/v1/create-podcast", response_model=PodcastResponse, tags=["Podcast"])
async def create_podcast(
    request: Request,
    # Text mode: JSON body with PodcastRequest
    podcast_req: PodcastRequest = None,
    # PDF mode: multipart/form-data
    pdf_file: UploadFile = File(default=None),
    metadata_json: str = Form(default=None),
    tts_options_json: str = Form(default=None),
    audio_options_json: str = Form(default=None),
    processing_options_json: str = Form(default=None),
):
    """
    Create podcast from text OR PDF

    This endpoint supports two input modes:

    **Text Mode** (application/json):
    - Provide `podcast_req` JSON with `text` field

    **PDF Mode** (multipart/form-data):
    - Upload `pdf_file` (PDF document)
    - Provide `metadata_json`, `tts_options_json`, `audio_options_json`, `processing_options_json`
    - PDF is uploaded to Gemini File API (native PDF processing with OCR)
    - Gemini analyzes document, detects language, and creates chapters
    - Voice is auto-selected based on detected language
    - Cover extraction handled separately with pypdf

    Processing steps:
    1. Smart text chunking (sentence-aware) OR PDF extraction + chaptering
    2. Parallel TTS synthesis (Kokoro)
    3. ffmpeg audio merge
    4. Metadata embedding

    Returns:
        PodcastResponse with podcast file info
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())

    # ============================================================================
    # Mode Detection: Text (JSON) vs PDF (multipart/form-data)
    # ============================================================================
    if pdf_file:
        # ========================================================================
        # PDF MODE: Extract text from PDF, analyze with Gemini, auto-select voice
        # ========================================================================
        logger.info(f"[{job_id}] PDF mode: processing {pdf_file.filename}")

        try:
            # Parse JSON options from form data
            metadata = PodcastMetadata(**json.loads(metadata_json)) if metadata_json else PodcastMetadata(title=pdf_file.filename.replace('.pdf', ''))
            tts_options = TTSOptions(**json.loads(tts_options_json)) if tts_options_json else TTSOptions()
            audio_options = AudioOptions(**json.loads(audio_options_json)) if audio_options_json else AudioOptions()
            processing_options = ProcessingOptions(**json.loads(processing_options_json)) if processing_options_json else ProcessingOptions()

            # Save uploaded PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf_path = Path(tmp_pdf.name)
                content = await pdf_file.read()
                tmp_pdf.write(content)
                tmp_pdf.flush()

            logger.info(f"[{job_id}] PDF saved to {tmp_pdf_path} ({len(content)} bytes)")

            # Step 1: Validate PDF
            logger.info(f"[{job_id}] Step 1: Validating PDF")
            try:
                validate_pdf(tmp_pdf_path)
            except PDFValidationError as e:
                tmp_pdf_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=str(e))

            # Step 2: Upload PDF to Gemini File API
            logger.info(f"[{job_id}] Step 2: Uploading PDF to Gemini File API")
            try:
                gemini_client = GeminiClient()
                uploaded_file = gemini_client.upload_pdf_file(tmp_pdf_path, pdf_file.filename)
                logger.info(f"[{job_id}] PDF uploaded successfully to Gemini (URI: {uploaded_file.uri})")

                # Extract cover image (optional, parallel to Gemini processing)
                try:
                    pdf_processor = SimplePDFProcessor()
                    cover_path = pdf_processor.extract_cover_image(tmp_pdf_path)

                    if cover_path and cover_path.exists():
                        # Copy cover to shared storage for worker access
                        cover_storage_dir = Path(settings.storage_base_path) / "covers"
                        cover_storage_dir.mkdir(parents=True, exist_ok=True)
                        cover_dest = cover_storage_dir / f"{job_id}_cover.jpg"

                        import shutil
                        shutil.copy2(cover_path, cover_dest)
                        logger.info(f"[{job_id}] 📷 PDF cover extracted and saved to {cover_dest}")

                        # Store local path in metadata (worker will handle local files)
                        metadata.cover_image_url = f"file://{cover_dest}"

                        # Clean up temporary cover
                        cover_path.unlink(missing_ok=True)
                    else:
                        logger.info(f"[{job_id}] No cover image found in PDF")
                except Exception as e:
                    # Cover extraction is optional - don't fail if it doesn't work
                    logger.warning(f"[{job_id}] Cover extraction failed (non-fatal): {e}")

            except Exception as e:
                tmp_pdf_path.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")
            finally:
                # Clean up temporary PDF
                tmp_pdf_path.unlink(missing_ok=True)

            # Step 3: Analyze PDF with Gemini (language detection + chaptering via File API)
            logger.info(f"[{job_id}] Step 3: Analyzing PDF document with Gemini File API")
            try:
                # Build metadata hint from form data
                metadata_hint = {
                    'title': metadata.title,
                    'author': metadata.author,
                    'filename': pdf_file.filename
                }

                # Analyze PDF directly (Gemini reads PDF natively - no text extraction needed)
                analysis_result = gemini_client.analyze_pdf_document(
                    pdf_file=uploaded_file,
                    metadata_hint=metadata_hint
                )
                detected_language = analysis_result['language']
                chapters = analysis_result['chapters']
                logger.info(f"[{job_id}] Gemini detected language: {detected_language}, chapters: {len(chapters)}")
            except ValueError as e:
                # Gemini API key not configured
                if "GEMINI_API_KEY" in str(e):
                    raise HTTPException(
                        status_code=501,
                        detail="PDF processing requires Gemini API key. Please configure GEMINI_API_KEY environment variable in Dokploy."
                    )
                raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
            except Exception as e:
                logger.exception(f"[{job_id}] Gemini analysis failed")
                raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")

            # Step 4: Auto-select voice based on detected language
            selected_voice = select_voice(
                language=detected_language,
                gender=tts_options.voice.split('_')[0][1] if tts_options.voice != "af_bella" else "female",  # Extract gender from voice ID
                override=None  # No manual override
            )
            logger.info(f"[{job_id}] Auto-selected voice: {selected_voice} (language={detected_language})")
            tts_options.voice = selected_voice

            # Update metadata with detected language
            metadata.language = detected_language

            # Step 5: Build WebhookPodcastRequest with chapters for multi-file output
            # Convert Gemini chapters to ChapterInfo models
            from app.api.models import ChapterInfo, WebhookPodcastRequest

            chapter_infos = [
                ChapterInfo(
                    title=ch['title'],
                    text=ch['text'],
                    start_time=None  # Will be calculated during processing
                )
                for ch in chapters
            ]

            logger.info(f"[{job_id}] Prepared {len(chapter_infos)} chapters for multi-file output")

            # Build WebhookPodcastRequest with chapters (triggers multi-file mode in worker)
            # Use dummy text (required by model validation, but not used in multi-file mode)
            podcast_req = WebhookPodcastRequest(
                text="[PDF Mode - Text replaced by chapters]",
                metadata=metadata,
                tts_options=tts_options,
                audio_options=audio_options,
                processing_options=processing_options,
                chapters=chapter_infos  # Multi-file mode: process each chapter separately
            )

            logger.info(f"[{job_id}] PDF pipeline complete, proceeding to multi-file TTS synthesis ({len(chapters)} chapters)")

            # ========================================================================
            # PDF MODE: Check async_mode and enqueue job or continue sync
            # ========================================================================
            if processing_options.async_mode:
                logger.info(f"[{job_id}] Async mode enabled for PDF - enqueuing job in Redis Queue")

                # Validate callback_url for async mode (optional for GUI - will use polling instead)
                callback_url = processing_options.callback_url
                if not callback_url:
                    # GUI doesn't provide callback_url, it will poll /api/v1/jobs instead
                    logger.info(f"[{job_id}] No callback_url provided - GUI will poll for results")

                # Enqueue job in Redis Queue (persistent, retryable)
                try:
                    rq_job = enqueue_podcast_job(
                        job_id=job_id,
                        podcast_req=podcast_req,
                        callback_url=str(callback_url) if callback_url else None
                    )
                    logger.info(f"[{job_id}] RQ Job enqueued: {rq_job.get_status()}")
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to enqueue job: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to enqueue job: {str(e)}"
                    )

                # Return immediately with job_id (GUI will poll /api/v1/jobs/{job_id})
                return PodcastResponse(
                    success=True,
                    job_id=job_id,
                    podcast=None,  # Will be available when job completes
                    processing=None,
                    message=f"Job {job_id} enqueued in Redis Queue. Use /api/v1/jobs/{job_id} to check status."
                )
            else:
                # Sync mode for PDF not yet supported (requires multi-file worker logic)
                raise HTTPException(
                    status_code=501,
                    detail="PDF processing requires async_mode=true. Please enable async mode."
                )

        except HTTPException:
            raise
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in form data: {str(e)}")
        except Exception as e:
            logger.exception(f"[{job_id}] PDF mode failed: {e}")
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    elif podcast_req is None:
        # Neither PDF nor text provided
        raise HTTPException(
            status_code=400,
            detail="Either provide 'pdf_file' (multipart/form-data) OR JSON body with 'text' field"
        )

    # ============================================================================
    # TEXT MODE: Standard text-to-podcast pipeline (existing logic)
    # ============================================================================
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
            provider=podcast_req.tts_options.provider,  # Pass provider selection
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

        # CRITICAL FIX: Create dedicated TTS client for this job to avoid shared httpx.AsyncClient deadlocks
        # When multiple jobs run concurrently, sharing app_state.tts_client causes connection pool exhaustion
        tts_client = TTSProviderManager()
        await tts_client.initialize()

        try:
            tts_results = await tts_client.synthesize_chunks_parallel(
                chunks=chunks,
                output_dir=chunks_dir,
                voice=podcast_req.tts_options.voice,
                speed=podcast_req.tts_options.speed,
                max_parallel=podcast_req.processing_options.max_parallel_tts,
                pause_between=podcast_req.tts_options.pause_between_chunks,
                provider=podcast_req.tts_options.provider,  # Pass provider selection
            )
        finally:
            # Always close the dedicated client to free resources
            await tts_client.close()

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
        # CRITICAL: Log with full traceback for debugging
        logger.exception(f"[{job_id}] ❌ ASYNC JOB FAILED - Exception caught in background task")
        logger.error(f"[{job_id}] Error type: {type(e).__name__}")
        logger.error(f"[{job_id}] Error message: {str(e)}")

        # Cleanup on error
        try:
            if podcast_req.processing_options.cleanup_on_error:
                import shutil
                job_dir = Path(settings.temp_dir) / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                    logger.info(f"[{job_id}] Cleaned up job directory")
        except Exception as cleanup_error:
            logger.error(f"[{job_id}] Cleanup failed: {cleanup_error}")

        # ALWAYS send webhook callback on failure (even if cleanup fails)
        if callback_url:
            try:
                await send_webhook_callback(
                    callback_url=str(callback_url),
                    job_id=job_id,
                    success=False,
                    error=f"{type(e).__name__}: {str(e)}",
                    callbacks=callbacks
                )
            except Exception as callback_error:
                logger.error(f"[{job_id}] Failed to send error callback: {callback_error}")
        else:
            logger.warning(f"[{job_id}] No callback_url configured - error not reported to n8n")


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
    # ASYNC MODE: Return immediately and enqueue in Redis Queue
    # ============================================================================
    if podcast_req.processing_options.async_mode:
        logger.info(f"[{job_id}] Async mode enabled - enqueuing job in Redis Queue")

        # Validate callback_url if async mode
        callback_url = podcast_req.processing_options.callback_url
        if not callback_url:
            raise HTTPException(
                status_code=400,
                detail="callback_url is required when async_mode=true"
            )

        # Enqueue job in Redis Queue (persistent, retryable)
        try:
            rq_job = enqueue_podcast_job(
                job_id=job_id,
                podcast_req=podcast_req,
                callback_url=str(callback_url)
            )
            logger.info(f"[{job_id}] RQ Job enqueued: {rq_job.get_status()}")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to enqueue job: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to enqueue job: {str(e)}"
            )

        # Return immediately with job_id
        return WebhookPodcastResponse(
            success=True,
            job_id=job_id,
            message=f"Job {job_id} enqueued in Redis Queue (persistent, 3x retry). Callback will be sent to {callback_url} when complete.",
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
            provider=podcast_req.tts_options.provider,  # Pass provider selection
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
