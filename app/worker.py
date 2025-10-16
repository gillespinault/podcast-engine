"""
Podcast Engine - RQ Worker Module
Background job processing with Redis Queue for job persistence and retry logic
"""
import time
import asyncio
from pathlib import Path
from typing import Dict, Any

from redis import Redis
from rq import Queue, Retry
from rq.job import Job
from rq import get_current_job
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

from app.config import settings, AUDIO_FORMATS
from app.api.models import ProcessingStats, WebhookPodcastRequest
from app.core.chunking import TextChunker
from app.core.tts_providers import TTSProviderManager
from app.core.audio import AudioProcessor


# Redis connection
redis_conn = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=False  # We need bytes for job data
)

# RQ Queue configuration
podcast_queue = Queue(
    name='podcast_processing',
    connection=redis_conn,
    default_timeout='30m',  # 30 minutes max per job
)


class TTSError(Exception):
    """Custom exception for TTS failures (retryable)"""
    pass


class PodcastProcessingError(Exception):
    """Custom exception for non-retryable errors"""
    pass


def get_next_episode_number(series_name: str) -> int:
    """
    Get next episode number for a podcast series using Redis atomic counter

    Args:
        series_name: Name of podcast series (e.g., "Wallabag Articles")

    Returns:
        Next episode number (1, 2, 3, ...)

    Example:
        >>> get_next_episode_number("Wallabag Articles")
        1
        >>> get_next_episode_number("Wallabag Articles")
        2
    """
    try:
        # Redis key: podcast:series:{series_name}:counter
        # Use slugified series name for key safety
        series_slug = series_name.lower().replace(" ", "_").replace("/", "_")
        redis_key = f"podcast:series:{series_slug}:counter"

        # INCR is atomic - safe for concurrent jobs
        episode_number = redis_conn.incr(redis_key)

        logger.info(f"[Series: {series_name}] Generated episode number: {episode_number}")
        return episode_number

    except Exception as e:
        logger.error(f"Failed to get episode number for series '{series_name}': {e}")
        # Fallback to timestamp-based numbering if Redis fails
        import time
        fallback_number = int(time.time() % 100000)
        logger.warning(f"Using fallback episode number: {fallback_number}")
        return fallback_number


def update_job_progress(job_id: str, current_step: int, step_name: str, progress_percent: int, estimated_time_remaining: float = None, chapters_progress: list = None):
    """
    Update job progress metadata in Redis for GUI tracking

    Args:
        job_id: RQ job ID
        current_step: Current step number (1-6)
        step_name: Human-readable step name
        progress_percent: Overall progress percentage (0-100)
        estimated_time_remaining: Estimated seconds remaining (optional)
        chapters_progress: List of chapter progress dicts (multi-file mode only)
    """
    try:
        # CRITICAL FIX: Use get_current_job() to retrieve job from RQ context
        # instead of Job.fetch() which can fail when job is actively running
        job = get_current_job(connection=redis_conn)
        if not job:
            # Fallback to Job.fetch() if not running in a worker context
            logger.warning(f"[{job_id}] No current job in context, using Job.fetch() as fallback")
            job = Job.fetch(job_id, connection=redis_conn)

        progress_data = {
            "current_step": current_step,
            "total_steps": 6,
            "step_name": step_name,
            "progress_percent": progress_percent,
            "estimated_time_remaining": estimated_time_remaining
        }

        # Add chapter-level progress if provided
        if chapters_progress is not None:
            progress_data["chapters_progress"] = chapters_progress

        job.meta["progress"] = progress_data
        job.save_meta()
        logger.debug(f"[{job.id}] Progress updated: Step {current_step}/6 ({progress_percent}%) - {step_name}")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to update progress: {e}", exc_info=True)


def update_chapter_progress(job_id: str, chapter_number: int, status: str, duration_seconds: float = None, duration_est_seconds: float = None, error: str = None):
    """
    Update progress for a specific chapter in job metadata

    Args:
        job_id: RQ job ID
        chapter_number: Chapter number (1-indexed)
        status: Chapter status ("pending", "processing", "completed", "failed")
        duration_seconds: Actual duration for completed chapters
        duration_est_seconds: Estimated duration for pending/processing chapters
        error: Error message for failed chapters
    """
    try:
        job = get_current_job(connection=redis_conn)
        if not job:
            job = Job.fetch(job_id, connection=redis_conn)

        # Get existing progress
        progress = job.meta.get("progress", {})
        chapters_progress = progress.get("chapters_progress", [])

        # Find and update chapter
        for chapter in chapters_progress:
            if chapter["number"] == chapter_number:
                chapter["status"] = status
                if duration_seconds is not None:
                    chapter["duration_seconds"] = duration_seconds
                if duration_est_seconds is not None:
                    chapter["duration_est_seconds"] = duration_est_seconds
                if error is not None:
                    chapter["error"] = error
                break

        # Update progress in job.meta
        progress["chapters_progress"] = chapters_progress
        job.meta["progress"] = progress
        job.save_meta()

        logger.debug(f"[{job_id}] Chapter {chapter_number} updated: {status}")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to update chapter {chapter_number} progress: {e}", exc_info=True)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=30, min=30, max=120),  # 30s, 60s, 120s
    retry=retry_if_exception_type(TTSError),
    reraise=True
)
async def _synthesize_with_retry(tts_client, chunks, chunks_dir, voice, speed, max_parallel, pause_between):
    """
    TTS synthesis with automatic retry on timeout/network errors

    Raises:
        TTSError: On retriable errors (network, timeout)
        PodcastProcessingError: On permanent errors
    """
    try:
        tts_results = await tts_client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=chunks_dir,
            voice=voice,
            speed=speed,
            max_parallel=max_parallel,
            pause_between=pause_between,
        )
        return tts_results
    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
        logger.warning(f"TTS network error (will retry): {e}")
        raise TTSError(f"TTS network error: {e}")
    except Exception as e:
        logger.error(f"TTS permanent error (won't retry): {e}")
        raise PodcastProcessingError(f"TTS error: {e}")


async def send_webhook_callback_with_retry(
    callback_url: str,
    job_id: str,
    success: bool,
    podcast_data: dict = None,
    processing_stats: ProcessingStats = None,
    callbacks: dict = None,
    error: str = None,
):
    """
    Send webhook callback with exponential backoff retry (3 attempts: 1s, 2s, 4s)
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

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{job_id}] Sending webhook callback (attempt {attempt}/{max_retries})")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    callback_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code in [200, 201, 202, 204]:
                    logger.success(f"[{job_id}] Webhook callback delivered (HTTP {response.status_code})")
                    return True
                else:
                    logger.warning(f"[{job_id}] Webhook returned {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.error(f"[{job_id}] Webhook attempt {attempt} failed: {e}")

        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries:
            wait_time = 2 ** (attempt - 1)
            logger.info(f"[{job_id}] Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    logger.error(f"[{job_id}] Webhook failed after {max_retries} attempts")
    return False


def process_podcast_job(
    job_id: str,
    podcast_req_dict: Dict[str, Any],
    callback_url: str = None,
    callbacks: dict = None
):
    """
    RQ Job function for podcast processing (synchronous wrapper for async code)

    This function is executed by RQ workers and provides:
    - Job persistence (survives worker crashes/restarts)
    - Automatic retry (3x with exponential backoff via tenacity)
    - Error tracking in Redis
    - Webhook callbacks on completion

    Args:
        job_id: Unique job identifier
        podcast_req_dict: WebhookPodcastRequest as dict
        callback_url: Webhook URL to notify on completion
        callbacks: Original callbacks (source_workflow_id, etc.)
    """
    # Run async code in new event loop (RQ workers are sync)
    asyncio.run(_process_podcast_job_async(
        job_id=job_id,
        podcast_req_dict=podcast_req_dict,
        callback_url=callback_url,
        callbacks=callbacks
    ))


async def _process_chapter_to_audio(
    job_id: str,
    chapter_index: int,
    chapter_info: dict,
    output_dir: Path,
    podcast_req,
    tts_client: TTSProviderManager,
    audio_processor: AudioProcessor,
    cover_image_path: Path = None
) -> dict:
    """
    Process a single chapter: chunking → TTS → merge → metadata → save as separate file

    Args:
        job_id: Job identifier for logging
        chapter_index: Chapter number (1-based)
        chapter_info: ChapterInfo dict with title and text
        output_dir: Output directory for chapter files
        podcast_req: Full podcast request with TTS/audio options
        tts_client: Shared TTS client
        audio_processor: Shared audio processor
        cover_image_path: Optional cover image

    Returns:
        dict with filename, file_path, file_size, duration_seconds
    """
    chapter_num = chapter_index + 1
    chapter_title = chapter_info['title']
    chapter_text = chapter_info['text']

    logger.info(f"[{job_id}] ⏳ Chapter {chapter_num}: {chapter_title} ({len(chapter_text)} chars)")

    # Create temp directory for this chapter
    chapter_temp_dir = Path(settings.temp_dir) / job_id / f"chapter_{chapter_num}"
    chapter_temp_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = chapter_temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Chunk chapter text
        chunker = TextChunker(
            max_chunk_size=podcast_req.tts_options.chunk_size,
            preserve_sentence=podcast_req.tts_options.preserve_sentence,
            remove_urls=podcast_req.tts_options.remove_urls,
            remove_markdown=podcast_req.tts_options.remove_markdown,
        )

        chunks = chunker.create_chunks(
            text=chapter_text,
            add_chapter_markers=False  # No chapter markers within chapters
        )

        if not chunks:
            raise PodcastProcessingError(f"Chapter {chapter_num} generated no text chunks")

        logger.info(f"[{job_id}]  └─ Generated {len(chunks)} chunks for chapter {chapter_num}")

        # Step 2: TTS synthesis
        tts_results = await _synthesize_with_retry(
            tts_client=tts_client,
            chunks=chunks,
            chunks_dir=chunks_dir,
            voice=podcast_req.tts_options.voice,
            speed=podcast_req.tts_options.speed,
            max_parallel=podcast_req.processing_options.max_parallel_tts,
            pause_between=podcast_req.tts_options.pause_between_chunks,
        )

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]
        failed_chunks = [r for r in tts_results if not r[2]]

        if failed_chunks:
            logger.warning(f"[{job_id}]  └─ {len(failed_chunks)}/{len(chunks)} chunks failed for chapter {chapter_num}")

        if not audio_files:
            raise PodcastProcessingError(f"Chapter {chapter_num} generated no audio")

        # Step 3: Merge audio files for this chapter
        # Sanitize chapter title for filename (remove special chars)
        safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '' for c in chapter_title)
        safe_title = safe_title.replace(' ', '-').lower()[:50]  # Max 50 chars

        chapter_filename = f"{chapter_num:02d}-{safe_title}.m4a"
        chapter_output_path = output_dir / chapter_filename

        merged_audio = await audio_processor.merge_audio_files(
            input_files=audio_files,
            output_path=chapter_output_path,
            format="m4a",  # Always M4A for chapters
            bitrate=podcast_req.audio_options.bitrate,
            sample_rate=podcast_req.audio_options.sample_rate,
            channels=podcast_req.audio_options.channels,
            add_silence_start=podcast_req.audio_options.add_silence_start,
            add_silence_end=podcast_req.audio_options.add_silence_end,
        )

        # Step 4: Embed metadata for this chapter
        audio_processor.embed_metadata(
            audio_path=merged_audio,
            title=f"{chapter_num}. {chapter_title}",  # "1. Introduction"
            author=podcast_req.metadata.author,
            description=f"Chapter {chapter_num} of {podcast_req.metadata.title}",
            album=podcast_req.metadata.title,  # Audiobook title as album
            genre=podcast_req.metadata.genre,
            narrator=podcast_req.metadata.narrator,
            publisher=podcast_req.metadata.publisher,
            copyright=podcast_req.metadata.copyright,
            publication_date=podcast_req.metadata.publication_date,
            cover_image_path=cover_image_path,
            track_number=chapter_num,  # Chapter number as track number
        )

        # Step 5: Get duration
        duration = await audio_processor.get_audio_duration(merged_audio)

        logger.success(f"[{job_id}] ✓ Chapter {chapter_num}: {chapter_filename} ({merged_audio.stat().st_size / 1024 / 1024:.1f} MB, {duration:.1f}s)")

        return {
            "chapter_number": chapter_num,
            "filename": chapter_filename,
            "file_path": str(merged_audio),
            "file_size": merged_audio.stat().st_size,
            "duration_seconds": duration,
            "chunks_processed": len(chunks),
            "chunks_successful": len(audio_files),
        }

    finally:
        # Cleanup chapter temp directory
        import shutil
        shutil.rmtree(chapter_temp_dir, ignore_errors=True)


async def _process_multi_file_audiobook(
    job_id: str,
    podcast_req,
    callback_url: str,
    callbacks: dict,
    start_time: float
):
    """
    Process multi-file audiobook: PDF chapters → separate M4A files

    Creates directory structure:
    /data/shared/podcasts/final/{audiobook_title}/
        01-chapter-one.m4a
        02-chapter-two.m4a
        ...

    Args:
        job_id: Job identifier
        podcast_req: WebhookPodcastRequest with chapters
        callback_url: Webhook callback URL
        callbacks: Callback metadata
        start_time: Job start timestamp
    """
    try:
        # Create audiobook directory
        audiobook_title_safe = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '' for c in podcast_req.metadata.title)
        audiobook_title_safe = audiobook_title_safe.replace(' ', '-').lower()[:100]
        audiobook_dir = Path(settings.final_dir) / audiobook_title_safe
        audiobook_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{job_id}] 📁 Audiobook directory: {audiobook_dir}")

        # Download cover image if provided (or copy from local file)
        cover_image_path = None
        if podcast_req.metadata.cover_image_url and podcast_req.audio_options.embed_cover:
            job_dir = Path(settings.temp_dir) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            cover_image_path = job_dir / "cover.jpg"

            cover_url = str(podcast_req.metadata.cover_image_url)

            # Handle local file:// URLs (from PDF extraction)
            if cover_url.startswith("file://"):
                import shutil
                source_path = Path(cover_url.replace("file://", ""))
                if source_path.exists():
                    shutil.copy2(source_path, cover_image_path)
                    logger.info(f"[{job_id}] 📷 Cover image copied from local file: {source_path}")
                else:
                    logger.warning(f"[{job_id}] Local cover file not found: {source_path}")
                    cover_image_path = None
            else:
                # HTTP/HTTPS URL - download normally
                audio_processor = AudioProcessor()
                cover_image_path = await audio_processor.download_cover_image(
                    cover_url,
                    cover_image_path
                )
                logger.info(f"[{job_id}] 📷 Cover image downloaded from URL")

        # Initialize chapters progress (all pending)
        total_chapters = len(podcast_req.chapters)
        chapters_progress = []
        for idx, chapter in enumerate(podcast_req.chapters):
            chapter_title = chapter.title if hasattr(chapter, 'title') else f"Chapter {idx+1}"
            chapters_progress.append({
                "number": idx + 1,
                "title": chapter_title,
                "status": "pending",
                "duration_est_seconds": 180  # Rough estimate: 3 minutes per chapter
            })

        # Initialize progress with chapter list
        update_job_progress(
            job_id,
            current_step=1,
            step_name="Preparing chapters",
            progress_percent=5,
            chapters_progress=chapters_progress
        )

        logger.info(f"[{job_id}] Initialized {total_chapters} chapters in progress tracker")

        # Initialize shared TTS client and audio processor
        tts_client = TTSProviderManager()
        await tts_client.initialize()
        audio_processor = AudioProcessor()

        try:
            # Process each chapter sequentially
            chapter_results = []

            for idx, chapter in enumerate(podcast_req.chapters):
                # Check if job was cancelled
                try:
                    job = Job.fetch(job_id, connection=redis_conn)
                    if job.is_canceled:
                        logger.warning(f"[{job_id}] Job was cancelled - stopping chapter processing")
                        raise PodcastProcessingError(f"Job {job_id} was cancelled by user")
                except Exception as e:
                    if "cancelled" in str(e).lower():
                        raise
                    # If we can't check cancellation status, continue (don't break existing behavior)
                    logger.debug(f"[{job_id}] Could not check cancellation status: {e}")

                # Mark chapter as processing
                chapter_num = idx + 1
                update_chapter_progress(job_id, chapter_num, "processing")

                # Update overall progress
                progress_percent = int(10 + (idx / total_chapters) * 80)  # 10-90%
                update_job_progress(
                    job_id,
                    current_step=2,
                    step_name=f"Processing chapter {idx+1}/{total_chapters}",
                    progress_percent=progress_percent
                )

                # Process chapter
                try:
                    chapter_result = await _process_chapter_to_audio(
                        job_id=job_id,
                        chapter_index=idx,
                        chapter_info=chapter.model_dump() if hasattr(chapter, 'model_dump') else chapter,
                        output_dir=audiobook_dir,
                        podcast_req=podcast_req,
                        tts_client=tts_client,
                        audio_processor=audio_processor,
                        cover_image_path=cover_image_path
                    )

                    # Mark chapter as completed
                    update_chapter_progress(
                        job_id,
                        chapter_num,
                        "completed",
                        duration_seconds=chapter_result['duration_seconds']
                    )

                    chapter_results.append(chapter_result)

                except Exception as chapter_error:
                    # Mark chapter as failed
                    logger.error(f"[{job_id}] Chapter {chapter_num} failed: {chapter_error}")
                    update_chapter_progress(
                        job_id,
                        chapter_num,
                        "failed",
                        error=str(chapter_error)
                    )
                    # Re-raise to fail the entire job (or you could continue to next chapter)
                    raise

        finally:
            # Always close TTS client
            await tts_client.close()

        # Calculate aggregate statistics
        total_duration = sum(ch['duration_seconds'] for ch in chapter_results)
        total_size = sum(ch['file_size'] for ch in chapter_results)
        total_chunks = sum(ch['chunks_processed'] for ch in chapter_results)
        successful_chunks = sum(ch['chunks_successful'] for ch in chapter_results)
        processing_time = time.time() - start_time

        logger.success(f"[{job_id}] 🎉 Multi-file audiobook complete!")
        logger.info(f"[{job_id}]  └─ {len(chapter_results)} chapters → {audiobook_dir}")
        logger.info(f"[{job_id}]  └─ Total duration: {total_duration/60:.1f} min")
        logger.info(f"[{job_id}]  └─ Total size: {total_size/1024/1024:.1f} MB")
        logger.info(f"[{job_id}]  └─ Processing time: {processing_time:.1f}s")

        # Copy to AudioBookshelf audiobooks directory (not podcasts!)
        try:
            # AudioBookshelf audiobooks directory (mounted volume)
            abs_audiobooks = Path("/audiobooks")

            if abs_audiobooks.exists():
                # Create book folder in AudioBookshelf with same name as local folder
                audiobook_name = audiobook_dir.name
                abs_dest_dir = abs_audiobooks / audiobook_name
                abs_dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy all chapter files + cover to AudioBookshelf
                import shutil
                files_copied = 0
                for chapter_file in audiobook_dir.glob("*.m4a"):
                    dest_file = abs_dest_dir / chapter_file.name
                    shutil.copy2(chapter_file, dest_file)
                    files_copied += 1

                # Copy cover if exists
                if cover_image_path and cover_image_path.exists():
                    cover_dest = abs_dest_dir / "cover.jpg"
                    shutil.copy2(cover_image_path, cover_dest)
                    logger.info(f"[{job_id}] 📷 Cover copied to AudioBookshelf")

                logger.success(f"[{job_id}] 📚 Copied to AudioBookshelf: /audiobooks/{audiobook_name}/ ({files_copied} chapters)")
                logger.info(f"[{job_id}]  └─ AudioBookshelf will display this as a multi-chapter audiobook")
            else:
                logger.warning(f"[{job_id}] AudioBookshelf audiobooks directory not found: {abs_audiobooks}")
                logger.info(f"[{job_id}] Files remain available at: {audiobook_dir}")
        except Exception as e:
            # Non-fatal: log warning but don't fail the job
            logger.warning(f"[{job_id}] Failed to copy to AudioBookshelf: {e}")
            logger.info(f"[{job_id}] Files still available at: {audiobook_dir}")

        # Update progress to 100%
        update_job_progress(job_id, 6, "Complete", 100)

        # Build response data
        stats = ProcessingStats(
            total_chunks=total_chunks,
            successful_chunks=successful_chunks,
            failed_chunks=total_chunks - successful_chunks,
            total_duration_seconds=total_duration,
            processing_time_seconds=processing_time,
            tts_api_calls=total_chunks,
            average_chunk_time=processing_time / total_chunks if total_chunks else 0,
            text_length_chars=sum(len(ch.text) for ch in podcast_req.chapters),
            text_length_words=sum(len(ch.text.split()) for ch in podcast_req.chapters),
            estimated_listening_time_minutes=total_duration / 60,
        )

        podcast_data = {
            "audiobook_title": podcast_req.metadata.title,
            "audiobook_directory": str(audiobook_dir),
            "total_chapters": len(chapter_results),
            "chapters": chapter_results,
            "total_duration_seconds": total_duration,
            "total_size_bytes": total_size,
            "format": "m4a",  # Multi-file always uses M4A
        }

        # Send webhook callback on success
        if callback_url:
            logger.info(f"[{job_id}] 📡 Sending webhook callback...")
            await send_webhook_callback_with_retry(
                callback_url=str(callback_url),
                job_id=job_id,
                success=True,
                podcast_data=podcast_data,
                processing_stats=stats,
                callbacks=callbacks
            )
            logger.success(f"[{job_id}] ✓ Webhook callback sent")

        # Cleanup temp files
        job_dir = Path(settings.temp_dir) / job_id
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)

    except Exception as e:
        logger.exception(f"[{job_id}] Multi-file processing failed")

        # Cleanup on error
        try:
            if podcast_req.processing_options.cleanup_on_error:
                import shutil
                # Cleanup temp directory
                job_dir = Path(settings.temp_dir) / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                # DO NOT delete audiobook_dir - partial chapters may be useful
        except Exception as cleanup_error:
            logger.error(f"[{job_id}] Cleanup failed: {cleanup_error}")

        # Send error callback
        if callback_url:
            try:
                logger.info(f"[{job_id}] 📡 Sending error webhook callback...")
                await send_webhook_callback_with_retry(
                    callback_url=str(callback_url),
                    job_id=job_id,
                    success=False,
                    error=f"{type(e).__name__}: {str(e)}",
                    callbacks=callbacks
                )
            except Exception as callback_error:
                logger.error(f"[{job_id}] Failed to send error callback: {callback_error}")

        # Re-raise to mark RQ job as failed
        raise


async def _process_podcast_job_async(
    job_id: str,
    podcast_req_dict: Dict[str, Any],
    callback_url: str = None,
    callbacks: dict = None
):
    """
    Async podcast processing logic (moved from main.py)

    Supports two modes:
    - Single-file mode: Standard text → single M4B (existing behavior)
    - Multi-file mode: PDF chapters → separate M4A files per chapter (Phase 4)
    """
    start_time = time.time()

    # Reconstruct WebhookPodcastRequest from dict
    podcast_req = WebhookPodcastRequest(**podcast_req_dict)

    logger.info(f"[{job_id}] RQ Worker processing: {podcast_req.metadata.title}")

    # Store podcast title in job metadata for GUI display
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        job.meta["title"] = podcast_req.metadata.title
        job.save_meta()
    except Exception as e:
        logger.warning(f"[{job_id}] Failed to store title in job meta: {e}")

    # ============================================================================
    # MODE DETECTION: Multi-file (chapters) vs Single-file (text)
    # ============================================================================
    if podcast_req.chapters and len(podcast_req.chapters) > 0:
        # ========================================================================
        # MULTI-FILE MODE: Process each chapter as separate M4A file
        # ========================================================================
        logger.info(f"[{job_id}] 📚 Multi-file mode: {len(podcast_req.chapters)} chapters detected")
        await _process_multi_file_audiobook(job_id, podcast_req, callback_url, callbacks, start_time)
        return  # Exit after multi-file processing

    # ============================================================================
    # SINGLE-FILE MODE: Standard text-to-podcast pipeline (existing behavior)
    # ============================================================================
    logger.info(f"[{job_id}] 📄 Single-file mode: Standard text-to-podcast")

    try:
        # Create job directory
        job_dir = Path(settings.temp_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Step 1: Text chunking (0-10%)
        logger.info(f"[{job_id}] ⏳ Step 1/6 (0%): Chunking text ({len(podcast_req.text)} chars)")
        update_job_progress(job_id, 1, "Chunking text", 5)

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
            raise PodcastProcessingError("No text chunks generated")

        logger.success(f"[{job_id}] ✓ Step 1/6 (10%): Generated {len(chunks)} chunks")
        update_job_progress(job_id, 1, "Text chunked", 10)

        # Check if job was cancelled before starting TTS
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            if job.is_canceled:
                logger.warning(f"[{job_id}] Job was cancelled - stopping before TTS synthesis")
                raise PodcastProcessingError(f"Job {job_id} was cancelled by user")
        except Exception as e:
            if "cancelled" in str(e).lower():
                raise
            logger.debug(f"[{job_id}] Could not check cancellation status: {e}")

        # Step 2: TTS synthesis (10-80%)
        logger.info(f"[{job_id}] ⏳ Step 2/6 (10%): TTS synthesis - {len(chunks)} chunks (max_parallel={podcast_req.processing_options.max_parallel_tts})")
        logger.info(f"[{job_id}]  └─ Estimated time: {len(chunks) * 15}s-{len(chunks) * 30}s (~20s/chunk with Kokoro TTS)")
        estimated_tts_time = len(chunks) * 20  # Rough estimate: 20s per chunk
        update_job_progress(job_id, 2, "TTS synthesis", 15, estimated_tts_time)

        tts_client = TTSProviderManager()
        await tts_client.initialize()
        try:
            # Use retry wrapper for TTS calls
            tts_results = await _synthesize_with_retry(
                tts_client=tts_client,
                chunks=chunks,
                chunks_dir=chunks_dir,
                voice=podcast_req.tts_options.voice,
                speed=podcast_req.tts_options.speed,
                max_parallel=podcast_req.processing_options.max_parallel_tts,
                pause_between=podcast_req.tts_options.pause_between_chunks,
            )
        finally:
            await tts_client.close()

        # Check for failures
        failed_chunks = [r for r in tts_results if not r[2]]
        if failed_chunks:
            logger.warning(f"[{job_id}] ⚠️  {len(failed_chunks)}/{len(chunks)} chunks failed TTS")

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]

        if not audio_files:
            raise PodcastProcessingError("No audio chunks generated")

        logger.success(f"[{job_id}] ✓ Step 2/6 (80%): TTS completed - {len(audio_files)}/{len(chunks)} chunks successful")
        update_job_progress(job_id, 2, "TTS synthesis complete", 80)

        # Step 3: Merge audio (80-90%)
        logger.info(f"[{job_id}] ⏳ Step 3/6 (80%): Merging {len(audio_files)} audio files with ffmpeg")
        update_job_progress(job_id, 3, "Merging audio files", 82)
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

        logger.success(f"[{job_id}] ✓ Step 3/6 (90%): Audio merged - {merged_audio.name} ({merged_audio.stat().st_size / 1024 / 1024:.1f} MB)")
        update_job_progress(job_id, 3, "Audio merge complete", 90)

        # Step 4: Embed metadata (90-95%)
        logger.info(f"[{job_id}] ⏳ Step 4/6 (90%): Embedding metadata (title, author, cover image)")
        update_job_progress(job_id, 4, "Embedding metadata", 91)

        cover_image_path = None
        if podcast_req.metadata.cover_image_url and podcast_req.audio_options.embed_cover:
            cover_image_path = job_dir / "cover.jpg"
            cover_image_path = await audio_processor.download_cover_image(
                str(podcast_req.metadata.cover_image_url),
                cover_image_path
            )

        # Phase 4: Podcast Series Support (Episode grouping)
        # If podcast_series is set, use it as album and auto-generate episode number
        album_value = podcast_req.metadata.publisher  # Default behavior (backward compatible)
        author_value = podcast_req.metadata.author    # Default behavior (backward compatible)
        track_number = None

        if podcast_req.metadata.podcast_series:
            # Use podcast_series as album name
            album_value = podcast_req.metadata.podcast_series
            logger.info(f"[{job_id}] Using podcast series: '{album_value}'")

            # Use consistent author for all episodes in series
            author_value = "Podcast Engine"
            logger.info(f"[{job_id}] Using series author: '{author_value}'")

            # Auto-generate episode number if not provided
            if podcast_req.metadata.episode_number is None:
                track_number = get_next_episode_number(podcast_req.metadata.podcast_series)
                logger.info(f"[{job_id}] Auto-generated episode number: {track_number}")
            else:
                track_number = podcast_req.metadata.episode_number
                logger.info(f"[{job_id}] Using provided episode number: {track_number}")

        audio_processor.embed_metadata(
            audio_path=merged_audio,
            title=podcast_req.metadata.title,
            author=author_value,              # Dynamic: "Podcast Engine" for series, or original author
            description=podcast_req.metadata.description,
            album=album_value,                # Dynamic: podcast_series or publisher
            genre=podcast_req.metadata.genre,
            narrator=podcast_req.metadata.narrator,
            publisher=podcast_req.metadata.publisher,
            copyright=podcast_req.metadata.copyright,
            publication_date=podcast_req.metadata.publication_date,
            cover_image_path=cover_image_path,
            track_number=track_number,  # New: Episode number support
        )

        logger.success(f"[{job_id}] ✓ Step 4/6 (95%): Metadata embedded successfully")
        update_job_progress(job_id, 4, "Metadata embedded", 95)

        # Step 5: Get audio duration (95-98%)
        logger.info(f"[{job_id}] ⏳ Step 5/6 (95%): Calculating audio duration")
        update_job_progress(job_id, 5, "Calculating duration", 96)
        duration = await audio_processor.get_audio_duration(merged_audio)
        logger.success(f"[{job_id}] ✓ Step 5/6 (98%): Duration: {duration:.1f}s ({duration/60:.1f} min)")
        update_job_progress(job_id, 5, "Duration calculated", 98)

        # Step 6: Cleanup and finalization (98-100%)
        logger.info(f"[{job_id}] ⏳ Step 6/6 (98%): Cleaning up temporary files")
        update_job_progress(job_id, 6, "Cleaning up", 99)
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.success(f"[{job_id}] ✓ Step 6/6 (99%): Cleanup complete")
        update_job_progress(job_id, 6, "Complete", 100)

        # Phase 4.5: Copy to Audiobookshelf if podcast_series is defined
        if podcast_req.metadata.podcast_series:
            try:
                # Slugify series name for folder safety (spaces, special chars)
                series_slug = podcast_req.metadata.podcast_series.replace(" ", "_").replace("/", "_").replace("\\", "_")

                # Audiobookshelf podcasts directory (mounted volume)
                abs_podcasts = Path("/podcasts")
                series_folder = abs_podcasts / series_slug

                # Create series folder if doesn't exist
                series_folder.mkdir(parents=True, exist_ok=True)

                # Copy M4B to Audiobookshelf (preserve timestamps)
                dest_path = series_folder / merged_audio.name
                shutil.copy2(merged_audio, dest_path)

                logger.success(f"[{job_id}] 📚 Copied to Audiobookshelf: /podcasts/{series_slug}/{merged_audio.name}")
                logger.info(f"[{job_id}]  └─ All episodes in this folder will be grouped as one podcast")
            except Exception as e:
                # Non-fatal: log warning but don't fail the job
                logger.warning(f"[{job_id}] Failed to copy to Audiobookshelf: {e}")
                logger.info(f"[{job_id}] File still available at: {merged_audio}")

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

        logger.success(f"[{job_id}] 🎉 Job completed in {processing_time:.1f}s - {output_filename} ({merged_audio.stat().st_size / 1024 / 1024:.1f} MB, {duration/60:.1f} min)")

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
            logger.info(f"[{job_id}] 📡 Sending webhook callback to n8n workflow...")
            await send_webhook_callback_with_retry(
                callback_url=str(callback_url),
                job_id=job_id,
                success=True,
                podcast_data=podcast_data,
                processing_stats=stats,
                callbacks=callbacks
            )
            logger.success(f"[{job_id}] ✓ Webhook callback sent - Processing complete!")

    except Exception as e:
        logger.exception(f"[{job_id}] RQ Job failed")
        logger.error(f"[{job_id}] Error: {type(e).__name__}: {str(e)}")

        # Cleanup on error
        try:
            if podcast_req.processing_options.cleanup_on_error:
                import shutil
                job_dir = Path(settings.temp_dir) / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"[{job_id}] Cleanup failed: {cleanup_error}")

        # Send webhook callback on failure
        if callback_url:
            try:
                logger.info(f"[{job_id}] 📡 Sending error webhook callback to n8n workflow...")
                await send_webhook_callback_with_retry(
                    callback_url=str(callback_url),
                    job_id=job_id,
                    success=False,
                    error=f"{type(e).__name__}: {str(e)}",
                    callbacks=callbacks
                )
                logger.info(f"[{job_id}] ✓ Error callback sent")
            except Exception as callback_error:
                logger.error(f"[{job_id}] Failed to send error callback: {callback_error}")

        # Re-raise to mark RQ job as failed
        raise


def enqueue_podcast_job(
    job_id: str,
    podcast_req: WebhookPodcastRequest,
    callback_url: str = None
) -> Job:
    """
    Enqueue a podcast processing job in Redis Queue

    Returns:
        RQ Job object with job_id, status, etc.
    """
    logger.info(f"[{job_id}] Enqueuing job in Redis Queue")

    job = podcast_queue.enqueue(
        process_podcast_job,
        job_id=job_id,  # Use custom job_id so Job.fetch() works correctly
        kwargs={
            'job_id': job_id,
            'podcast_req_dict': podcast_req.model_dump(),
            'callback_url': callback_url,
            'callbacks': podcast_req.callbacks.model_dump() if podcast_req.callbacks else None,
        },
        retry=Retry(max=3, interval=[30, 60, 120]),  # 3 retries with exponential backoff
        failure_ttl=86400,  # Keep failed jobs for 24h
        result_ttl=3600,  # Keep successful job results for 1h
    )

    logger.info(f"[{job_id}] Job enqueued: {job.get_status()}")
    return job
