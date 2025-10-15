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
from app.core.tts import KokoroTTSClient
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


def update_job_progress(job_id: str, current_step: int, step_name: str, progress_percent: int, estimated_time_remaining: float = None):
    """
    Update job progress metadata in Redis for GUI tracking

    Args:
        job_id: RQ job ID
        current_step: Current step number (1-6)
        step_name: Human-readable step name
        progress_percent: Overall progress percentage (0-100)
        estimated_time_remaining: Estimated seconds remaining (optional)
    """
    try:
        # CRITICAL FIX: Use get_current_job() to retrieve job from RQ context
        # instead of Job.fetch() which can fail when job is actively running
        job = get_current_job(connection=redis_conn)
        if not job:
            # Fallback to Job.fetch() if not running in a worker context
            logger.warning(f"[{job_id}] No current job in context, using Job.fetch() as fallback")
            job = Job.fetch(job_id, connection=redis_conn)

        job.meta["progress"] = {
            "current_step": current_step,
            "total_steps": 6,
            "step_name": step_name,
            "progress_percent": progress_percent,
            "estimated_time_remaining": estimated_time_remaining
        }
        job.save_meta()
        logger.debug(f"[{job.id}] Progress updated: Step {current_step}/6 ({progress_percent}%) - {step_name}")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to update progress: {e}", exc_info=True)


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


async def _process_podcast_job_async(
    job_id: str,
    podcast_req_dict: Dict[str, Any],
    callback_url: str = None,
    callbacks: dict = None
):
    """
    Async podcast processing logic (moved from main.py)
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

        # Step 2: TTS synthesis (10-80%)
        logger.info(f"[{job_id}] ⏳ Step 2/6 (10%): TTS synthesis - {len(chunks)} chunks (max_parallel={podcast_req.processing_options.max_parallel_tts})")
        logger.info(f"[{job_id}]  └─ Estimated time: {len(chunks) * 15}s-{len(chunks) * 30}s (~20s/chunk with Kokoro TTS)")
        estimated_tts_time = len(chunks) * 20  # Rough estimate: 20s per chunk
        update_job_progress(job_id, 2, "TTS synthesis", 15, estimated_tts_time)

        tts_client = KokoroTTSClient()
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
