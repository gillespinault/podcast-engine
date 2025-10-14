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
            raise PodcastProcessingError("No text chunks generated")

        logger.info(f"[{job_id}] Generated {len(chunks)} chunks")

        # Step 2: TTS synthesis (with automatic retry)
        logger.info(f"[{job_id}] Step 2: TTS synthesis (max_parallel={podcast_req.processing_options.max_parallel_tts})")

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
            logger.warning(f"[{job_id}] {len(failed_chunks)} chunks failed TTS")

        # Get successful audio files
        audio_files = [r[1] for r in tts_results if r[2]]

        if not audio_files:
            raise PodcastProcessingError("No audio chunks generated")

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

        logger.success(f"[{job_id}] RQ Job completed in {processing_time:.1f}s")

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
            await send_webhook_callback_with_retry(
                callback_url=str(callback_url),
                job_id=job_id,
                success=True,
                podcast_data=podcast_data,
                processing_stats=stats,
                callbacks=callbacks
            )

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
        job_id=job_id,
        podcast_req_dict=podcast_req.model_dump(),
        callback_url=callback_url,
        callbacks=podcast_req.callbacks.model_dump() if podcast_req.callbacks else None,
        retry=Retry(max=3, interval=[30, 60, 120]),  # 3 retries with exponential backoff
        failure_ttl=86400,  # Keep failed jobs for 24h
        result_ttl=3600,  # Keep successful job results for 1h
    )

    logger.info(f"[{job_id}] Job enqueued: {job.get_status()}")
    return job
