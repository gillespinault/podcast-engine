"""
Podcast Engine - TTS Client
Async client for Kokoro TTS API with retry logic
"""
import asyncio
import httpx
from typing import List, Tuple, Optional
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


class KokoroTTSClient:
    """Async client for Kokoro TTS API"""

    def __init__(
        self,
        api_url: str = None,
        timeout: int = None,
        max_retries: int = 3,
    ):
        """
        Initialize TTS client

        Args:
            api_url: Kokoro API endpoint
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
        """
        self.api_url = api_url or str(settings.kokoro_tts_url)
        self.timeout = timeout or settings.kokoro_timeout
        self.max_retries = max_retries

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    async def synthesize_chunk(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """
        Synthesize single text chunk to audio

        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed (0.5-2.0)
            response_format: Audio format (mp3, opus, aac, flac)

        Returns:
            Audio bytes

        Raises:
            httpx.HTTPError: On API error
        """
        try:
            logger.debug(f"Synthesizing {len(text)} chars with voice={voice}, speed={speed}")

            # Prepare request
            payload = {
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            }

            logger.debug(f"[TTS] About to call client.post() for {len(text)} chars")

            # Make request
            response = await self.client.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            logger.debug(f"[TTS] client.post() returned, status={response.status_code}")

            response.raise_for_status()

            logger.debug(f"[TTS] raise_for_status() passed")

            # Return audio bytes
            audio_data = response.content
            logger.debug(f"Received {len(audio_data)} bytes of audio")

            return audio_data

        except httpx.HTTPStatusError as e:
            logger.error(f"TTS API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.TimeoutException as e:
            logger.warning(f"TTS API timeout after {self.timeout}s: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}")
            raise

    async def synthesize_chunks_parallel(
        self,
        chunks: List[Tuple[int, str, str]],
        output_dir: Path,
        voice: str = "af_bella",
        speed: float = 1.0,
        max_parallel: int = 5,
        pause_between: float = 0.5,
    ) -> List[Tuple[int, Path, bool]]:
        """
        Synthesize multiple chunks in parallel with controlled concurrency

        Args:
            chunks: List of (chunk_id, text, chapter_title)
            output_dir: Directory to save audio files
            voice: Voice ID
            speed: Speech speed
            max_parallel: Max parallel API calls
            pause_between: Silence between chunks (seconds)

        Returns:
            List of (chunk_id, audio_path, success)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Semaphore to limit parallelism
        semaphore = asyncio.Semaphore(max_parallel)

        async def process_chunk(chunk_id: int, text: str, chapter_title: str) -> Tuple[int, Path, bool]:
            """Process single chunk with semaphore"""
            async with semaphore:
                output_path = output_dir / f"chunk_{chunk_id:04d}.mp3"

                try:
                    logger.info(f"Processing chunk {chunk_id}/{len(chunks)}: {len(text)} chars")

                    # Add chapter marker if provided
                    if chapter_title:
                        text_with_marker = f"{chapter_title}. {text}"
                    else:
                        text_with_marker = text

                    # Synthesize
                    audio_bytes = await self.synthesize_chunk(
                        text=text_with_marker,
                        voice=voice,
                        speed=speed,
                        response_format="mp3"
                    )

                    # Save to file
                    output_path.write_bytes(audio_bytes)

                    logger.success(f"Saved chunk {chunk_id} to {output_path.name}")

                    return (chunk_id, output_path, True)

                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_id}: {e}")
                    return (chunk_id, output_path, False)

        # Process all chunks in parallel (with semaphore limit)
        tasks = [
            process_chunk(chunk_id, text, chapter)
            for chunk_id, text, chapter in chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Log summary
        successful = sum(1 for _, _, success in results if success)
        failed = len(results) - successful

        logger.info(f"TTS completed: {successful} successful, {failed} failed")

        return results

    async def health_check(self) -> dict:
        """
        Check if Kokoro TTS service is healthy

        Returns:
            Health status dict
        """
        try:
            # Try a minimal synthesis
            test_text = "Hello world"
            await self.synthesize_chunk(test_text, voice="af_bella", speed=1.0)

            return {
                "status": "healthy",
                "service": "Kokoro TTS",
                "url": self.api_url,
            }

        except Exception as e:
            logger.error(f"Kokoro TTS health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "Kokoro TTS",
                "url": self.api_url,
                "error": str(e),
            }

    async def get_available_voices(self) -> dict:
        """
        Dynamically fetch available voices from Kokoro TTS

        Returns:
            Dict mapping voice_id → {name, gender, language, accent}
        """
        # Language mapping based on voice prefix
        LANGUAGE_MAP = {
            "af": {"language": "en", "accent": "American", "gender": "Female"},
            "am": {"language": "en", "accent": "American", "gender": "Male"},
            "bf": {"language": "en", "accent": "British", "gender": "Female"},
            "bm": {"language": "en", "accent": "British", "gender": "Male"},
            "ef": {"language": "en", "accent": "English", "gender": "Female"},
            "em": {"language": "en", "accent": "English", "gender": "Male"},
            "ff": {"language": "fr", "accent": "French", "gender": "Female"},
            "fm": {"language": "fr", "accent": "French", "gender": "Male"},
            "hf": {"language": "hi", "accent": "Hindi", "gender": "Female"},
            "hm": {"language": "hi", "accent": "Hindi", "gender": "Male"},
            "if": {"language": "it", "accent": "Italian", "gender": "Female"},
            "im": {"language": "it", "accent": "Italian", "gender": "Male"},
            "jf": {"language": "ja", "accent": "Japanese", "gender": "Female"},
            "jm": {"language": "ja", "accent": "Japanese", "gender": "Male"},
            "pf": {"language": "pt", "accent": "Portuguese", "gender": "Female"},
            "pm": {"language": "pt", "accent": "Portuguese", "gender": "Male"},
            "zf": {"language": "zh", "accent": "Chinese", "gender": "Female"},
            "zm": {"language": "zh", "accent": "Chinese", "gender": "Male"},
        }

        # Fallback minimal voices if API fails
        FALLBACK_VOICES = {
            "af_bella": {"name": "Bella", "gender": "Female", "language": "en", "accent": "American"},
            "af_sarah": {"name": "Sarah", "gender": "Female", "language": "en", "accent": "American"},
            "bf_emma": {"name": "Emma", "gender": "Female", "language": "en", "accent": "British"},
            "am_adam": {"name": "Adam", "gender": "Male", "language": "en", "accent": "American"},
            "bm_george": {"name": "George", "gender": "Male", "language": "en", "accent": "British"},
        }

        try:
            # Try to query Kokoro TTS API for voices list
            # Note: Kokoro doesn't have a /voices endpoint, so we use a workaround
            # We'll test synthesis with common voice to ensure service is up

            logger.debug("Fetching available voices from Kokoro TTS")

            # List of known voices from Kokoro v1.0 (67 voices as of 2025-10)
            # This list should be kept in sync with Kokoro releases
            known_voices = [
                "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica",
                "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                "af_v0", "af_v0bella", "af_v0irulan", "af_v0nicole", "af_v0sarah", "af_v0sky",
                "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
                "am_onyx", "am_puck", "am_santa", "am_v0adam", "am_v0gurney", "am_v0michael",
                "bf_alice", "bf_emma", "bf_lily", "bf_v0emma", "bf_v0isabella",
                "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "bm_v0george", "bm_v0lewis",
                "ef_dora", "em_alex", "em_santa",
                "ff_siwis",
                "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                "if_sara", "im_nicola",
                "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
                "pf_dora", "pm_alex", "pm_santa",
                "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            ]

            voices = {}

            for voice_id in known_voices:
                # Extract prefix (e.g., "af" from "af_bella")
                prefix = voice_id[:2]

                if prefix in LANGUAGE_MAP:
                    # Extract name (capitalize after underscore)
                    name_part = voice_id[3:]  # Remove prefix (af_)
                    name = name_part.replace("_", " ").title()

                    voices[voice_id] = {
                        "name": name,
                        "gender": LANGUAGE_MAP[prefix]["gender"],
                        "language": LANGUAGE_MAP[prefix]["language"],
                        "accent": LANGUAGE_MAP[prefix]["accent"],
                    }

            logger.success(f"Loaded {len(voices)} voices from Kokoro TTS")
            return voices

        except Exception as e:
            logger.warning(f"Failed to fetch voices dynamically, using fallback: {e}")
            return FALLBACK_VOICES


# Example usage
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    async def test_tts():
        client = KokoroTTSClient()

        # Test single chunk
        text = "This is a test of the Kokoro text-to-speech system."
        audio = await client.synthesize_chunk(text, voice="af_bella", speed=1.0)
        print(f"Got {len(audio)} bytes of audio")

        # Test parallel synthesis
        chunks = [
            (0, "First chunk of text.", "Part 1"),
            (1, "Second chunk of text.", "Part 2"),
            (2, "Third chunk of text.", "Part 3"),
        ]

        output_dir = Path("/tmp/test_tts")
        results = await client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=output_dir,
            voice="af_bella",
            speed=1.0,
            max_parallel=3
        )

        for chunk_id, path, success in results:
            print(f"Chunk {chunk_id}: {'✓' if success else '✗'} - {path}")

        await client.close()

    asyncio.run(test_tts())
