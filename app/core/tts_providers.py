"""
Podcast Engine - Multi-Provider TTS System
Supports Google Cloud TTS, Piper TTS (openedai-speech), and Kokoro TTS
with automatic fallback on failures
"""
import asyncio
import httpx
import base64
import json
from typing import List, Tuple, Optional
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


class BaseTTSClient:
    """Base class for TTS clients"""

    def __init__(self, name: str, timeout: int = 60):
        self.name = name
        self.timeout = timeout
        self.client = None

    async def initialize(self):
        """Initialize HTTP client (if needed)"""
        pass

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()

    async def synthesize_chunk(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesize single text chunk to audio - must be implemented by subclass"""
        raise NotImplementedError

    async def health_check(self) -> dict:
        """Check if TTS service is healthy"""
        try:
            test_text = "Bonjour"
            await self.synthesize_chunk(test_text, voice="", speed=1.0)
            return {"status": "healthy", "provider": self.name}
        except Exception as e:
            logger.error(f"[{self.name}] Health check failed: {e}")
            return {"status": "unhealthy", "provider": self.name, "error": str(e)}


class GoogleCloudTTSClient(BaseTTSClient):
    """Google Cloud Text-to-Speech client

    Uses Google Cloud TTS API with Neural2 voices.
    Free tier: 1M characters/month
    Pricing: $16/1M characters after free tier
    """

    def __init__(self):
        super().__init__(name="Google Cloud TTS", timeout=settings.google_tts_timeout)
        self.project_id = settings.google_cloud_project_id
        self.voice_name = settings.google_tts_voice
        self.api_endpoint = f"https://texttospeech.googleapis.com/v1/text:synthesize"

    async def initialize(self):
        """Initialize Google Cloud TTS client"""
        # Import google-cloud-texttospeech only when needed
        try:
            from google.cloud import texttospeech
            from google.oauth2 import service_account
            import google.auth

            # Try to load credentials from env var (base64 encoded JSON)
            if settings.google_cloud_credentials_json:
                try:
                    creds_json = base64.b64decode(settings.google_cloud_credentials_json).decode('utf-8')
                    creds_dict = json.loads(creds_json)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self.client = texttospeech.TextToSpeechAsyncClient(credentials=credentials)
                    logger.info(f"[{self.name}] Initialized with service account credentials")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to load credentials from env var: {e}")
                    # Fallback to default credentials
                    self.client = texttospeech.TextToSpeechAsyncClient()
                    logger.info(f"[{self.name}] Initialized with default credentials")
            else:
                # Use default credentials (ADC - Application Default Credentials)
                self.client = texttospeech.TextToSpeechAsyncClient()
                logger.info(f"[{self.name}] Initialized with default credentials (ADC)")

        except ImportError:
            logger.error(f"[{self.name}] google-cloud-texttospeech not installed. Install with: pip install google-cloud-texttospeech")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def synthesize_chunk(
        self,
        text: str,
        voice: str = None,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesize text using Google Cloud TTS"""
        try:
            from google.cloud import texttospeech

            # Use configured voice or default French Neural2
            voice_name = voice if voice else self.voice_name

            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Voice configuration
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="fr-FR",  # French
                name=voice_name,
            )

            # Audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speed,
            )

            # Perform TTS request
            logger.debug(f"[{self.name}] Synthesizing {len(text)} chars with voice={voice_name}, speed={speed}")
            response = await self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
                timeout=self.timeout
            )

            audio_data = response.audio_content
            logger.debug(f"[{self.name}] Received {len(audio_data)} bytes of audio")

            return audio_data

        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            raise


class PiperTTSClient(BaseTTSClient):
    """Piper TTS client via openedai-speech

    Self-hosted TTS using Piper engine.
    Compatible with OpenAI API format.
    CPU-optimized, fast inference (<1s latency).
    """

    def __init__(self):
        super().__init__(name="Piper TTS", timeout=settings.piper_timeout)
        self.api_url = str(settings.piper_tts_url)
        self.voice_map = {
            # Map Kokoro voices to Piper voices
            "ff_siwis": "fr_FR-siwis-medium",
            "french": "fr_FR-siwis-medium",
            "default": "fr_FR-siwis-medium",
        }

    async def initialize(self):
        """Initialize HTTP client for Piper TTS"""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        logger.info(f"[{self.name}] Initialized with endpoint: {self.api_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    async def synthesize_chunk(
        self,
        text: str,
        voice: str = None,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesize text using Piper TTS (OpenAI-compatible API)"""
        try:
            # Map voice to Piper voice name
            piper_voice = self.voice_map.get(voice, self.voice_map["default"])

            logger.debug(f"[{self.name}] Synthesizing {len(text)} chars with voice={piper_voice}, speed={speed}")

            # OpenAI-compatible request format
            payload = {
                "model": "tts-1",  # openedai-speech uses tts-1 for Piper
                "input": text,
                "voice": piper_voice,
                "response_format": response_format,
                "speed": speed,
            }

            # Make request
            response = await self.client.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()

            # Return audio bytes
            audio_data = response.content
            logger.debug(f"[{self.name}] Received {len(audio_data)} bytes of audio")

            return audio_data

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.name}] API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.TimeoutException as e:
            logger.warning(f"[{self.name}] API timeout after {self.timeout}s: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            raise


class TTSProviderManager:
    """
    Multi-provider TTS manager with automatic fallback.

    Tries providers in order specified by TTS_PROVIDER config:
    1. Google Cloud TTS (if configured)
    2. Piper TTS (self-hosted, stable)
    3. Kokoro TTS (self-hosted, unstable fallback)
    """

    def __init__(self):
        self.providers: List[BaseTTSClient] = []
        self.provider_names: List[str] = []
        self.current_provider: Optional[BaseTTSClient] = None

        # Parse provider priority from config
        provider_order = [p.strip().lower() for p in settings.tts_provider.split(",")]
        logger.info(f"[TTSProviderManager] Provider priority: {provider_order}")

        # Initialize providers
        from app.core.tts import KokoroTTSClient

        for provider_name in provider_order:
            if provider_name == "google":
                if settings.google_cloud_project_id:
                    self.providers.append(GoogleCloudTTSClient())
                    self.provider_names.append("google")
                    logger.info("[TTSProviderManager] Added Google Cloud TTS")
                else:
                    logger.warning("[TTSProviderManager] Google Cloud TTS skipped (no project_id configured)")

            elif provider_name == "piper":
                self.providers.append(PiperTTSClient())
                self.provider_names.append("piper")
                logger.info("[TTSProviderManager] Added Piper TTS")

            elif provider_name == "kokoro":
                self.providers.append(KokoroTTSClient())
                self.provider_names.append("kokoro")
                logger.info("[TTSProviderManager] Added Kokoro TTS")

            else:
                logger.warning(f"[TTSProviderManager] Unknown provider '{provider_name}' - skipping")

        if not self.providers:
            raise ValueError("No TTS providers configured! Check TTS_PROVIDER env var")

        logger.success(f"[TTSProviderManager] Initialized with {len(self.providers)} providers: {self.provider_names}")

    async def initialize(self):
        """Initialize all providers"""
        for provider in self.providers:
            try:
                await provider.initialize()
                logger.success(f"[TTSProviderManager] {provider.name} initialized successfully")
            except Exception as e:
                logger.error(f"[TTSProviderManager] Failed to initialize {provider.name}: {e}")
                # Remove failed provider from list
                self.providers.remove(provider)
                self.provider_names.remove(provider.name.lower().split()[0])

        if not self.providers:
            raise RuntimeError("All TTS providers failed to initialize!")

        # Set first provider as current
        self.current_provider = self.providers[0]
        logger.info(f"[TTSProviderManager] Current provider: {self.current_provider.name}")

    async def close(self):
        """Close all provider connections"""
        for provider in self.providers:
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"[TTSProviderManager] Failed to close {provider.name}: {e}")

    async def synthesize_chunk(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """
        Synthesize audio with automatic provider fallback.

        Tries each provider in order until one succeeds.
        Raises exception only if all providers fail.
        """
        last_error = None

        for idx, provider in enumerate(self.providers):
            try:
                logger.debug(f"[TTSProviderManager] Trying provider {idx+1}/{len(self.providers)}: {provider.name}")

                audio_data = await provider.synthesize_chunk(
                    text=text,
                    voice=voice,
                    speed=speed,
                    response_format=response_format
                )

                # Success! Update current provider if changed
                if provider != self.current_provider:
                    logger.info(f"[TTSProviderManager] Switched to {provider.name} (fallback success)")
                    self.current_provider = provider

                logger.success(f"[TTSProviderManager] Synthesis successful with {provider.name}")
                return audio_data

            except Exception as e:
                logger.warning(f"[TTSProviderManager] {provider.name} failed: {e}")
                last_error = e

                # Try next provider
                if idx < len(self.providers) - 1:
                    logger.info(f"[TTSProviderManager] Falling back to next provider...")
                    continue
                else:
                    logger.error(f"[TTSProviderManager] All {len(self.providers)} providers failed!")
                    raise RuntimeError(f"All TTS providers failed. Last error: {last_error}")

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
        Synthesize multiple chunks in parallel with controlled concurrency.
        Uses current provider with fallback support per chunk.
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

                    # Synthesize with automatic fallback
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
        """Check health of all providers"""
        results = {}
        for provider in self.providers:
            health = await provider.health_check()
            results[provider.name] = health
        return results
