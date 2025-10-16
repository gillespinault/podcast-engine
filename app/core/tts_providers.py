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

        # Map Kokoro/common voices to Google Cloud TTS voices
        self.voice_map = {
            # French voices (Kokoro ff_* → Google fr-FR-Neural2-*)
            "ff_siwis": "fr-FR-Neural2-A",  # French female
            "french": "fr-FR-Neural2-A",
            "ff_bella": "fr-FR-Neural2-C",
            "ff_sarah": "fr-FR-Neural2-E",
            # English voices (Kokoro af_* → Google en-US-Neural2-*)
            "af_bella": "en-US-Neural2-F",  # American female
            "af_sarah": "en-US-Neural2-H",
            "am_adam": "en-US-Neural2-A",   # American male
            "am_michael": "en-US-Neural2-D",
            # Default fallback
            "default": "fr-FR-Neural2-A",
        }

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

    async def close(self):
        """Close Google Cloud TTS client (uses sync close(), not aclose())"""
        if self.client:
            # Google Cloud TTS async client uses close() not aclose()
            self.client.close()
            logger.debug(f"[{self.name}] Client closed successfully")

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

            # Map Kokoro voice to Google voice, or use configured default
            if voice and voice in self.voice_map:
                voice_name = self.voice_map[voice]
                logger.debug(f"[{self.name}] Mapped voice '{voice}' → '{voice_name}'")
            elif voice:
                # Voice not in map - try to use it directly (might be a native Google voice)
                voice_name = voice
                logger.debug(f"[{self.name}] Using unmapped voice '{voice}' (native Google voice?)")
            else:
                # No voice specified - use configured default
                voice_name = self.voice_name
                logger.debug(f"[{self.name}] Using default voice '{voice_name}'")

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


class KokoroTTSAdapter(BaseTTSClient):
    """
    Adapter for legacy KokoroTTSClient to work with multi-provider system.

    Wraps the original KokoroTTSClient and adapts it to BaseTTSClient interface.
    """

    def __init__(self):
        super().__init__(name="Kokoro TTS", timeout=settings.kokoro_timeout)
        self._kokoro_client = None

    async def initialize(self):
        """Initialize Kokoro TTS client (lazy initialization)"""
        from app.core.tts import KokoroTTSClient

        self._kokoro_client = KokoroTTSClient(
            api_url=str(settings.kokoro_tts_url),
            timeout=settings.kokoro_timeout
        )
        logger.info(f"[{self.name}] Initialized with endpoint: {settings.kokoro_tts_url}")

    async def close(self):
        """Close Kokoro client"""
        if self._kokoro_client:
            await self._kokoro_client.close()

    async def synthesize_chunk(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesize using Kokoro TTS (delegated to original client)"""
        if not self._kokoro_client:
            raise RuntimeError(f"{self.name} not initialized - call initialize() first")

        return await self._kokoro_client.synthesize_chunk(
            text=text,
            voice=voice,
            speed=speed,
            response_format=response_format
        )


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

        # Initialize providers (no imports needed - all providers are in this file)
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
                self.providers.append(KokoroTTSAdapter())
                self.provider_names.append("kokoro")
                logger.info("[TTSProviderManager] Added Kokoro TTS (via adapter)")

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

    async def get_provider_by_id(self, provider_id: str) -> Optional[BaseTTSClient]:
        """
        Get a specific provider by ID (google, piper, kokoro).

        Args:
            provider_id: Provider identifier (google, piper, kokoro)

        Returns:
            Provider instance if found, None otherwise
        """
        provider_id = provider_id.lower()
        for provider in self.providers:
            # Extract provider ID from name (e.g., "Google Cloud TTS" → "google")
            provider_name = provider.name.lower().split()[0]
            if provider_name == provider_id:
                return provider
        return None

    async def synthesize_chunk(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        response_format: str = "mp3",
        provider: Optional[str] = None,
    ) -> bytes:
        """
        Synthesize audio with automatic provider fallback (or explicit provider selection).

        Args:
            text: Text to synthesize
            voice: Voice ID (provider-specific or cross-provider)
            speed: Speech speed (0.5-2.0)
            response_format: Audio format (mp3, opus, etc.)
            provider: Optional explicit provider ID (google, piper, kokoro).
                     If specified, uses only that provider (no fallback).
                     If None, uses automatic fallback through all providers.

        Returns:
            Audio bytes

        Raises:
            ValueError: If explicit provider is specified but not found
            RuntimeError: If all providers fail (automatic mode) or explicit provider fails
        """
        # Explicit provider selection (no fallback)
        if provider:
            provider_client = await self.get_provider_by_id(provider)
            if not provider_client:
                available_providers = ", ".join([p.name.lower().split()[0] for p in self.providers])
                raise ValueError(
                    f"Provider '{provider}' not found or not initialized. "
                    f"Available providers: {available_providers}"
                )

            logger.info(f"[TTSProviderManager] Using explicit provider: {provider_client.name}")
            try:
                audio_data = await provider_client.synthesize_chunk(
                    text=text,
                    voice=voice,
                    speed=speed,
                    response_format=response_format
                )
                logger.success(f"[TTSProviderManager] Synthesis successful with {provider_client.name}")
                return audio_data
            except Exception as e:
                logger.error(f"[TTSProviderManager] Explicit provider {provider_client.name} failed: {e}")
                raise RuntimeError(f"Provider '{provider}' failed: {e}")

        # Automatic fallback mode (try all providers in order)
        last_error = None
        for idx, provider_client in enumerate(self.providers):
            try:
                logger.debug(f"[TTSProviderManager] Trying provider {idx+1}/{len(self.providers)}: {provider_client.name}")

                audio_data = await provider_client.synthesize_chunk(
                    text=text,
                    voice=voice,
                    speed=speed,
                    response_format=response_format
                )

                # Success! Update current provider if changed
                if provider_client != self.current_provider:
                    logger.info(f"[TTSProviderManager] Switched to {provider_client.name} (fallback success)")
                    self.current_provider = provider_client

                logger.success(f"[TTSProviderManager] Synthesis successful with {provider_client.name}")
                return audio_data

            except Exception as e:
                logger.warning(f"[TTSProviderManager] {provider_client.name} failed: {e}")
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
        provider: Optional[str] = None,
    ) -> List[Tuple[int, Path, bool]]:
        """
        Synthesize multiple chunks in parallel with controlled concurrency.

        Args:
            chunks: List of (chunk_id, text, chapter_title) tuples
            output_dir: Directory to save audio chunks
            voice: Voice ID (provider-specific or cross-provider)
            speed: Speech speed (0.5-2.0)
            max_parallel: Maximum parallel requests
            pause_between: Silence between chunks (seconds)
            provider: Optional explicit provider ID (google, piper, kokoro).
                     If specified, uses only that provider for all chunks.
                     If None, uses automatic fallback through all providers.

        Returns:
            List of (chunk_id, output_path, success) tuples
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

                    # Synthesize with explicit provider or automatic fallback
                    audio_bytes = await self.synthesize_chunk(
                        text=text_with_marker,
                        voice=voice,
                        speed=speed,
                        response_format="mp3",
                        provider=provider  # Pass provider parameter through
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
