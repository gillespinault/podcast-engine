"""
Podcast Engine - Test Configuration
Pytest fixtures and test utilities
"""
import asyncio
import pytest
from pathlib import Path
from typing import AsyncGenerator, Generator

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


# ============================================================================
# Pytest Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Client Fixtures
# ============================================================================

@pytest.fixture
def client(mock_kokoro_tts) -> TestClient:
    """
    FastAPI test client with mocked app.state

    Initializes app.state attributes normally set in lifespan()
    """
    # Initialize app state (normally done in lifespan contextmanager)
    import time
    from app.core.tts import KokoroTTSClient

    app.state.start_time = time.time()
    app.state.tts_client = KokoroTTSClient()
    app.state.available_voices = {
        "af_bella": {"name": "Bella", "gender": "Female", "language": "en", "accent": "American"},
        "ff_siwis": {"name": "Siwis", "gender": "Female", "language": "fr", "accent": "French"},
        "jf_alpha": {"name": "Alpha", "gender": "Female", "language": "ja", "accent": "Japanese"},
    }

    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator:
    """Async HTTP client for testing API"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ============================================================================
# Mock TTS Client
# ============================================================================

@pytest.fixture
def mock_tts_audio() -> bytes:
    """Fake MP3 audio data for testing"""
    # Minimal valid MP3 header + fake data
    return b"\xff\xfb\x90\x00" + b"\x00" * 1024


@pytest.fixture
def mock_kokoro_tts(monkeypatch, mock_tts_audio):
    """Mock Kokoro TTS API calls"""

    async def fake_synthesize_chunk(*args, **kwargs):
        """Return fake audio data"""
        await asyncio.sleep(0.01)  # Simulate network delay
        return mock_tts_audio

    async def fake_get_available_voices(self):
        """Return minimal voice list"""
        return {
            "af_bella": {"name": "Bella", "gender": "Female", "language": "en", "accent": "American"},
            "ff_siwis": {"name": "Siwis", "gender": "Female", "language": "fr", "accent": "French"},
            "jf_alpha": {"name": "Alpha", "gender": "Female", "language": "ja", "accent": "Japanese"},
        }

    async def fake_health_check(self):
        """Return healthy status"""
        return {"status": "healthy", "service": "Kokoro TTS"}

    # Patch methods
    monkeypatch.setattr(
        "app.core.tts.KokoroTTSClient.synthesize_chunk",
        fake_synthesize_chunk
    )
    monkeypatch.setattr(
        "app.core.tts.KokoroTTSClient.get_available_voices",
        fake_get_available_voices
    )
    monkeypatch.setattr(
        "app.core.tts.KokoroTTSClient.health_check",
        fake_health_check
    )

    return fake_synthesize_chunk


# ============================================================================
# Mock Audio Processing
# ============================================================================

@pytest.fixture
def mock_ffmpeg(monkeypatch, tmp_path):
    """Mock ffmpeg audio merging"""

    async def fake_merge_audio_files(self, input_files, output_path, **kwargs):
        """Create fake merged audio file"""
        output_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 5000)
        return output_path

    async def fake_get_audio_duration(self, audio_path):
        """Return fake duration"""
        return 42.5  # seconds

    def fake_embed_metadata(self, audio_path, **kwargs):
        """No-op metadata embedding"""
        pass

    async def fake_download_cover_image(self, url, output_path):
        """Create fake cover image"""
        output_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 1000)  # JPEG header
        return output_path

    # Patch methods
    monkeypatch.setattr(
        "app.core.audio.AudioProcessor.merge_audio_files",
        fake_merge_audio_files
    )
    monkeypatch.setattr(
        "app.core.audio.AudioProcessor.get_audio_duration",
        fake_get_audio_duration
    )
    monkeypatch.setattr(
        "app.core.audio.AudioProcessor.embed_metadata",
        fake_embed_metadata
    )
    monkeypatch.setattr(
        "app.core.audio.AudioProcessor.download_cover_image",
        fake_download_cover_image
    )


# ============================================================================
# Temporary Directories
# ============================================================================

@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    """Create temporary storage directories"""
    storage = tmp_path / "podcasts"
    jobs = storage / "jobs"
    final = storage / "final"
    chunks = storage / "chunks"

    jobs.mkdir(parents=True)
    final.mkdir(parents=True)
    chunks.mkdir(parents=True)

    # Override settings
    monkeypatch.setattr(settings, "storage_base_path", str(storage))
    monkeypatch.setattr(settings, "temp_dir", str(jobs))
    monkeypatch.setattr(settings, "final_dir", str(final))

    return {
        "base": storage,
        "jobs": jobs,
        "final": final,
        "chunks": chunks,
    }


# ============================================================================
# Sample Data
# ============================================================================

@pytest.fixture
def sample_text_short() -> str:
    """Short text for testing (100 chars minimum)"""
    return "This is a test of the Kokoro text-to-speech system. " * 3


@pytest.fixture
def sample_text_long() -> str:
    """Long text for chunking tests"""
    return """
    Chapter 1: Introduction

    This is the beginning of a long story. It has many sentences. Each sentence is important.
    The story continues with more details. There are paragraphs and structure.

    Chapter 2: Development

    The plot thickens! New characters appear. Dramatic events unfold.
    The tension rises as we approach the climax of the narrative.

    Chapter 3: Conclusion

    Finally, everything comes together. The story reaches its satisfying end.
    All questions are answered. The journey is complete.
    """ * 5


@pytest.fixture
def sample_podcast_request() -> dict:
    """Sample valid podcast creation request"""
    return {
        "text": "This is a test podcast. It contains multiple sentences. " * 10,
        "metadata": {
            "title": "Test Podcast",
            "author": "Test Author",
            "description": "A test podcast for automated testing",
            "language": "en",
            "genre": "Testing",
        },
        "tts_options": {
            "voice": "af_bella",
            "speed": 1.0,
            "chunk_size": 2000,
        },
        "audio_options": {
            "format": "mp3",
            "bitrate": "64k",
        },
        "processing_options": {
            "max_parallel_tts": 3,
        }
    }


# ============================================================================
# Utility Functions
# ============================================================================

def assert_valid_mp3(file_path: Path):
    """Assert file is a valid MP3 (basic check)"""
    assert file_path.exists(), "Audio file does not exist"
    assert file_path.stat().st_size > 0, "Audio file is empty"

    # Check MP3 header
    header = file_path.read_bytes()[:4]
    assert header[:2] in [b"\xff\xfb", b"\xff\xf3"], "Invalid MP3 header"


def assert_valid_json_response(response):
    """Assert response is valid JSON with expected structure"""
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert isinstance(data, dict), "Response is not a JSON object"
    return data
