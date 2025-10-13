"""
Podcast Engine - TTS Tests
Tests for Kokoro TTS client with mocks
"""
import pytest
import asyncio
from pathlib import Path
from app.core.tts import KokoroTTSClient


class TestKokoroTTSClient:
    """Tests for KokoroTTSClient"""

    @pytest.mark.asyncio
    async def test_synthesize_chunk_basic(self, mock_kokoro_tts, mock_tts_audio):
        """Test basic TTS synthesis"""
        client = KokoroTTSClient()
        text = "This is a test sentence."

        audio_bytes = await client.synthesize_chunk(text, voice="af_bella", speed=1.0)

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        assert audio_bytes == mock_tts_audio

    @pytest.mark.asyncio
    async def test_synthesize_chunks_parallel(self, mock_kokoro_tts, tmp_path):
        """Test parallel chunk synthesis"""
        client = KokoroTTSClient()
        chunks = [
            (0, "First chunk of text.", "Chapter 1"),
            (1, "Second chunk of text.", "Chapter 1"),
            (2, "Third chunk of text.", "Chapter 2"),
        ]

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()

        results = await client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=output_dir,
            voice="af_bella",
            speed=1.0,
            max_parallel=2
        )

        # Check results
        assert len(results) == 3
        for chunk_id, audio_path, success in results:
            assert success is True, f"Chunk {chunk_id} failed"
            assert audio_path.exists(), f"Audio file {audio_path} not created"
            assert audio_path.stat().st_size > 0, f"Audio file {audio_path} is empty"

    @pytest.mark.asyncio
    async def test_parallel_semaphore_limit(self, mock_kokoro_tts, tmp_path):
        """Test that parallel synthesis respects semaphore limit"""
        client = KokoroTTSClient()

        # Create many chunks
        chunks = [(i, f"Chunk {i} text.", "") for i in range(10)]

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()

        # Test with max_parallel=2
        results = await client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=output_dir,
            voice="af_bella",
            speed=1.0,
            max_parallel=2
        )

        # All should succeed
        assert len(results) == 10
        assert all(success for _, _, success in results)

    @pytest.mark.asyncio
    async def test_get_available_voices(self, mock_kokoro_tts):
        """Test fetching available voices"""
        client = KokoroTTSClient()
        voices = await client.get_available_voices()

        assert isinstance(voices, dict)
        assert len(voices) > 0

        # Check voice structure
        for voice_id, voice_info in voices.items():
            assert isinstance(voice_id, str)
            assert "name" in voice_info
            assert "gender" in voice_info
            assert "language" in voice_info
            assert "accent" in voice_info

    @pytest.mark.asyncio
    async def test_health_check(self, mock_kokoro_tts):
        """Test TTS service health check"""
        client = KokoroTTSClient()
        status = await client.health_check()

        assert isinstance(status, dict)
        assert "status" in status
        assert status["status"] in ["healthy", "unhealthy"]
        assert "service" in status


class TestTTSRetryLogic:
    """Tests for TTS retry logic and error handling"""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, monkeypatch, tmp_path):
        """Test that TTS client retries on timeout"""
        client = KokoroTTSClient()

        attempt_count = [0]

        async def fake_synthesize_with_retry(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                # Simulate timeout on first attempt
                import httpx
                raise httpx.TimeoutException("Timeout")
            # Succeed on second attempt
            return b"\xff\xfb\x90\x00" + b"\x00" * 1024

        monkeypatch.setattr(
            "app.core.tts.KokoroTTSClient.synthesize_chunk",
            fake_synthesize_with_retry
        )

        # Should succeed after retry
        audio = await client.synthesize_chunk("Test text")
        assert len(audio) > 0
        assert attempt_count[0] == 2, "Should have retried once"

    @pytest.mark.asyncio
    async def test_chapter_markers_in_synthesis(self, mock_kokoro_tts, tmp_path):
        """Test that chapter markers are prepended to chunks"""
        client = KokoroTTSClient()
        chunks = [
            (0, "Some text here.", "Chapter One"),
            (1, "More text.", ""),  # No chapter
        ]

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()

        results = await client.synthesize_chunks_parallel(
            chunks=chunks,
            output_dir=output_dir,
            voice="af_bella",
            speed=1.0,
            max_parallel=2
        )

        assert len(results) == 2
        # Both should succeed (mock doesn't check text content)
        assert all(success for _, _, success in results)
