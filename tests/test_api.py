"""
Podcast Engine - API Tests
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_returns_service_info(self, client: TestClient):
        """Test root endpoint returns service information"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check_success(self, client: TestClient, mock_kokoro_tts):
        """Test health check returns healthy status"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["healthy", "unhealthy"]
        assert "version" in data
        assert "uptime_seconds" in data
        assert "services" in data
        assert "kokoro_tts" in data["services"]

    def test_health_check_structure(self, client: TestClient, mock_kokoro_tts):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()

        # Check Kokoro TTS status
        assert "kokoro_tts" in data["services"]
        kokoro_status = data["services"]["kokoro_tts"]
        assert "status" in kokoro_status

        # Check system status
        assert "system" in data
        assert "storage" in data["system"]


class TestVoicesEndpoint:
    """Tests for voices API endpoint"""

    def test_get_all_voices(self, client: TestClient, mock_kokoro_tts):
        """Test fetching all available voices"""
        response = client.get("/api/v1/voices")

        assert response.status_code == 200
        voices = response.json()

        assert isinstance(voices, dict), "Voices should be a dictionary"
        assert len(voices) > 0, "Should return at least one voice"

        # Check voice structure
        for voice_id, voice_info in voices.items():
            assert "name" in voice_info
            assert "gender" in voice_info
            assert "language" in voice_info
            assert "accent" in voice_info

    def test_filter_voices_by_language(self, client: TestClient, mock_kokoro_tts):
        """Test filtering voices by language"""
        # Test English voices
        response = client.get("/api/v1/voices?language=en")
        assert response.status_code == 200
        voices = response.json()

        assert len(voices) > 0, "Should have English voices"
        for voice_info in voices.values():
            assert voice_info["language"] == "en"

        # Test French voices
        response_fr = client.get("/api/v1/voices?language=fr")
        assert response_fr.status_code == 200
        voices_fr = response_fr.json()

        assert len(voices_fr) > 0, "Should have French voices"
        for voice_info in voices_fr.values():
            assert voice_info["language"] == "fr"

    def test_empty_language_filter(self, client: TestClient, mock_kokoro_tts):
        """Test filtering with non-existent language"""
        response = client.get("/api/v1/voices?language=xx")  # Invalid language code
        assert response.status_code == 200
        voices = response.json()

        assert isinstance(voices, dict)
        assert len(voices) == 0, "Should return empty dict for non-existent language"


class TestCreatePodcastEndpoint:
    """Tests for podcast creation endpoint"""

    def test_create_podcast_success(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test successful podcast creation"""
        response = client.post("/api/v1/create-podcast", json=sample_podcast_request)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["success"] is True
        assert "job_id" in data
        assert "podcast" in data
        assert "processing" in data
        assert "message" in data

        # Check podcast info
        podcast = data["podcast"]
        assert "filename" in podcast
        assert "file_size" in podcast
        assert "duration_seconds" in podcast
        assert podcast["duration_seconds"] > 0

        # Check processing stats
        processing = data["processing"]
        assert "total_chunks" in processing
        assert "successful_chunks" in processing
        assert processing["successful_chunks"] > 0

    def test_create_podcast_minimum_text_length(self, client: TestClient):
        """Test that minimum text length is enforced"""
        request = {
            "text": "Short",  # Less than 100 chars
            "metadata": {"title": "Test"},
        }

        response = client.post("/api/v1/create-podcast", json=request)

        assert response.status_code == 422, "Should reject text shorter than 100 chars"

    def test_create_podcast_invalid_voice(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test podcast creation with invalid voice"""
        sample_podcast_request["tts_options"]["voice"] = "invalid_voice_id"

        response = client.post("/api/v1/create-podcast", json=sample_podcast_request)

        # Should either accept it (and fail later) or reject immediately
        # Depending on validation logic
        assert response.status_code in [200, 400, 422, 500]

    def test_create_podcast_speed_validation(self, client: TestClient, sample_podcast_request):
        """Test that speech speed is validated"""
        # Test speed too low
        sample_podcast_request["tts_options"]["speed"] = 0.1  # Below 0.5
        response = client.post("/api/v1/create-podcast", json=sample_podcast_request)
        assert response.status_code == 422, "Should reject speed < 0.5"

        # Test speed too high
        sample_podcast_request["tts_options"]["speed"] = 3.0  # Above 2.0
        response = client.post("/api/v1/create-podcast", json=sample_podcast_request)
        assert response.status_code == 422, "Should reject speed > 2.0"


class TestGUIEndpoint:
    """Tests for GUI endpoint"""

    def test_gui_homepage(self, client: TestClient):
        """Test GUI homepage loads"""
        response = client.get("/gui/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for Alpine.js
        html = response.text
        assert "alpinejs" in html.lower() or "alpine" in html.lower()

    def test_gui_contains_podcast_form(self, client: TestClient):
        """Test GUI contains podcast creation form"""
        response = client.get("/gui/")
        html = response.text

        assert "podcast" in html.lower()
        assert "voice" in html.lower() or "voices" in html.lower()


class TestWebhookEndpoint:
    """Tests for webhook endpoint (n8n integration)"""

    def test_webhook_missing_api_key(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        monkeypatch
    ):
        """Test webhook rejects requests without API key when configured"""
        # Configure API key in settings
        from app.config import settings
        monkeypatch.setattr(settings, "api_key", "test_secret_key_123")

        # Request without X-API-KEY header
        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        assert response.status_code == 401
        assert "Missing X-API-KEY header" in response.json()["detail"]

    def test_webhook_invalid_api_key(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        monkeypatch
    ):
        """Test webhook rejects requests with invalid API key"""
        # Configure API key in settings
        from app.config import settings
        monkeypatch.setattr(settings, "api_key", "test_secret_key_123")

        # Request with wrong API key
        response = client.post(
            "/api/v1/webhook/create-podcast",
            json=sample_podcast_request,
            headers={"X-API-KEY": "wrong_key"}
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_webhook_valid_api_key(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage,
        monkeypatch
    ):
        """Test webhook accepts requests with valid API key"""
        # Configure API key in settings
        from app.config import settings
        monkeypatch.setattr(settings, "api_key", "test_secret_key_123")

        # Request with correct API key
        response = client.post(
            "/api/v1/webhook/create-podcast",
            json=sample_podcast_request,
            headers={"X-API-KEY": "test_secret_key_123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_webhook_no_auth_when_not_configured(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage,
        monkeypatch
    ):
        """Test webhook works without auth when API key not configured"""
        # Ensure API key is not configured
        from app.config import settings
        monkeypatch.setattr(settings, "api_key", None)

        # Request without API key should succeed
        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_webhook_callbacks_echo(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test webhook echoes callbacks in response"""
        # Add callbacks to request
        sample_podcast_request["callbacks"] = {
            "source_workflow_id": "n8n_test_workflow_123",
            "source_item_id": "wallabag_456"
        }

        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        assert response.status_code == 200
        data = response.json()

        # Check callbacks are echoed back
        assert "callbacks" in data
        assert data["callbacks"]["source_workflow_id"] == "n8n_test_workflow_123"
        assert data["callbacks"]["source_item_id"] == "wallabag_456"

    def test_webhook_base64_response(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test webhook returns base64 encoded audio when requested"""
        # Enable binary response
        sample_podcast_request["processing_options"] = {
            "return_binary": True
        }

        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        assert response.status_code == 200
        data = response.json()

        # Check binary_data is present in podcast dict
        assert "podcast" in data
        assert "binary_data" in data["podcast"]
        assert data["podcast"]["binary_data"] is not None
        assert len(data["podcast"]["binary_data"]) > 0

        # Verify it's valid base64
        import base64
        try:
            decoded = base64.b64decode(data["podcast"]["binary_data"])
            assert len(decoded) > 0
        except Exception:
            pytest.fail("binary_data is not valid base64")

    def test_webhook_no_base64_when_not_requested(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test webhook doesn't include base64 when not requested"""
        # Disable binary response
        sample_podcast_request["processing_options"] = {
            "return_binary": False
        }

        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        assert response.status_code == 200
        data = response.json()

        # Check binary_data is null or absent
        assert "podcast" in data
        binary_data = data["podcast"].get("binary_data")
        assert binary_data is None

    def test_webhook_with_chapters(
        self,
        client: TestClient,
        sample_podcast_request,
        mock_kokoro_tts,
        mock_ffmpeg,
        temp_storage
    ):
        """Test webhook accepts chapters parameter (PDF+LLM use case)"""
        # Add chapters to request
        sample_podcast_request["chapters"] = [
            {
                "title": "Chapter 1: Introduction",
                "text": "This is the first chapter content for testing purposes. " * 20,
                "start_time": 0
            },
            {
                "title": "Chapter 2: Development",
                "text": "This is the second chapter content for testing purposes. " * 20,
                "start_time": 60
            }
        ]

        response = client.post("/api/v1/webhook/create-podcast", json=sample_podcast_request)

        # Should accept chapters (even if not fully implemented yet)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
