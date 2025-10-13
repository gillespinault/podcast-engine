"""
Podcast Engine - Configuration
Environment-based configuration using pydantic-settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Application
    app_name: str = Field(default="Podcast Engine", description="Application name")
    app_version: str = Field(default="1.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=2, ge=1, le=10, description="Uvicorn workers")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")

    # External Services
    kokoro_tts_url: HttpUrl = Field(
        default="http://serverlabapps-kokorotts-skwerq:8880/v1/audio/speech",
        description="Kokoro TTS API endpoint"
    )
    kokoro_timeout: int = Field(default=120, ge=10, le=600, description="Kokoro API timeout (seconds)")

    # Storage
    storage_base_path: str = Field(default="/data/shared/podcasts", description="Base storage directory")
    temp_dir: str = Field(default="/data/shared/podcasts/jobs", description="Temporary files directory")
    final_dir: str = Field(default="/data/shared/podcasts/final", description="Final podcasts directory")
    cleanup_temp_after_hours: int = Field(default=24, ge=1, le=168, description="Cleanup temp files after N hours")

    # Processing Limits
    max_text_length: int = Field(default=5_000_000, ge=100, description="Max text length (chars)")
    max_concurrent_jobs: int = Field(default=3, ge=1, le=10, description="Max simultaneous podcast creations")
    max_parallel_tts_calls: int = Field(default=10, ge=1, le=20, description="Max parallel TTS API calls")

    # Defaults
    default_voice: str = Field(default="af_bella", description="Default TTS voice")
    default_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Default speech speed")
    default_bitrate: str = Field(default="64k", description="Default audio bitrate")
    default_sample_rate: int = Field(default=24000, description="Default sample rate (Hz)")

    # Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication (optional)")
    enable_rate_limiting: bool = Field(default=False, description="Enable API rate limiting")
    rate_limit_per_minute: int = Field(default=10, ge=1, le=1000, description="Max requests per minute")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="Prometheus metrics port")

    # GUI
    enable_gui: bool = Field(default=True, description="Enable web GUI")
    gui_theme: str = Field(default="dark", description="GUI theme (dark/light)")

    class Config:
        env_file = ".env"
        env_prefix = "PODCAST_ENGINE_"
        case_sensitive = False


# Singleton instance
settings = Settings()


# ============================================================================
# DEPRECATED: Legacy hardcoded voices (replaced by dynamic loading)
# ============================================================================
# This dict is NO LONGER USED by the application.
# Voices are now loaded dynamically at startup via:
#   app.state.available_voices = await tts_client.get_available_voices()
#
# Current implementation loads 67 voices from Kokoro TTS dynamically.
# See: app/core/tts.py:215-303 (get_available_voices method)
#      app/main.py:39-44 (lifespan startup)
#
# This variable is kept temporarily for backwards compatibility but should be
# removed in future versions once all references are eliminated.
#
# TODO: Remove completely after verifying no external dependencies
# ============================================================================
# AVAILABLE_VOICES = {
#     "af_bella": {"name": "Bella", "gender": "Female", "accent": "American", "quality": "High"},
#     "af_sarah": {"name": "Sarah", "gender": "Female", "accent": "American", "quality": "High"},
#     "af_nicole": {"name": "Nicole", "gender": "Female", "accent": "American", "quality": "Medium"},
#     "bf_emma": {"name": "Emma", "gender": "Female", "accent": "British", "quality": "High"},
#     "bf_isabella": {"name": "Isabella", "gender": "Female", "accent": "British", "quality": "Medium"},
#     "am_adam": {"name": "Adam", "gender": "Male", "accent": "American", "quality": "High"},
#     "am_michael": {"name": "Michael", "gender": "Male", "accent": "American", "quality": "Medium"},
#     "bm_george": {"name": "George", "gender": "Male", "accent": "British", "quality": "High"},
#     "bm_lewis": {"name": "Lewis", "gender": "Male", "accent": "British", "quality": "Medium"},
# }


# Audio format specifications
AUDIO_FORMATS = {
    "m4b": {"extension": ".m4b", "codec": "aac", "container": "mp4", "best_for": "Audiobooks"},
    "mp3": {"extension": ".mp3", "codec": "libmp3lame", "container": "mp3", "best_for": "Podcasts"},
    "opus": {"extension": ".opus", "codec": "libopus", "container": "ogg", "best_for": "Voice"},
    "aac": {"extension": ".m4a", "codec": "aac", "container": "mp4", "best_for": "Music"},
}
