"""
Podcast Engine - API Models
Pydantic models for request/response validation
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator
from datetime import datetime


class PodcastMetadata(BaseModel):
    """Metadata for the podcast/audiobook"""
    title: str = Field(..., min_length=1, max_length=500, description="Podcast title (required)")
    author: Optional[str] = Field(default="Unknown", max_length=200, description="Author name")
    description: Optional[str] = Field(default=None, max_length=5000, description="Podcast description")
    source_url: Optional[HttpUrl] = Field(default=None, description="Original source URL")
    language: str = Field(default="fr", pattern="^[a-z]{2}$", description="Language code (ISO 639-1)")
    cover_image_url: Optional[HttpUrl] = Field(default=None, description="Cover image URL")
    tags: List[str] = Field(default_factory=list, max_length=20, description="Tags/categories")
    publication_date: Optional[datetime] = Field(default=None, description="Publication date (ISO 8601)")
    isbn: Optional[str] = Field(default=None, max_length=20, description="ISBN (for books)")
    narrator: Optional[str] = Field(default="Kokoro TTS", max_length=200, description="Narrator name")
    publisher: Optional[str] = Field(default="Podcast Engine", max_length=200, description="Publisher")
    copyright: Optional[str] = Field(default=None, max_length=500, description="Copyright notice")
    genre: Optional[str] = Field(default="Audiobook", max_length=100, description="Genre")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Deep Learning Fundamentals",
                "author": "Andrew Ng",
                "description": "Introduction to deep learning concepts",
                "source_url": "https://example.com/article",
                "language": "en",
                "tags": ["AI", "machine-learning", "education"],
                "genre": "Technology"
            }
        }


class TTSOptions(BaseModel):
    """Text-to-Speech configuration options"""
    voice: str = Field(
        default="af_bella",
        description="TTS voice ID (Kokoro voices: af_bella, af_sarah, af_nicole, bf_emma, bf_isabella, am_adam, am_michael, bm_george, bm_lewis)"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5 = slow, 2.0 = fast)")
    chunk_size: int = Field(default=4000, ge=1000, le=10000, description="Max characters per TTS chunk")
    preserve_sentence: bool = Field(default=True, description="Split chunks at sentence boundaries")
    add_chapter_markers: bool = Field(default=True, description="Add 'Part X/Y' chapter markers")
    pause_between_chunks: float = Field(default=0.5, ge=0.0, le=5.0, description="Silence between chunks (seconds)")
    normalize_audio: bool = Field(default=True, description="Normalize audio volume")
    remove_urls: bool = Field(default=True, description="Remove URLs from text before TTS")
    remove_markdown: bool = Field(default=True, description="Strip markdown formatting")

    class Config:
        json_schema_extra = {
            "example": {
                "voice": "af_bella",
                "speed": 1.1,
                "chunk_size": 5000,
                "preserve_sentence": True,
                "add_chapter_markers": True
            }
        }


class AudioOptions(BaseModel):
    """Audio output configuration"""
    format: Literal["m4b", "mp3", "opus", "aac"] = Field(default="m4b", description="Output audio format")
    bitrate: str = Field(default="64k", pattern="^[0-9]+k$", description="Audio bitrate (e.g., 32k, 64k, 128k)")
    sample_rate: int = Field(default=24000, ge=16000, le=48000, description="Sample rate in Hz")
    channels: Literal[1, 2] = Field(default=1, description="Audio channels (1=mono, 2=stereo)")
    codec: Optional[str] = Field(default=None, description="Audio codec (auto-detected if None)")
    embed_cover: bool = Field(default=True, description="Embed cover art in audio file")
    add_silence_start: float = Field(default=0.5, ge=0.0, le=5.0, description="Silence at start (seconds)")
    add_silence_end: float = Field(default=1.0, ge=0.0, le=5.0, description="Silence at end (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "m4b",
                "bitrate": "64k",
                "sample_rate": 24000,
                "channels": 1,
                "embed_cover": True
            }
        }


class ProcessingOptions(BaseModel):
    """Processing behavior options"""
    async_mode: bool = Field(default=False, description="Return immediately with job ID (true) or wait for completion (false)")
    max_parallel_tts: int = Field(default=5, ge=1, le=20, description="Max parallel TTS API calls")
    retry_on_error: bool = Field(default=True, description="Retry failed TTS chunks")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts per chunk")
    cleanup_on_error: bool = Field(default=True, description="Delete temporary files on error")
    return_binary: bool = Field(default=True, description="Return binary audio file in response")
    save_to_storage: bool = Field(default=True, description="Save final podcast to /data/shared/podcasts/final/")
    storage_path: Optional[str] = Field(default=None, description="Custom storage path (overrides default)")

    class Config:
        json_schema_extra = {
            "example": {
                "async_mode": False,
                "max_parallel_tts": 5,
                "retry_on_error": True,
                "return_binary": True
            }
        }


class PodcastRequest(BaseModel):
    """Complete podcast creation request"""
    text: str = Field(..., min_length=100, max_length=5_000_000, description="Text content to convert (required)")
    metadata: PodcastMetadata
    tts_options: TTSOptions = Field(default_factory=TTSOptions)
    audio_options: AudioOptions = Field(default_factory=AudioOptions)
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text is not just whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample article about artificial intelligence...",
                "metadata": {
                    "title": "AI Introduction",
                    "author": "John Doe",
                    "language": "en"
                },
                "tts_options": {
                    "voice": "af_bella",
                    "speed": 1.0
                },
                "audio_options": {
                    "format": "m4b",
                    "bitrate": "64k"
                }
            }
        }


class ProcessingStats(BaseModel):
    """Statistics about the processing"""
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    total_duration_seconds: float
    processing_time_seconds: float
    tts_api_calls: int
    average_chunk_time: float
    text_length_chars: int
    text_length_words: int
    estimated_listening_time_minutes: float


class PodcastResponse(BaseModel):
    """Response after podcast creation"""
    success: bool
    job_id: str
    podcast: Optional[dict] = Field(default=None, description="Podcast file information")
    processing: Optional[ProcessingStats] = Field(default=None)
    message: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "abc123-def456",
                "podcast": {
                    "filename": "ai-introduction-2025-10-12.m4b",
                    "file_path": "/data/shared/podcasts/final/abc123.m4b",
                    "file_size": 12345678,
                    "duration_seconds": 3600,
                    "format": "m4b"
                },
                "processing": {
                    "total_chunks": 12,
                    "successful_chunks": 12,
                    "processing_time_seconds": 450
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"]
    version: str
    uptime_seconds: float
    services: dict = Field(description="Status of dependent services (Kokoro TTS, etc.)")
    system: dict = Field(description="System resources (disk, memory)")


# ============================================================================
# Webhook-specific Models (for n8n automation)
# ============================================================================

class WebhookCallbacks(BaseModel):
    """Optional callbacks for workflow tracking (n8n integration)"""
    source_workflow_id: Optional[str] = Field(
        default=None,
        description="n8n workflow ID that triggered this request"
    )
    source_item_id: Optional[str] = Field(
        default=None,
        description="Source item ID (e.g., wallabag article ID, PDF document ID)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_workflow_id": "n8n_workflow_abc123",
                "source_item_id": "wallabag_456"
            }
        }


class ChapterInfo(BaseModel):
    """Chapter information for chaptered audiobooks (PDF+LLM use case)"""
    title: str = Field(..., min_length=1, max_length=200, description="Chapter title")
    text: str = Field(..., min_length=1, description="Chapter text content")
    start_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Start time in seconds (auto-calculated if None)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Chapter 1: Introduction",
                "text": "This is the first chapter content...",
                "start_time": 0
            }
        }


class WebhookPodcastRequest(PodcastRequest):
    """Extended podcast request for webhook endpoint with callbacks & chapters"""
    callbacks: Optional[WebhookCallbacks] = Field(
        default=None,
        description="Optional tracking callbacks for n8n workflows"
    )
    chapters: Optional[List[ChapterInfo]] = Field(
        default=None,
        description="Optional chapter information for multi-chapter audiobooks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample article about artificial intelligence...",
                "metadata": {
                    "title": "AI Introduction",
                    "author": "example.com",
                    "language": "en",
                    "tags": ["wallabag", "AI", "tech"]
                },
                "tts_options": {
                    "voice": "af_bella",
                    "speed": 1.0
                },
                "audio_options": {
                    "format": "m4b",
                    "bitrate": "64k"
                },
                "processing_options": {
                    "return_binary": True
                },
                "callbacks": {
                    "source_workflow_id": "n8n_wallabag_automation",
                    "source_item_id": "wallabag_123"
                }
            }
        }


class WebhookPodcastResponse(PodcastResponse):
    """Extended response for webhook endpoint with callbacks echo"""
    callbacks: Optional[WebhookCallbacks] = Field(
        default=None,
        description="Echo of the callbacks from the request"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "abc123-def456",
                "podcast": {
                    "filename": "ai-introduction-2025-10-12.m4b",
                    "file_path": "/data/shared/podcasts/final/abc123.m4b",
                    "file_size": 12345678,
                    "duration_seconds": 3600,
                    "format": "m4b",
                    "binary_data": "base64_encoded_audio_here..."
                },
                "processing": {
                    "total_chunks": 12,
                    "successful_chunks": 12,
                    "processing_time_seconds": 450
                },
                "callbacks": {
                    "source_workflow_id": "n8n_wallabag_automation",
                    "source_item_id": "wallabag_123"
                }
            }
        }
