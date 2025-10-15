"""
Podcast Engine - Audio Processing
ffmpeg operations and metadata embedding
"""
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List
from loguru import logger
import mutagen
from mutagen.mp4 import MP4, MP4Cover
from datetime import datetime

from app.config import settings, AUDIO_FORMATS


class AudioProcessor:
    """Audio processing with ffmpeg and mutagen"""

    def __init__(self):
        """Initialize audio processor"""
        self.ffmpeg_bin = "ffmpeg"
        self.ffprobe_bin = "ffprobe"

        # Verify ffmpeg installation
        try:
            subprocess.run([self.ffmpeg_bin, "-version"], capture_output=True, check=True)
            logger.info("✓ ffmpeg available")
        except Exception as e:
            logger.error(f"ffmpeg not found: {e}")
            raise RuntimeError("ffmpeg is required but not installed")

    async def merge_audio_files(
        self,
        input_files: List[Path],
        output_path: Path,
        format: str = "m4b",
        bitrate: str = "64k",
        sample_rate: int = 24000,
        channels: int = 1,
        add_silence_start: float = 0.5,
        add_silence_end: float = 1.0,
    ) -> Path:
        """
        Merge multiple audio files into single output

        Args:
            input_files: List of audio file paths (ordered)
            output_path: Output file path
            format: Output format (m4b, mp3, opus, aac)
            bitrate: Audio bitrate
            sample_rate: Sample rate in Hz
            channels: Number of channels (1=mono, 2=stereo)
            add_silence_start: Silence at start (seconds)
            add_silence_end: Silence at end (seconds)

        Returns:
            Path to output file

        Raises:
            RuntimeError: On ffmpeg error
        """
        try:
            logger.info(f"Merging {len(input_files)} audio files to {output_path.name}")

            # Sort input files by name (chunk_0000.mp3, chunk_0001.mp3, ...)
            input_files = sorted(input_files, key=lambda p: p.name)

            # Create concat list file
            concat_file = output_path.parent / "concat_list.txt"
            with open(concat_file, "w") as f:
                for audio_file in input_files:
                    # ffmpeg concat requires absolute paths with escaped quotes
                    f.write(f"file '{audio_file.absolute()}'\n")

            # Get format config
            format_config = AUDIO_FORMATS.get(format, AUDIO_FORMATS["m4b"])
            codec = format_config["codec"]
            extension = format_config["extension"]

            # Ensure output path has correct extension
            if not output_path.suffix == extension:
                output_path = output_path.with_suffix(extension)

            # Build ffmpeg command
            cmd = [
                self.ffmpeg_bin,
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
            ]

            # Add silence at start if requested
            if add_silence_start > 0:
                cmd.extend([
                    "-af", f"adelay={int(add_silence_start * 1000)}|{int(add_silence_start * 1000)}"
                ])

            # Audio encoding options
            cmd.extend([
                "-c:a", codec,
                "-b:a", bitrate,
                "-ar", str(sample_rate),
                "-ac", str(channels),
            ])

            # Add silence at end (via duration padding)
            if add_silence_end > 0:
                cmd.extend(["-af", f"apad=pad_dur={add_silence_end}"])

            # Output file
            cmd.append(str(output_path))

            # Execute ffmpeg
            logger.debug(f"Running ffmpeg: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode()
                logger.error(f"ffmpeg failed: {error_msg}")
                raise RuntimeError(f"ffmpeg merge failed: {error_msg}")

            # Cleanup concat file
            concat_file.unlink(missing_ok=True)

            # Verify output exists
            if not output_path.exists():
                raise RuntimeError(f"Output file not created: {output_path}")

            logger.success(f"Merged audio saved to {output_path.name} ({output_path.stat().st_size} bytes)")

            return output_path

        except Exception as e:
            logger.error(f"Audio merge failed: {e}")
            raise

    def embed_metadata(
        self,
        audio_path: Path,
        title: str,
        author: Optional[str] = None,
        description: Optional[str] = None,
        album: Optional[str] = None,
        genre: Optional[str] = None,
        narrator: Optional[str] = None,
        publisher: Optional[str] = None,
        copyright: Optional[str] = None,
        publication_date: Optional[datetime] = None,
        cover_image_path: Optional[Path] = None,
        track_number: Optional[int] = None,
    ) -> None:
        """
        Embed metadata into audio file

        Args:
            audio_path: Path to audio file (M4B/M4A)
            title: Title
            author: Author name
            description: Description
            album: Album name
            genre: Genre
            narrator: Narrator name
            publisher: Publisher name
            copyright: Copyright notice
            publication_date: Publication date
            cover_image_path: Path to cover image (JPEG/PNG)
            track_number: Track/episode number (for podcast series)
        """
        try:
            logger.info(f"Embedding metadata into {audio_path.name}")

            # Load audio file with mutagen
            audio = MP4(str(audio_path))

            # Set metadata
            if title:
                audio["\xa9nam"] = title  # Title
            if author:
                audio["\xa9ART"] = author  # Artist/Author
            if album:
                audio["\xa9alb"] = album  # Album
            if description:
                audio["\xa9cmt"] = description  # Comment/Description
            if genre:
                audio["\xa9gen"] = genre  # Genre
            if narrator:
                audio["\xa9wrt"] = narrator  # Writer/Narrator
            if publisher:
                audio["----:com.apple.iTunes:PUBLISHER"] = publisher.encode("utf-8")
            if copyright:
                audio["cprt"] = copyright  # Copyright
            if publication_date:
                audio["\xa9day"] = publication_date.strftime("%Y-%m-%d")  # Date
            if track_number is not None:
                # Track number (for podcast episodes): format is [(track, total_tracks)]
                # We only set track number, total_tracks is 0 (unknown)
                audio["trkn"] = [(track_number, 0)]
                logger.info(f"Set track number: {track_number}")

            # Set as audiobook
            audio["stik"] = [2]  # Media type: Audiobook (2)

            # Embed cover art if provided
            if cover_image_path and cover_image_path.exists():
                logger.info(f"Embedding cover art from {cover_image_path.name}")

                with open(cover_image_path, "rb") as f:
                    cover_data = f.read()

                # Determine image format
                if cover_image_path.suffix.lower() in [".jpg", ".jpeg"]:
                    image_format = MP4Cover.FORMAT_JPEG
                elif cover_image_path.suffix.lower() == ".png":
                    image_format = MP4Cover.FORMAT_PNG
                else:
                    logger.warning(f"Unsupported cover image format: {cover_image_path.suffix}")
                    image_format = MP4Cover.FORMAT_JPEG

                audio["covr"] = [MP4Cover(cover_data, imageformat=image_format)]

            # Save metadata
            audio.save()

            logger.success(f"Metadata embedded successfully")

        except Exception as e:
            logger.error(f"Failed to embed metadata: {e}")
            raise

    async def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration using ffprobe

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                self.ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                duration = float(stdout.decode().strip())
                return duration
            else:
                logger.error(f"ffprobe failed: {stderr.decode()}")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    async def download_cover_image(self, cover_url: str, output_path: Path) -> Optional[Path]:
        """
        Download cover image from URL

        Args:
            cover_url: URL to cover image
            output_path: Path to save image

        Returns:
            Path to downloaded image, or None on failure
        """
        try:
            import httpx

            logger.info(f"Downloading cover image from {cover_url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(cover_url)
                response.raise_for_status()

                # Save image
                output_path.write_bytes(response.content)

                logger.success(f"Cover image downloaded to {output_path.name}")

                return output_path

        except Exception as e:
            logger.error(f"Failed to download cover image: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    async def test_audio():
        processor = AudioProcessor()

        # Test merge (mock files)
        input_dir = Path("/tmp/test_audio")
        input_dir.mkdir(exist_ok=True)

        # Create dummy audio files (would be generated by TTS in real scenario)
        # ...

        # Test metadata embedding
        test_file = Path("/tmp/test.m4b")
        if test_file.exists():
            processor.embed_metadata(
                audio_path=test_file,
                title="Test Audiobook",
                author="Test Author",
                description="This is a test audiobook",
                genre="Technology",
                narrator="Kokoro TTS"
            )

    asyncio.run(test_audio())
