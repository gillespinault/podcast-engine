"""
PDF Processing Module (Simplified)
Handles PDF validation and cover image extraction only
Text extraction is handled by Gemini File API
"""

try:
    from pypdf import PdfReader  # pypdf 3.x+ (modern)
except ImportError:
    from PyPDF2 import PdfReader  # Fallback for PyPDF2 (legacy)

from pathlib import Path
from typing import Optional
from PIL import Image
import io
import logging
import os

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Raised when PDF validation fails"""
    pass


class SimplePDFProcessor:
    """
    Simplified PDF processor for cover extraction only
    Text extraction is handled by Gemini File API
    """

    def extract_cover_image(
        self,
        pdf_path: Path,
        output_path: Path = None
    ) -> Optional[Path]:
        """
        Extract first image from PDF as cover

        Args:
            pdf_path: Path to PDF file
            output_path: Where to save cover (default: same dir as PDF with _cover.jpg)

        Returns:
            Path to extracted cover image, or None if no image found
        """
        logger.info(f"Extracting cover image from PDF: {pdf_path}")

        try:
            # Default output path
            if not output_path:
                output_path = pdf_path.parent / f"{pdf_path.stem}_cover.jpg"

            # Extract first image from first pages (try pages 1-5 for flexibility)
            for page_no in range(1, min(6, self._get_page_count(pdf_path) + 1)):
                image_data = self._extract_image_from_page(pdf_path, page_no)

                if image_data:
                    # Save image
                    img = Image.open(io.BytesIO(image_data))
                    img.save(output_path, 'JPEG', quality=85)

                    logger.info(f"Extracted cover from page {page_no}: {output_path}")
                    return output_path

            logger.warning(f"No cover image found in first 5 pages of {pdf_path.name}")
            return None

        except Exception as e:
            logger.warning(f"Failed to extract cover image: {e}")
            return None

    def _get_page_count(self, pdf_path: Path) -> int:
        """Get PDF page count"""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception:
            return 0

    def _extract_image_from_page(self, pdf_path: Path, page_no: int) -> Optional[bytes]:
        """
        Extract first image from specific PDF page using pypdf

        Args:
            pdf_path: PDF file path
            page_no: Page number (1-indexed)

        Returns:
            Image bytes, or None if no image found
        """
        try:
            reader = PdfReader(pdf_path)
            page = reader.pages[page_no - 1]  # Convert to 0-indexed

            # Get images from page
            if '/XObject' in page['/Resources']:
                x_objects = page['/Resources']['/XObject'].get_object()

                for obj_name in x_objects:
                    obj = x_objects[obj_name]

                    if obj['/Subtype'] == '/Image':
                        # Found an image
                        size = (obj['/Width'], obj['/Height'])
                        data = obj.get_data()

                        # Skip tiny images (< 100x100)
                        if size[0] < 100 or size[1] < 100:
                            continue

                        logger.debug(f"Found image on page {page_no}: {size}")
                        return data

            return None

        except Exception as e:
            logger.debug(f"No image extracted from page {page_no}: {e}")
            return None


def validate_pdf(pdf_path: Path) -> None:
    """
    Validate PDF before processing

    Args:
        pdf_path: Path to PDF file

    Raises:
        PDFValidationError: If validation fails
    """
    # File exists check
    if not pdf_path.exists():
        raise PDFValidationError(f"PDF file not found: {pdf_path}")

    # File size check
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    max_size_mb = int(os.getenv("MAX_PDF_SIZE_MB", "50"))

    if size_mb > max_size_mb:
        raise PDFValidationError(
            f"PDF too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        )

    logger.info(f"PDF size: {size_mb:.1f} MB (within limit)")

    # Page count check (quick scan without full processing)
    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
    except Exception as e:
        raise PDFValidationError(f"Invalid PDF file: {e}")

    # Gemini File API supports up to 2000 pages, but we set a lower limit for performance
    max_pages = int(os.getenv("MAX_PDF_PAGES", "500"))
    if page_count > max_pages:
        raise PDFValidationError(
            f"Too many pages: {page_count} (max: {max_pages})"
        )

    logger.info(f"PDF validated: {page_count} pages, {size_mb:.1f} MB")


# Convenience function for cover extraction
def extract_cover(pdf_path: Path, output_path: Path = None) -> Optional[Path]:
    """
    Extract cover image from PDF (with validation)

    Args:
        pdf_path: Path to PDF file
        output_path: Where to save cover (default: auto-generated)

    Returns:
        Path to cover image, or None if extraction fails

    Raises:
        PDFValidationError: If validation fails
    """
    # Validate first
    validate_pdf(pdf_path)

    # Extract cover
    processor = SimplePDFProcessor()
    return processor.extract_cover_image(pdf_path, output_path)
