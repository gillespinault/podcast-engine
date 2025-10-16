"""
PDF Processing Module with Docling
Handles PDF text extraction with OCR support
"""

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
from typing import Dict
import logging
import os
try:
    from pypdf import PdfReader  # pypdf 3.x+ (modern)
except ImportError:
    from PyPDF2 import PdfReader  # Fallback for PyPDF2 (legacy)

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Raised when PDF validation fails"""
    pass


class DoclingPDFProcessor:
    """
    PDF processor using IBM Docling
    Supports OCR for scanned PDFs and image extraction
    """

    def __init__(self):
        """Initialize Docling with OCR enabled"""
        # Configure Docling pipeline options (Docling 2.9+ API)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
        pipeline_options.do_table_structure = False  # Skip tables (not needed for audio)

        try:
            # Docling 2.9+ uses format_options instead of pipeline_options
            from docling.document_converter import PdfFormatOption

            self.converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling PDF processor initialized (OCR enabled)")
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            raise

    def extract_text_and_images(
        self,
        pdf_path: Path,
        extract_cover: bool = True,
        cover_output_path: Path = None
    ) -> Dict[str, any]:
        """
        Extract text and image metadata from PDF using Docling

        Args:
            pdf_path: Path to PDF file
            extract_cover: Extract first image as cover (default: True)
            cover_output_path: Where to save cover image (default: temp dir)

        Returns:
            {
                "text": "Full document text (Markdown format)",
                "pages": 123,
                "images": [
                    {"page": 5, "position": "top", "caption": "Figure 1"},
                    ...
                ],
                "metadata": {
                    "title": "Book Title",
                    "author": "Author Name",
                    "headings": ["Chapter 1", "Chapter 2", ...]
                },
                "cover_image_path": "/path/to/cover.jpg" (if extracted)
            }
        """
        logger.info(f"Processing PDF with Docling: {pdf_path}")

        try:
            # Convert PDF to Docling document
            result = self.converter.convert(str(pdf_path))
            doc = result.document

            # Extract text (Markdown format preserves structure)
            full_text = doc.export_to_markdown()
            logger.info(f"Extracted {len(full_text)} characters from PDF")

            # Extract images
            images_metadata = []
            cover_image_path = None

            for idx, picture in enumerate(doc.pictures):
                images_metadata.append({
                    "page": picture.prov[0].page_no if picture.prov else 0,
                    "position": self._estimate_position(picture),
                    # Docling 2.9 compatibility: caption may not exist
                    "caption": getattr(picture, 'caption', None),
                    "obj_name": getattr(picture, 'self_ref', None)
                })

                # Extract first image as cover
                if extract_cover and idx == 0 and not cover_image_path:
                    cover_image_path = self._extract_cover_image(
                        picture,
                        pdf_path,
                        cover_output_path
                    )

            logger.info(f"Found {len(images_metadata)} images in PDF")
            if cover_image_path:
                logger.info(f"Extracted cover image: {cover_image_path}")

            # Extract headings (for chapter detection assistance)
            headings = [
                text.text for text in doc.texts
                if text.label == "heading"
            ]

            logger.info(f"Found {len(headings)} headings in PDF")

            # Extract metadata
            metadata = {
                "title": doc.name or pdf_path.stem,
                "pages": len(doc.pages),
                "headings": headings[:50],  # Limit to first 50 headings
            }

            return {
                "text": full_text,
                "pages": len(doc.pages),
                "images": images_metadata,
                "metadata": metadata,
                "cover_image_path": cover_image_path,
            }

        except Exception as e:
            logger.exception(f"Docling extraction failed: {e}")
            raise PDFValidationError(f"Failed to extract text from PDF: {e}")

    def _estimate_position(self, picture) -> str:
        """
        Estimate image position: top, middle, bottom

        Args:
            picture: Docling picture object

        Returns:
            Position string
        """
        if not picture.prov:
            return "unknown"

        # Use bounding box coordinates if available
        bbox = picture.prov[0].bbox
        if bbox:
            # Normalize y-coordinate (0-1 scale)
            y_center = (bbox.t + bbox.b) / 2
            if y_center < 0.33:
                return "top"
            elif y_center < 0.66:
                return "middle"
            else:
                return "bottom"

        return "middle"  # Default fallback

    def _extract_cover_image(
        self,
        picture,
        pdf_path: Path,
        output_path: Path = None
    ) -> Path:
        """
        Extract first image from PDF as cover

        Args:
            picture: Docling PictureItem object
            pdf_path: Source PDF path
            output_path: Where to save cover (default: same dir as PDF with _cover.jpg)

        Returns:
            Path to extracted cover image, or None if extraction fails
        """
        try:
            import tempfile
            from PIL import Image
            import io

            # Default output path
            if not output_path:
                output_path = pdf_path.parent / f"{pdf_path.stem}_cover.jpg"

            # Try to get image from Docling picture object
            # Docling 2.9 may have image data in different attributes
            image_data = None

            # Attempt 1: Check if picture has PIL image directly
            if hasattr(picture, 'pil_image') and picture.pil_image:
                image_data = picture.pil_image
                logger.debug("Extracted cover from Docling pil_image")

            # Attempt 2: Use pypdf to extract first image from page
            if not image_data and picture.prov:
                try:
                    page_no = picture.prov[0].page_no if picture.prov else 1
                    image_data = self._extract_image_from_page(pdf_path, page_no)
                    logger.debug(f"Extracted cover from page {page_no} using pypdf")
                except Exception as e:
                    logger.warning(f"Failed to extract image from page: {e}")

            # Save image
            if image_data:
                if isinstance(image_data, Image.Image):
                    # PIL Image - save directly
                    image_data.save(output_path, 'JPEG', quality=85)
                else:
                    # Bytes - convert to PIL first
                    img = Image.open(io.BytesIO(image_data))
                    img.save(output_path, 'JPEG', quality=85)

                logger.info(f"Saved cover image to {output_path}")
                return output_path
            else:
                logger.warning("Could not extract cover image - no image data available")
                return None

        except Exception as e:
            logger.warning(f"Failed to extract cover image: {e}")
            return None

    def _extract_image_from_page(self, pdf_path: Path, page_no: int) -> bytes:
        """
        Extract first image from specific PDF page using pypdf

        Args:
            pdf_path: PDF file path
            page_no: Page number (1-indexed)

        Returns:
            Image bytes
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

                        # Return first image found
                        logger.debug(f"Found image on page {page_no}: {size}")
                        return data

            return None

        except Exception as e:
            logger.warning(f"pypdf image extraction failed: {e}")
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

    max_pages = int(os.getenv("MAX_PDF_PAGES", "500"))
    if page_count > max_pages:
        raise PDFValidationError(
            f"Too many pages: {page_count} (max: {max_pages})"
        )

    logger.info(f"PDF validated: {page_count} pages, {size_mb:.1f} MB")


# Convenience function for simple extraction
def extract_pdf(pdf_path: Path) -> Dict[str, any]:
    """
    Extract text and images from PDF (with validation)

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extraction result dictionary

    Raises:
        PDFValidationError: If validation fails
    """
    # Validate first
    validate_pdf(pdf_path)

    # Extract with Docling
    processor = DoclingPDFProcessor()
    return processor.extract_text_and_images(pdf_path)
