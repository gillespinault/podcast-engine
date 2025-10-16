"""
Gemini API Client for Document Analysis
Handles chapter detection, language detection, and audio-friendly reformatting
Uses Gemini File API for direct PDF processing (no Docling dependency)
"""

import google.generativeai as genai
from typing import List, Dict, Optional
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import re
from pathlib import Path
import time

# JSON repair for handling malformed Gemini responses
try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("json_repair not installed, using fallback JSON parser")

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Google Gemini 2.5 Pro API client
    Analyzes documents for audiobook conversion
    """

    def __init__(self):
        """Initialize Gemini API client"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Please configure in environment variables."
            )

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini client initialized (model: {model_name})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True
    )
    def analyze_document(
        self,
        text: str,
        images_metadata: List[Dict],
        metadata_hint: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze document structure and generate chapters

        Args:
            text: Document text (Markdown format from Docling)
            images_metadata: List of image positions
            metadata_hint: Optional hints (headings, title)

        Returns:
            {
                "language": "fr" | "en" | "es" | ...,  # ISO 639-1 code
                "document_type": "book" | "magazine" | "article",
                "chapters": [
                    {
                        "number": 1,
                        "title": "Introduction",
                        "text": "Reformatted chapter text...",
                        "audio_duration_estimate": 120  # seconds
                    },
                    ...
                ],
                "images_integrated": 5,  # Count of images described
                "tts_voice_suggestion": "ff_siwis"  # Suggested Kokoro voice
            }

        Raises:
            Exception: If Gemini API call fails after retries
        """
        prompt = self._build_chaptering_prompt(text, images_metadata, metadata_hint)

        logger.info(
            f"Sending document to Gemini for analysis "
            f"(text_length={len(text)}, images={len(images_metadata)})"
        )

        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini analysis complete, parsing response")

            # Try to parse JSON response
            result = self._parse_gemini_response(response.text)

            # Validate required fields
            self._validate_analysis_result(result)

            logger.info(
                f"Analysis successful: language={result['language']}, "
                f"type={result['document_type']}, chapters={len(result['chapters'])}"
            )

            return result

        except Exception as e:
            logger.exception(f"Gemini analysis failed: {e}")
            raise

    def _build_chaptering_prompt(
        self,
        text: str,
        images_metadata: List[Dict],
        metadata_hint: Optional[Dict]
    ) -> str:
        """Generate prompt for Gemini 2.5 Pro"""

        # Add headings hint if available
        headings_hint = ""
        if metadata_hint and "headings" in metadata_hint and metadata_hint["headings"]:
            headings_sample = ", ".join(metadata_hint["headings"][:10])
            headings_hint = f"\n**Document Headings Detected**: {headings_sample}"

        # Truncate text to fit context (Gemini 2.5 Pro has 1M tokens ~ 750k words)
        # Conservative limit: 100k characters (~ 15k words)
        text_sample = text[:100000]
        if len(text) > 100000:
            logger.warning(
                f"Text truncated from {len(text)} to 100,000 characters for Gemini"
            )

        return f"""
You are an expert at analyzing documents and preparing them for audiobook narration.

**Task**: Analyze this document and prepare it for text-to-speech conversion.

**Input Document** (Markdown format):
```
{text_sample}
```

**Images Detected**: {len(images_metadata)} images found in document{headings_hint}

**Instructions**:

1. **Detect Language** (CRITICAL for TTS):
   - Identify the primary language of the document
   - Return ISO 639-1 code (fr, en, es, de, it, pt, nl, pl, ru, ja, zh, etc.)
   - Field: "language"

2. **Identify document type**:
   - "book" (with chapters)
   - "magazine" (with articles)
   - "article" (single long text)
   - Field: "document_type"

3. **Detect chapters/sections** (CRITICAL - respect document structure):
   - Identify natural break points based on TOP-LEVEL headings only (# Level 1)
   - Use Markdown headings as primary indicators
   - **Books**: Use chapter headings (usually 5-15 chapters)
   - **Magazines**: ALWAYS create 1 chapter per article (NEVER subdivide articles, even if long)
   - **Articles**: Create 1 single chapter (do NOT split unless multiple distinct topics)

   IMPORTANT: For magazines and articles, PRESERVE the original document structure.
   Do NOT artificially split content into smaller chunks for length optimization.

4. **Reformat each chapter for audio narration**:
   - **Remove URLs**: Replace "https://example.com" with "voir lien en description" (French) or "see link in description" (English)
   - **Remove citations**: [1] [2] → "référence" (French) or "reference" (English)
   - **Remove page numbers**: Delete "Page 42" references
   - **Add image descriptions**: Where images appear, insert "L'image montre..." (French) or "The image shows..." (English)
   - **Format footnotes**: Convert to "Note: ..." or "Remarque: ..."
   - **Natural spoken style**: Adapt text for audio narration (conversational tone)

5. **Generate chapter metadata**:
   - Title: Clear chapter title
   - Estimated duration: words / 150 = minutes (assume 150 words/min reading speed)

6. **Suggest TTS voice** based on detected language:
   - French → "ff_siwis" (female) or "fm_brune" (male)
   - English → "af_bella" (female) or "am_adam" (male)
   - Spanish → "sf_isabella" (female) or "sm_diego" (male)
   - German → "gf_heidi" (female) or "gm_klaus" (male)
   - Italian → "if_sofia" (female) or "im_marco" (male)
   - Portuguese → "pf_maria" (female) or "pm_joao" (male)
   - Default → "af_bella" (English female)

**Target Chapter Length** (informational only, NOT a strict constraint):
- Books: Aim for 10-30 minutes per chapter when possible
- Magazines: Chapter length varies by article (do NOT force length limits)
- Articles: Single chapter of any length

NOTE: Chapter length is secondary to respecting document structure.
NEVER artificially split articles or chapters to meet length targets.

**Output Format** (MUST BE VALID JSON):
{{
  "language": "fr",
  "document_type": "book",
  "chapters": [
    {{
      "number": 1,
      "title": "Introduction",
      "text": "Reformatted text ready for narration...",
      "audio_duration_estimate": 180
    }}
  ],
  "images_integrated": 3,
  "tts_voice_suggestion": "ff_siwis"
}}

**CRITICAL**:
- Return ONLY valid JSON, no markdown code blocks
- Language detection is MANDATORY (field "language")
- At least 1 chapter required (field "chapters")
- Image descriptions must match detected language
"""

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse Gemini response (handle JSON in markdown blocks)

        Args:
            response_text: Raw response from Gemini

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        # Helper function to attempt JSON parsing with repair
        def try_parse_with_repair(json_str: str) -> Dict:
            try:
                # Try direct parsing first
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try JSON repair if available
                if JSON_REPAIR_AVAILABLE:
                    logger.warning(f"JSON parsing failed ({e}), attempting repair")
                    try:
                        repaired = repair_json(json_str)
                        result = json.loads(repaired)
                        logger.info("JSON successfully repaired")
                        return result
                    except Exception as repair_error:
                        logger.error(f"JSON repair failed: {repair_error}")
                        raise ValueError(
                            f"Gemini returned malformed JSON: {e}. Repair failed: {repair_error}"
                        )
                else:
                    raise ValueError(f"Gemini returned malformed JSON: {e}")

        try:
            # Try direct JSON parsing first
            return try_parse_with_repair(response_text)
        except ValueError:
            # Try to extract JSON from markdown code block
            logger.warning("Direct JSON parsing failed, trying markdown extraction")

            # Match ```json ... ``` or ``` ... ```
            json_match = re.search(
                r'```(?:json)?\s*\n(.*?)\n```',
                response_text,
                re.DOTALL
            )

            if json_match:
                try:
                    return try_parse_with_repair(json_match.group(1))
                except ValueError as e:
                    logger.error(f"JSON extraction from markdown failed: {e}")
                    raise

            # Last resort: try to find any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return try_parse_with_repair(json_match.group(0))
                except ValueError as e:
                    logger.error(f"JSON extraction from text failed: {e}")
                    raise

            # Complete failure
            logger.error(f"No valid JSON found in response: {response_text[:500]}")
            raise ValueError(
                f"Gemini returned non-JSON response: {response_text[:500]}"
            )

    def _validate_analysis_result(self, result: Dict) -> None:
        """
        Validate Gemini analysis result

        Args:
            result: Parsed JSON result

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Check language field
        if "language" not in result or not result["language"]:
            logger.warning("Language not detected by Gemini, using fallback")
            result["language"] = "en"  # Default to English

        # Check chapters field
        if "chapters" not in result or len(result["chapters"]) == 0:
            raise ValueError("Gemini failed to generate any chapters")

        # Validate each chapter
        for i, chapter in enumerate(result["chapters"]):
            if "number" not in chapter or "title" not in chapter or "text" not in chapter:
                raise ValueError(
                    f"Chapter {i+1} missing required fields (number, title, text)"
                )

            if len(chapter["text"]) < 50:
                logger.warning(
                    f"Chapter {i+1} has very short text ({len(chapter['text'])} chars)"
                )

        # Add default values if missing
        if "document_type" not in result:
            result["document_type"] = "article"
            logger.warning("Document type not detected, defaulting to 'article'")

        if "images_integrated" not in result:
            result["images_integrated"] = 0

        if "tts_voice_suggestion" not in result:
            result["tts_voice_suggestion"] = "af_bella"  # Default voice

    def detect_language_quick(self, text: str) -> str:
        """
        Quick language detection (for fallback)

        Args:
            text: Sample text (first 1000 chars recommended)

        Returns:
            ISO 639-1 language code
        """
        try:
            from langdetect import detect
            lang = detect(text[:1000])
            logger.info(f"Language detected via langdetect: {lang}")
            return lang
        except Exception as e:
            logger.warning(f"langdetect failed: {e}, using Gemini fallback")

            # Fallback: Use Gemini with simplified prompt
            prompt = f"""
Detect the language of this text and return ONLY the ISO 639-1 code (2 letters: fr, en, es, etc.).

Text sample:
{text[:2000]}

Return format: Just the 2-letter code, nothing else.
"""
            try:
                response = self.model.generate_content(prompt)
                lang = response.text.strip().lower()[:2]
                logger.info(f"Language detected via Gemini fallback: {lang}")
                return lang
            except Exception as e2:
                logger.error(f"Gemini language detection fallback failed: {e2}")
                return "en"  # Ultimate fallback

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True
    )
    def extract_metadata(
        self,
        text: str,
        filename: str = None,
        source_url: str = None
    ) -> Dict:
        """
        Extract metadata from content using Gemini (for autofill feature)

        Analyzes text to extract:
        - Title (from content or filename)
        - Author
        - Language (ISO 639-1)
        - Description (1-2 sentence summary)
        - Genre
        - Tags (keywords)
        - Publication date (if available)
        - Voice suggestion based on language

        Args:
            text: Full content text (will be truncated to 50k chars for analysis)
            filename: Optional filename hint (e.g., "deep-learning.pdf")
            source_url: Optional source URL hint (e.g., "https://example.com/article")

        Returns:
            {
                "title": "Extracted title",
                "author": "Author Name",
                "language": "fr",
                "description": "Brief summary...",
                "genre": "Technology",
                "tags": ["AI", "machine-learning", "education"],
                "publication_date": "2025" (or None),
                "voice_suggestion": "ff_siwis"
            }

        Raises:
            Exception: If Gemini API call fails after retries
        """
        # Truncate text to reasonable size for metadata extraction
        text_sample = text[:50000]  # First 50k chars should be enough
        if len(text) > 50000:
            logger.info(f"Text truncated from {len(text)} to 50,000 chars for metadata extraction")

        # Build prompt with hints
        hints = []
        if filename:
            hints.append(f"**Filename**: {filename}")
        if source_url:
            hints.append(f"**Source URL**: {source_url}")

        hints_text = "\n".join(hints) if hints else "(No hints provided)"

        prompt = f"""
You are an expert at analyzing documents and extracting metadata.

**Task**: Extract metadata from this content for podcast/audiobook creation.

**Content to analyze**:
```
{text_sample}
```

**Hints**:
{hints_text}

**Instructions**:
1. **Title**: Extract the main title from the content. If no clear title exists, generate one based on the main topic. Keep it concise (max 100 chars).

2. **Author**: Identify the author if mentioned in the text. If not found, use "Unknown" or the website name (if source_url provided).

3. **Language**: Detect the primary language and return ISO 639-1 code (fr, en, es, de, it, pt, nl, pl, ru, ja, zh, etc.).

4. **Description**: Write a 1-2 sentence summary of the content (max 200 chars).

5. **Genre**: Classify the content genre (Technology, Science, Business, Education, News, Fiction, etc.).

6. **Tags**: Extract 3-7 relevant keywords/topics from the content.

7. **Publication Date**: If a publication date is mentioned, extract it (format: YYYY or YYYY-MM-DD). If not found, return null.

8. **Voice Suggestion**: Based on detected language, suggest appropriate TTS voice:
   - French → "ff_siwis" (female) or "fm_brune" (male)
   - English → "af_bella" (female) or "am_adam" (male)
   - Spanish → "sf_isabella" (female) or "sm_diego" (male)
   - German → "gf_heidi" (female) or "gm_klaus" (male)
   - Italian → "if_sofia" (female) or "im_marco" (male)
   - Portuguese → "pf_maria" (female) or "pm_joao" (male)
   - Default → "af_bella" (English female)

**Output Format** (MUST BE VALID JSON):
{{
  "title": "Clear, concise title",
  "author": "Author Name or Unknown",
  "language": "fr",
  "description": "Brief 1-2 sentence summary",
  "genre": "Technology",
  "tags": ["tag1", "tag2", "tag3"],
  "publication_date": "2025" or null,
  "voice_suggestion": "ff_siwis"
}}

**CRITICAL**:
- Return ONLY valid JSON, no markdown code blocks
- Language detection is MANDATORY (field "language")
- All fields are required (use "Unknown" or null if information not available)
"""

        logger.info(f"Sending text to Gemini for metadata extraction (text_length={len(text_sample)})")

        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini metadata extraction complete, parsing response")

            # Try to parse JSON response
            result = self._parse_gemini_response(response.text)

            # Validate required fields
            required_fields = ["title", "author", "language", "description", "genre", "tags", "voice_suggestion"]
            for field in required_fields:
                if field not in result or result[field] is None:
                    logger.warning(f"Missing field '{field}' in Gemini metadata response, using default")
                    # Provide defaults
                    defaults = {
                        "title": filename or "Untitled",
                        "author": "Unknown",
                        "language": "en",
                        "description": "No description available",
                        "genre": "Unknown",
                        "tags": [],
                        "voice_suggestion": "af_bella"
                    }
                    result[field] = defaults.get(field, "Unknown")

            logger.info(
                f"Metadata extracted: title='{result['title']}', "
                f"author='{result['author']}', language={result['language']}, "
                f"genre={result['genre']}, tags={len(result['tags'])}"
            )

            return result

        except Exception as e:
            logger.exception(f"Gemini metadata extraction failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True
    )
    def upload_pdf_file(self, pdf_path: Path, display_name: str = None):
        """
        Upload PDF file to Gemini File API

        Args:
            pdf_path: Path to PDF file
            display_name: Optional display name for the file

        Returns:
            genai.File object with file URI

        Raises:
            Exception: If upload fails after retries
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logger.info(f"Uploading PDF to Gemini File API: {pdf_path.name} ({file_size_mb:.1f} MB)")

        try:
            # Upload file
            uploaded_file = genai.upload_file(
                path=str(pdf_path),
                display_name=display_name or pdf_path.name
            )

            logger.info(f"PDF uploaded successfully: {uploaded_file.name} (URI: {uploaded_file.uri})")

            # Wait for file to be processed (ACTIVE state)
            while uploaded_file.state.name == "PROCESSING":
                logger.debug(f"Waiting for file processing... (state: {uploaded_file.state.name})")
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name != "ACTIVE":
                raise Exception(f"File processing failed: {uploaded_file.state.name}")

            logger.info(f"File ready for processing (state: {uploaded_file.state.name})")
            return uploaded_file

        except Exception as e:
            logger.exception(f"PDF upload failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True
    )
    def analyze_pdf_document(
        self,
        pdf_file,
        metadata_hint: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze PDF document using Gemini File API (replaces Docling-based analysis)

        Args:
            pdf_file: Uploaded genai.File object
            metadata_hint: Optional hints (title, author, etc.)

        Returns:
            {
                "language": "fr" | "en" | "es" | ...,
                "document_type": "book" | "magazine" | "article",
                "chapters": [
                    {
                        "number": 1,
                        "title": "Introduction",
                        "text": "Reformatted chapter text...",
                        "audio_duration_estimate": 120
                    },
                    ...
                ],
                "images_integrated": 0,  # Gemini handles images natively
                "tts_voice_suggestion": "ff_siwis"
            }

        Raises:
            Exception: If Gemini API call fails after retries
        """
        prompt = self._build_pdf_analysis_prompt(metadata_hint)

        logger.info(f"Sending PDF to Gemini for analysis (file: {pdf_file.name})")

        try:
            # Generate content with PDF file
            response = self.model.generate_content([pdf_file, prompt])
            logger.info("Gemini PDF analysis complete, parsing response")

            # Parse JSON response
            result = self._parse_gemini_response(response.text)

            # Validate required fields
            self._validate_analysis_result(result)

            logger.info(
                f"PDF analysis successful: language={result['language']}, "
                f"type={result['document_type']}, chapters={len(result['chapters'])}"
            )

            return result

        except Exception as e:
            logger.exception(f"Gemini PDF analysis failed: {e}")
            raise

    def _build_pdf_analysis_prompt(self, metadata_hint: Optional[Dict]) -> str:
        """
        Generate prompt for PDF analysis via Gemini File API

        Args:
            metadata_hint: Optional hints (title, author, etc.)

        Returns:
            Prompt string for Gemini
        """
        # Add metadata hints if available
        hints_text = ""
        if metadata_hint:
            hints_parts = []
            if "title" in metadata_hint and metadata_hint["title"]:
                hints_parts.append(f"**Title**: {metadata_hint['title']}")
            if "author" in metadata_hint and metadata_hint["author"]:
                hints_parts.append(f"**Author**: {metadata_hint['author']}")
            if hints_parts:
                hints_text = "\n".join(hints_parts) + "\n\n"

        return f"""
You are an expert at analyzing documents and preparing them for audiobook narration.

**Task**: Analyze this PDF document and prepare it for text-to-speech conversion.

{hints_text}**Instructions**:

1. **Detect Language** (CRITICAL for TTS):
   - Identify the primary language of the document
   - Return ISO 639-1 code (fr, en, es, de, it, pt, nl, pl, ru, ja, zh, etc.)
   - Field: "language"

2. **Identify document type**:
   - "book" (with chapters)
   - "magazine" (with articles)
   - "article" (single long text)
   - Field: "document_type"

3. **Detect chapters/sections** (CRITICAL - respect document structure):
   - Identify natural break points based on TOP-LEVEL headings only
   - Use document structure (chapters, articles, sections) as primary indicators
   - **Books**: Use chapter headings (usually 5-15 chapters)
   - **Magazines**: ALWAYS create 1 chapter per article (NEVER subdivide articles, even if long)
   - **Articles**: Create 1 single chapter (do NOT split unless multiple distinct topics)

   IMPORTANT: For magazines and articles, PRESERVE the original document structure.
   Do NOT artificially split content into smaller chunks for length optimization.

4. **Reformat each chapter for audio narration**:
   - **Remove URLs**: Replace "https://example.com" with "voir lien en description" (French) or "see link in description" (English)
   - **Remove citations**: [1] [2] → "référence" (French) or "reference" (English)
   - **Remove page numbers**: Delete "Page 42" references
   - **Describe images**: Where images appear, insert natural descriptions like "L'image montre..." (French) or "The image shows..." (English)
   - **Format footnotes**: Convert to "Note: ..." or "Remarque: ..."
   - **Natural spoken style**: Adapt text for audio narration (conversational tone)

5. **Generate chapter metadata**:
   - Title: Clear chapter title
   - Estimated duration: words / 150 = minutes (assume 150 words/min reading speed)

6. **Suggest TTS voice** based on detected language:
   - French → "ff_siwis" (female) or "fm_brune" (male)
   - English → "af_bella" (female) or "am_adam" (male)
   - Spanish → "sf_isabella" (female) or "sm_diego" (male)
   - German → "gf_heidi" (female) or "gm_klaus" (male)
   - Italian → "if_sofia" (female) or "im_marco" (male)
   - Portuguese → "pf_maria" (female) or "pm_joao" (male)
   - Default → "af_bella" (English female)

**Target Chapter Length** (informational only, NOT a strict constraint):
- Books: Aim for 10-30 minutes per chapter when possible
- Magazines: Chapter length varies by article (do NOT force length limits)
- Articles: Single chapter of any length

NOTE: Chapter length is secondary to respecting document structure.
NEVER artificially split articles or chapters to meet length targets.

**Output Format** (MUST BE VALID JSON):
{{
  "language": "fr",
  "document_type": "book",
  "chapters": [
    {{
      "number": 1,
      "title": "Introduction",
      "text": "Reformatted text ready for narration...",
      "audio_duration_estimate": 180
    }}
  ],
  "images_integrated": 0,
  "tts_voice_suggestion": "ff_siwis"
}}

**CRITICAL**:
- Return ONLY valid JSON, no markdown code blocks
- Language detection is MANDATORY (field "language")
- At least 1 chapter required (field "chapters")
- Image descriptions must match detected language
- Preserve document structure (chapters, articles, sections)
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True
    )
    def extract_metadata_from_pdf(
        self,
        pdf_file,
        filename: str = None,
        source_url: str = None
    ) -> Dict:
        """
        Extract metadata from PDF using Gemini File API (for autofill feature)

        Args:
            pdf_file: Uploaded genai.File object
            filename: Optional filename hint

        Returns:
            {
                "title": "Extracted title",
                "author": "Author Name",
                "language": "fr",
                "description": "Brief summary...",
                "genre": "Technology",
                "tags": ["AI", "machine-learning"],
                "publication_date": "2025" or null,
                "voice_suggestion": "ff_siwis"
            }

        Raises:
            Exception: If Gemini API call fails after retries
        """
        hints_text = f"**Filename**: {filename}" if filename else "(No hints provided)"

        prompt = f"""
You are an expert at analyzing documents and extracting metadata.

**Task**: Extract metadata from this PDF document for podcast/audiobook creation.

**Hints**:
{hints_text}

**Instructions**:
1. **Title**: Extract the main title from the PDF. If no clear title exists, generate one based on the main topic. Keep it concise (max 100 chars).

2. **Author**: Identify the author if mentioned in the PDF. If not found, use "Unknown".

3. **Language**: Detect the primary language and return ISO 639-1 code (fr, en, es, de, it, pt, nl, pl, ru, ja, zh, etc.).

4. **Description**: Write a 1-2 sentence summary of the content (max 200 chars).

5. **Genre**: Classify the content genre (Technology, Science, Business, Education, News, Fiction, etc.).

6. **Tags**: Extract 3-7 relevant keywords/topics from the content.

7. **Publication Date**: If a publication date is mentioned, extract it (format: YYYY or YYYY-MM-DD). If not found, return null.

8. **Voice Suggestion**: Based on detected language, suggest appropriate TTS voice:
   - French → "ff_siwis" (female) or "fm_brune" (male)
   - English → "af_bella" (female) or "am_adam" (male)
   - Spanish → "sf_isabella" (female) or "sm_diego" (male)
   - German → "gf_heidi" (female) or "gm_klaus" (male)
   - Italian → "if_sofia" (female) or "im_marco" (male)
   - Portuguese → "pf_maria" (female) or "pm_joao" (male)
   - Default → "af_bella" (English female)

**Output Format** (MUST BE VALID JSON):
{{
  "title": "Clear, concise title",
  "author": "Author Name or Unknown",
  "language": "fr",
  "description": "Brief 1-2 sentence summary",
  "genre": "Technology",
  "tags": ["tag1", "tag2", "tag3"],
  "publication_date": "2025" or null,
  "voice_suggestion": "ff_siwis"
}}

**CRITICAL**:
- Return ONLY valid JSON, no markdown code blocks
- Language detection is MANDATORY (field "language")
- All fields are required (use "Unknown" or null if information not available)
"""

        logger.info(f"Sending PDF to Gemini for metadata extraction (file: {pdf_file.name})")

        try:
            response = self.model.generate_content([pdf_file, prompt])
            logger.info("Gemini PDF metadata extraction complete, parsing response")

            # Parse JSON response
            result = self._parse_gemini_response(response.text)

            # Validate required fields
            required_fields = ["title", "author", "language", "description", "genre", "tags", "voice_suggestion"]
            for field in required_fields:
                if field not in result or result[field] is None:
                    logger.warning(f"Missing field '{field}' in Gemini PDF metadata response, using default")
                    defaults = {
                        "title": filename or "Untitled",
                        "author": "Unknown",
                        "language": "en",
                        "description": "No description available",
                        "genre": "Unknown",
                        "tags": [],
                        "voice_suggestion": "af_bella"
                    }
                    result[field] = defaults.get(field, "Unknown")

            logger.info(
                f"PDF metadata extracted: title='{result['title']}', "
                f"author='{result['author']}', language={result['language']}, "
                f"genre={result['genre']}, tags={len(result['tags'])}"
            )

            return result

        except Exception as e:
            logger.exception(f"Gemini PDF metadata extraction failed: {e}")
            raise
