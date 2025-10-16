"""
Gemini API Client for Document Analysis
Handles chapter detection, language detection, and audio-friendly reformatting
"""

import google.generativeai as genai
from typing import List, Dict, Optional
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import re

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

3. **Detect chapters/sections**:
   - Identify natural break points based on headings and structure
   - Use Markdown headings (# Chapter, ## Section) as primary indicators
   - Books: 5-15 chapters typical
   - Magazines: 1 chapter per article
   - Articles: Split by H2 headings or create 10-minute logical chunks

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

**Target Chapter Length**:
- 5-20 minutes audio (750-3000 words per chapter)
- If no clear structure: create logical 10-minute chunks

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
