"""
Podcast Engine - Chunking Tests
Tests for text chunking and preprocessing
"""
import pytest
from app.core.chunking import TextChunker


class TestTextChunker:
    """Tests for TextChunker class"""

    def test_basic_chunking(self, sample_text_short):
        """Test basic text chunking"""
        chunker = TextChunker(max_chunk_size=100)
        chunks = chunker.create_chunks(sample_text_short)

        assert len(chunks) > 0, "No chunks generated"
        assert all(isinstance(chunk, tuple) for chunk in chunks), "Chunks should be tuples"
        assert all(len(chunk) == 3 for chunk in chunks), "Chunks should have 3 elements (id, text, chapter)"

    def test_sentence_aware_splitting(self):
        """Test that chunking respects sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = TextChunker(max_chunk_size=40, preserve_sentence=True)
        chunks = chunker.create_chunks(text)

        # Each chunk should contain complete sentences
        for chunk_id, chunk_text, _ in chunks:
            # Should not split mid-sentence
            assert chunk_text.count(".") > 0, f"Chunk {chunk_id} has no sentence ending"
            # Should not have incomplete sentences (unless very long)
            if len(chunk_text) < 1000:
                assert not chunk_text.endswith(" "), "Chunk should not end with incomplete sentence"

    def test_chunk_size_limits(self, sample_text_long):
        """Test that chunks respect max size"""
        max_size = 500
        chunker = TextChunker(max_chunk_size=max_size)
        chunks = chunker.create_chunks(sample_text_long)

        for chunk_id, chunk_text, _ in chunks:
            # Allow some tolerance for long sentences
            assert len(chunk_text) <= max_size * 2, f"Chunk {chunk_id} exceeds max size by too much"

    def test_markdown_removal(self):
        """Test markdown syntax removal"""
        text = "**Bold text** and *italic text* and `code` here."
        chunker = TextChunker(remove_markdown=True)
        chunks = chunker.create_chunks(text)

        chunk_text = chunks[0][1]
        assert "**" not in chunk_text, "Bold markers not removed"
        assert "*" not in chunk_text, "Italic markers not removed"
        assert "`" not in chunk_text, "Code markers not removed"

    def test_url_removal(self):
        """Test URL removal from text"""
        text = "Check this link https://example.com and http://test.com for more info."
        chunker = TextChunker(remove_urls=True)
        chunks = chunker.create_chunks(text)

        chunk_text = chunks[0][1]
        assert "https://" not in chunk_text, "HTTPS URL not removed"
        assert "http://" not in chunk_text, "HTTP URL not removed"
        assert "example.com" not in chunk_text, "URL domain not removed"

    def test_chapter_detection(self, sample_text_long):
        """Test chapter marker detection"""
        chunker = TextChunker()
        chunks = chunker.create_chunks(sample_text_long, add_chapter_markers=True)

        # Should detect "Chapter 1", "Chapter 2", etc.
        chapter_chunks = [c for c in chunks if c[2]]  # chapter_title not empty
        assert len(chapter_chunks) > 0, "No chapters detected"

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        chunker = TextChunker()
        chunks = chunker.create_chunks("")

        assert len(chunks) == 0, "Empty text should produce no chunks"

    def test_whitespace_normalization(self):
        """Test that multiple whitespaces are normalized"""
        text = "Multiple    spaces    between    words."
        chunker = TextChunker()
        chunks = chunker.create_chunks(text)

        chunk_text = chunks[0][1]
        assert "    " not in chunk_text, "Multiple spaces not normalized"
        assert "  " not in chunk_text, "Double spaces not normalized"

    def test_minimum_chunk_size(self):
        """Test that very small chunks are avoided"""
        text = "A. B. C. D. E."  # Very short sentences
        chunker = TextChunker(max_chunk_size=100, preserve_sentence=True)
        chunks = chunker.create_chunks(text)

        # Should combine short sentences
        assert len(chunks) < 5, "Too many chunks for short sentences"

    def test_long_sentence_handling(self):
        """Test handling of extremely long sentences"""
        # Create a very long sentence without punctuation
        long_word = "word" * 500  # 2000 chars
        text = f"This is a sentence with a {long_word} in it."

        chunker = TextChunker(max_chunk_size=1000)
        chunks = chunker.create_chunks(text)

        # Should still split even without sentence boundaries
        assert len(chunks) > 1, "Long sentence should be split"


class TestTextPreprocessing:
    """Tests for text preprocessing"""

    def test_normalize_whitespace(self):
        """Test whitespace normalization"""
        chunker = TextChunker()
        text = "Text\twith\ttabs\nand\nnewlines"
        processed = chunker.preprocess_text(text)

        assert "\t" not in processed, "Tabs not removed"
        assert "\n" not in processed, "Newlines not removed"

    def test_markdown_headers(self):
        """Test markdown header removal"""
        chunker = TextChunker(remove_markdown=True)
        text = "# Header 1\n## Header 2\n### Header 3"
        processed = chunker.preprocess_text(text)

        assert "#" not in processed, "Markdown headers not removed"

    def test_preserve_essential_punctuation(self):
        """Test that essential punctuation is preserved"""
        chunker = TextChunker(remove_markdown=False)
        text = "Hello, world! How are you? I'm fine."
        processed = chunker.preprocess_text(text)

        assert "," in processed, "Comma removed incorrectly"
        assert "!" in processed, "Exclamation removed incorrectly"
        assert "?" in processed, "Question mark removed incorrectly"
        assert "'" in processed, "Apostrophe removed incorrectly"
