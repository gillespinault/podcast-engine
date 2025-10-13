"""
Podcast Engine - Smart Text Chunking
Intelligent text splitting for optimal TTS processing
"""
import re
from typing import List, Tuple
from loguru import logger


class TextChunker:
    """Smart text chunking with sentence preservation"""

    def __init__(
        self,
        max_chunk_size: int = 4000,
        preserve_sentence: bool = True,
        remove_urls: bool = True,
        remove_markdown: bool = True,
    ):
        """
        Initialize text chunker

        Args:
            max_chunk_size: Maximum characters per chunk
            preserve_sentence: Split at sentence boundaries
            remove_urls: Remove URLs from text
            remove_markdown: Strip markdown formatting
        """
        self.max_chunk_size = max_chunk_size
        self.preserve_sentence = preserve_sentence
        self.remove_urls = remove_urls
        self.remove_markdown = remove_markdown

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text before chunking

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        # Remove URLs
        if self.remove_urls:
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '',
                text
            )

        # Remove markdown formatting
        if self.remove_markdown:
            # Remove bold/italic
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'__(.+?)__', r'\1', text)
            text = re.sub(r'_(.+?)_', r'\1', text)

            # Remove links [text](url)
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

            # Remove headers
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

            # Remove code blocks
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            text = re.sub(r'`(.+?)`', r'\1', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Sentence boundaries: . ! ? followed by space and capital letter
        # But preserve abbreviations like "Mr.", "Dr.", "etc."
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def create_chunks(self, text: str, add_chapter_markers: bool = True) -> List[Tuple[int, str, str]]:
        """
        Create smart chunks from text

        Args:
            text: Input text
            add_chapter_markers: Add "Part X/Y" markers

        Returns:
            List of tuples: (chunk_id, text, chapter_title)
        """
        # Preprocess
        text = self.preprocess_text(text)

        if not text:
            logger.warning("Empty text after preprocessing")
            return []

        chunks = []

        if self.preserve_sentence:
            # Split by sentences for better quality
            sentences = self.split_into_sentences(text)

            current_chunk = ""
            chunk_id = 0

            for sentence in sentences:
                # Check if adding this sentence would exceed limit
                if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                    # Save current chunk if not empty
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        chunk_id += 1

                    # If single sentence is too long, split it
                    if len(sentence) > self.max_chunk_size:
                        # Split long sentence by commas or semicolons
                        parts = re.split(r'[,;]\s+', sentence)
                        sub_chunk = ""

                        for part in parts:
                            if len(sub_chunk) + len(part) + 2 > self.max_chunk_size:
                                if sub_chunk:
                                    chunks.append(sub_chunk.strip())
                                    chunk_id += 1
                                sub_chunk = part
                            else:
                                sub_chunk += (", " if sub_chunk else "") + part

                        if sub_chunk:
                            current_chunk = sub_chunk
                    else:
                        current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    current_chunk += (" " if current_chunk else "") + sentence

            # Add last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

        else:
            # Simple split by max_chunk_size (no sentence preservation)
            for i in range(0, len(text), self.max_chunk_size):
                chunk = text[i:i + self.max_chunk_size]
                chunks.append(chunk.strip())

        # Create final chunk list with metadata
        total_chunks = len(chunks)
        result = []

        for idx, chunk_text in enumerate(chunks):
            chunk_id = idx

            if add_chapter_markers and total_chunks > 1:
                chapter_title = f"Part {idx + 1} of {total_chunks}"
            else:
                chapter_title = ""

            result.append((chunk_id, chunk_text, chapter_title))

        logger.info(f"Created {total_chunks} chunks from {len(text)} chars")

        return result

    def estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """
        Estimate audio duration in seconds

        Args:
            text: Input text
            speed: Speech speed multiplier

        Returns:
            Estimated duration in seconds
        """
        # Average speaking rate: ~150 words per minute (2.5 words/second)
        # Adjust for speed
        words = len(text.split())
        words_per_second = 2.5 * speed
        duration = words / words_per_second

        return duration

    def get_stats(self, text: str) -> dict:
        """
        Get text statistics

        Args:
            text: Input text

        Returns:
            Statistics dictionary
        """
        processed = self.preprocess_text(text)
        chunks = self.create_chunks(text, add_chapter_markers=False)

        return {
            "original_length": len(text),
            "processed_length": len(processed),
            "total_words": len(text.split()),
            "total_chunks": len(chunks),
            "average_chunk_size": sum(len(c[1]) for c in chunks) / len(chunks) if chunks else 0,
            "estimated_duration_minutes": self.estimate_duration(text) / 60,
        }


# Example usage
if __name__ == "__main__":
    from loguru import logger

    logger.add("chunking.log", rotation="10 MB")

    sample_text = """
    This is a sample article about artificial intelligence. AI has revolutionized many industries.

    Machine learning is a subset of AI. It involves training models on data. Deep learning uses neural networks.

    **Benefits of AI:**
    - Automation
    - Efficiency
    - Accuracy

    Visit https://example.com for more info.

    In conclusion, AI is transforming our world. The future looks promising.
    """

    chunker = TextChunker(max_chunk_size=100, preserve_sentence=True)

    # Get stats
    stats = chunker.get_stats(sample_text)
    print("Stats:", stats)

    # Create chunks
    chunks = chunker.create_chunks(sample_text, add_chapter_markers=True)

    for chunk_id, text, chapter in chunks:
        print(f"\n--- Chunk {chunk_id} ---")
        if chapter:
            print(f"Chapter: {chapter}")
        print(f"Text: {text}")
        print(f"Length: {len(text)} chars")
