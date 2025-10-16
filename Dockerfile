# Podcast Engine - Docker Image
# FastAPI service for podcast generation from text
# Includes: ffmpeg, poppler-utils, Python 3.11

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio/Video processing
    ffmpeg \
    # PDF text extraction
    poppler-utils \
    # OCR support (for scanned PDFs via Docling)
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-nld \
    # Utilities
    curl \
    wget \
    # Build tools (for some Python packages)
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Verify critical tools installation
RUN ffmpeg -version && \
    pdftotext -v && \
    tesseract --version && \
    echo "✅ System dependencies installed successfully (ffmpeg, poppler-utils, tesseract)"

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Copy entrypoint scripts
COPY entrypoint.sh /app/
COPY worker_entrypoint.py /app/

# Create directories for temporary files and outputs
RUN mkdir -p /data/shared/podcasts/{jobs,final,chunks} && \
    chmod -R 777 /data/shared/podcasts && \
    chmod +x /app/entrypoint.sh

# Health check (disabled temporarily - workers may interfere)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run API + Workers via entrypoint
CMD ["/app/entrypoint.sh"]
