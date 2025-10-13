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
    echo "✅ System dependencies installed successfully"

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create directories for temporary files and outputs
RUN mkdir -p /data/shared/podcasts/{jobs,final,chunks} && \
    chmod -R 777 /data/shared/podcasts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
