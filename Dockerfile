# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with increased timeout and retry
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/chroma_db /app/documents /app/uploaded_documents

# Health check for the application
HEALTHCHECK CMD python -c "import rag_system; print('healthy')" || exit 1

# Default command - interactive RAG system
CMD ["python", "simple_rag.py"]
