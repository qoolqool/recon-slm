# Dockerfile for Report Reconciliation SLM Training
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/torch_cache
ENV HF_HOME=/app/huggingface_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directories
RUN mkdir -p /app/torch_cache /app/huggingface_cache

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Set permissions
RUN chmod +x *.py

# Default command
CMD ["python", "run_reconciliation_training.py"]
