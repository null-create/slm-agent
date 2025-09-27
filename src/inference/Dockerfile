# Use NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  python3-pip \
  git \
  wget \
  curl \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  libffi-dev \
  libssl-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev \
  && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
  ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA support first (this ensures proper CUDA compatibility)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention with specific CUDA support
RUN pip install flash-attn --no-build-isolation

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache

# Set HuggingFace cache directory
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/cache/huggingface/datasets

# Set Weights & Biases cache directory
ENV WANDB_DIR=/app/logs/wandb

# Create a non-root user for security and change ownership of app directory to appuser
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "App User" appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code (do this after switching user to maintain ownership)
COPY --chown=appuser:appuser ./src .
COPY --chown=appuser:appuser ./scripts .

EXPOSE 8000 8080

# Set default environment variables that can be overridden
ENV MODEL_NAME=""
ENV MODEL_PATH="/app/models"
ENV DATA_PATH="/app/data"
ENV LOG_LEVEL="INFO"
ENV DEVICE="cuda"

# Health check (adjust the endpoint as needed for your application)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command (adjust as needed for your application entry point)
CMD ["python", "main.py"]