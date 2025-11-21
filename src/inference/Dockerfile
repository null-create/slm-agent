# Stage 1: Build the UI
FROM node:18-alpine AS ui-builder

WORKDIR /ui

# Copy UI package files
COPY ui/package*.json ./

# Install dependencies
RUN npm ci

# Copy UI source files
COPY ui/ ./

# Build the UI
RUN npm run build

# Stage 2: Main application with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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

# Install PyTorch with CUDA support first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention with specific CUDA support
RUN pip install flash-attn --no-build-isolation

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache /app/ui/dist

# Set HuggingFace cache directory
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/cache/huggingface/datasets

# Set Weights & Biases cache directory
ENV WANDB_DIR=/app/logs/wandb

# Copy built UI from the first stage
COPY --from=ui-builder /ui/dist /app/ui/dist

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "App User" appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Expose ports
# 8000 - Flask API server
# 3000 - UI (if serving separately, optional)
EXPOSE 3000 8000 9191

# Set default environment variables
ENV MODEL_NAME=""
ENV MODEL_PATH="/app/model"
ENV DATA_PATH="/app/data"
ENV LOG_LEVEL="INFO"
ENV DEVICE="cuda"
ENV UI_DIST_PATH="/app/ui/dist"

# Default command
CMD ["python", "server.py", "--model-info", "model/microsoft/Phi-3.5-mini-instruct/model.json", "--host", "0.0.0.0", "--port", "9191"]