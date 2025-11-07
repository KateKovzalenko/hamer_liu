# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
ARG BASE=nvidia/cuda:12.6.2-devel-ubuntu22.04
FROM ${BASE}

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential cmake git wget ffmpeg \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    libsm6 libxext6 libglfw3-dev libgles2-mesa-dev \
    libgl1 libglib2.0-0 \
    libjpeg-dev zlib1g-dev libpng-dev \
    libtiff5-dev libfreetype6-dev \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python virtual environment
# ------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ------------------------------------------------------------
# Set working directory and copy all project files
# ------------------------------------------------------------
WORKDIR /app

# Activate virtual environment and install dependencies:
# REVIEW: We need to install/upgrade wheel and setuptools first because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade wheel setuptools

# Install base Python packages one by one
RUN pip install "numpy>=1,<2"
RUN pip install "opencv-python<4.12.0.88"
RUN pip install "gdown==5.2.0"
RUN pip install "Flask==2.3.3"
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Consider removing when switching to HaMeR
RUN pip install "opencv-contrib-python==4.11.0.86"
RUN pip install "mediapipe==0.10.7"

# ------------------------------------------------------------
# Install VS Code debugger (debugpy)
# ------------------------------------------------------------
RUN pip install debugpy

COPY . .

# ------------------------------------------------------------
# Install HaMeR in editable mode
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .[all]

# ------------------------------------------------------------
# Install Pillow after editable install
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "Pillow<10"

# ------------------------------------------------------------
# Install third-party dependencies (ViTPose)
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -v -e third-party/ViTPose

# ------------------------------------------------------------
# Expose server port and run server
# ------------------------------------------------------------
EXPOSE 5000
#CMD ["python", "-m", "server.server"]
