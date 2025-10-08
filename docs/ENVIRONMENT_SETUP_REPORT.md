# 🧩 Environment Setup & Limitations Report
**Project:** HaMeR Development Environment  
**Last Updated:** 2025-10-08
**Base Environment:** `nvidia/cuda:12.6.2-devel-ubuntu22.04`

---

## 1. Overview

This document describes the environment setup, dependency constraints, and compatibility limitations discovered during HaMeR development.  
It serves as both a reproducibility guide and a record of known pitfalls related to **NumPy**, **OpenCV**, **xtcocotools**, and **Pillow** version conflicts.

---

## 2. Key Environment Goals

- Provide a **GPU-enabled** Docker-based development environment using CUDA 12.6.
- Ensure **reproducible builds** with controlled versions of key packages.
- Maintain compatibility between PyTorch, NumPy, OpenCV, Pillow, and third-party libraries such as `xtcocotools` and `ViTPose`.
- Avoid version conflicts during pip installation (notably with `numpy>=2.0`, `opencv-python>=4.12`, and `pillow>=10`).

---

## 3. Environment Architecture

### 3.1 Docker Compose

```yaml
services:
  hamer-dev:
    build:
      context: ../
      dockerfile: ./docker/hamer-dev.Dockerfile
    volumes:
      - ../:/app
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

This enables GPU passthrough and local code mounting for live development.

---

## 4. Dockerfile Breakdown

### 4.1 Base Image
```dockerfile
ARG BASE=nvidia/cuda:12.6.2-devel-ubuntu22.04
FROM ${BASE} AS hamer
```
- Provides a CUDA 12.6 runtime and build tools for PyTorch.
- Ubuntu 22.04 ensures long-term support compatibility.

### 4.2 System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing     gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel     espeak-ng libsndfile1-dev git wget ffmpeg libsm6 libxext6     libglfw3-dev libgles2-mesa-dev     && rm -rf /var/lib/apt/lists/*
```
Installed core development utilities, Python toolchain, audio libraries, and OpenGL dependencies.

### 4.3 Python Virtual Environment
```dockerfile
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```
Prevents global pollution and allows isolated pip management.

### 4.4 Dependency Installation Strategy

Order of installation is **critical**:

1. **Upgrade build tools**:
   ```bash
   pip install --upgrade wheel setuptools
   ```

2. **Pin NumPy before any other dependency**:
   ```bash
   pip install "numpy>=1,<2"
   ```
   ✅ Prevents incompatibility with:
   - `xtcocotools` (fails with `numpy>=2`)
   - `opencv-python>=4.12`
   - `mmcv==1.3.9` (compiled against older NumPy C-API)

3. **Install compatible OpenCV**:
   ```bash
   pip install "opencv-python<4.12.0.88"
   ```
   ✅ Ensures binary compatibility with NumPy 1.x ABI.

4. **Install PyTorch (CUDA 11.8 build)**:
   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
   ```
   Although the base image supports CUDA 12, the project dependencies rely on the **11.8-compatible** torch wheel.

5. **Project & third-party dependencies**:
   ```bash
   pip install -e .[all]
   pip install "Pillow<10"
   pip install -v -e third-party/ViTPose
   ```

---

## 5. Python Package Configuration (`setup.py`)

### 5.1 Strict Version Pins
- **NumPy 1.26.4** — last stable 1.x version; avoids API breaks in NumPy 2.x.  
- **xtcocotools 1.14.3** — compiled C extensions depend on NumPy 1.x headers.  
- **OpenCV 4.11.0.86** — last version built against NumPy 1.x ABI.  
- **Pillow <10** — ensures compatibility with `pyrender` and older `imageio`.  
- **PyTorch 2.2.0 + torchvision 0.17.0 (cu118)** — stable for CUDA 11.8 runtime.  
- **mmcv 1.3.9** — older MMCV builds fail with NumPy 2.x.  

### 5.2 Git-based Dependencies
- `detectron2` and `chumpy` are installed directly from GitHub due to missing up-to-date wheels.

---

## 6. Known Limitations & Fixes

| Issue | Symptom | Resolution |
|-------|----------|-------------|
| **NumPy ≥2.0** | `ValueError: numpy.core.multiarray failed to import` | Pin to `<2.0` |
| **OpenCV ≥4.12** | Segfaults or missing symbols on import | Pin `<4.12.0.88` |
| **xtcocotools compile errors** | `#include <numpy/arrayobject.h>` fails | Install NumPy 1.26.x **before** xtcocotools |
| **Pillow ≥10** | `ImportError: cannot import name 'ImageResampling'` | Pin `<10` |
| **torch/cu mismatch** | Runtime error: “invalid device function” | Use `torch==2.2.0+cu118` |

---

## 7. Build Procedure

### Step-by-step

```bash
# Build the image
docker compose build hamer-dev

# Run interactive shell
docker compose run --rm hamer-dev bash

# Inside container
source /opt/venv/bin/activate
python -m pip list
python -c "import torch; print(torch.cuda.is_available())"
```

---


## 7.1 Running HaMeR with Docker Compose

If you wish to use HaMeR with Docker, you can build and launch the container as follows:

```bash
# Build and start the container
docker compose -f ./docker/docker-compose.yml up -d

# Enter the running container
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

Once inside the container, you can proceed with the setup or run the demo as follows:

```bash
# Fetch demo assets
bash fetch_demo_data.sh

# Run demo inference
python demo.py     --img_folder example_data --out_folder demo_out     --batch_size=48 --side_view --save_mesh --full_frame
```

The above commands verify that the full HaMeR environment and GPU pipeline are functioning correctly.

Expected output:
- Processed results under `demo_out/`
- Rendered meshes and side-view images if flags are enabled.
