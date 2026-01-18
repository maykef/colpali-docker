# Use exact same base as original
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Keep original ENV settings
ENV DEBIAN_FRONTEND=interactive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Identical system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    nano \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# PyTorch Nightly - EXACT same command
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Build helpers
RUN pip3 install packaging setuptools wheel

# [cite_start]ColPali + Vision Stack - WITH PROTOBUF [cite: 2]
RUN pip3 install --no-cache-dir \
    colpali-engine \
    transformers \
    accelerate \
    einops \
    bitsandbytes \
    pdf2image \
    qwen-vl-utils \
    sentencepiece \
    pillow \
    tqdm \
    protobuf

# Flash Attention
RUN pip3 install flash-attn --no-build-isolation

# PRE-DOWNLOAD ColPali MODEL
RUN python3 -c "from colpali_engine.models import ColPali, ColPaliProcessor; \
    print('Downloading ColPali...'); \
    ColPali.from_pretrained('vidore/colpali-v1.2', torch_dtype='float32', device_map='cpu'); \
    ColPaliProcessor.from_pretrained('vidore/colpali-v1.2'); \
    print('✅ ColPali cached')"

# Hardware check script
RUN echo 'import torch; \
import flash_attn; \
import qwen_vl_utils; \
name = torch.cuda.get_device_name(0); \
vram = torch.cuda.get_device_properties(0).total_memory / 1e9; \
print("-" * 40); \
print(f"✅ DEVICE: {name}"); \
print(f"✅ VRAM: {vram:.1f} GB"); \
print(f"✅ FLASH ATTENTION: LOADED"); \
print(f"✅ QWEN UTILS: LOADED"); \
print("-" * 40)' > check_hw.py

CMD ["python3", "check_hw.py"]
