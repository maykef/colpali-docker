# 1. Use the version that identifies Blackwell hardware correctly
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=interactive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Install system essentials
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip 
RUN pip3 install --upgrade pip

# 4. Install PyTorch Nightly - The only version with native sm_120 (Blackwell) support
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 5. Install build helpers
RUN pip3 install packaging setuptools wheel

# 6. Install ColPali + Vision Stack (Including qwen-vl-utils)
RUN pip3 install --no-cache-dir \
    colpali-engine \
    transformers \
    accelerate \
    einops \
    bitsandbytes \
    pdf2image \
    qwen-vl-utils \
    sentencepiece \
    pillow

# 7. Install Flash Attention (Uses pre-compiled wheels for this environment)
RUN pip3 install flash-attn --no-build-isolation

# 8. Create a verification script inside the image
RUN echo 'import torch; \
import flash_attn; \
import qwen_vl_utils; \
name = torch.cuda.get_device_name(0); \
print("-" * 30); \
print(f"✅ DEVICE: {name}"); \
print(f"✅ FLASH ATTENTION: LOADED"); \
print(f"✅ QWEN UTILS: LOADED"); \
print("-" * 30)' > check_hw.py

CMD ["python3", "check_hw.py"]
