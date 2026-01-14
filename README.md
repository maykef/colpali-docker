ColPali Research Engine (Blackwell-Optimized)

A high-performance Vision-RAG (Retrieval-Augmented Generation) pipeline optimized for NVIDIA Blackwell Architecture (RTX 6000 / sm_120). This project leverages the latest CUDA 12.6 toolkit and PyTorch Nightly to provide a stable, source-build-free environment for document AI.
🚀 Key Features

    Blackwell Support: Pre-configured for sm_120 GPUs using CUDA 12.6.

    Flash Attention 2: High-speed attention kernels enabled for vision-document processing.

    Vision-RAG Stack: Integrated with colpali-engine, transformers, and qwen-vl-utils.

    Dockerized Workflow: Entire research environment encapsulated for reproducibility.

🛠️ Setup & Installation
1. Prerequisites

    NVIDIA Driver 560+ installed on the host.

    Docker and NVIDIA Container Toolkit installed.

    An NVMe drive (recommended) mounted at /mnt/nvme8tb/RAG_Clean.

2. Initialization

Clone the repository and run the management script to build the image and launch your first container:
Bash

git clone https://github.com/YOUR_USERNAME/ColPali-Research-Engine.git
cd ColPali-Research-Engine
chmod +x manage_env.sh
./manage_env.sh

🕹️ Managing the Docker Container

Once the environment is built, use these commands to control your "Research Node."
How to Enter the Container

If you have already run the setup script and are currently at your host terminal, jump back into the active container:
Bash

docker exec -it colpali-research-node /bin/bash

How to Stop the Container

To free up GPU memory or pause your work:
Bash

docker stop colpali-research-node

How to Start a Stopped Container

If you've rebooted your machine or stopped the container previously:
Bash

docker start colpali-research-node
# Then enter it:
docker exec -it colpali-research-node /bin/bash

How to Check Status

To see if your research node is currently running:
Bash

docker ps -a | grep colpali

📂 Project Structure

    Dockerfile: Reconstructs the optimized CUDA 12.6 environment.

    manage_env.sh: Automation script for building and mounting local data volumes.

    index_library.py: Logic for scanning PDFs and generating multi-vector embeddings.

    research_engine.py: The query interface for document retrieval and synthesis.

🧪 Hardware Verification

Inside the container, run this command to verify that the Blackwell card and all libraries are correctly linked:
Bash

python3 check_hw.py

Expected Output:

    ✅ DEVICE: NVIDIA RTX 6000 Ada (or Blackwell equivalent)

    ✅ FLASH ATTENTION: LOADED

    ✅ QWEN UTILS: LOADED

    ✅ COLPALI ENGINE: READY

Ready to push to GitHub?

To finish the setup, simply run these final commands in your folder:

    git init

    git add .

    git commit -m "Initial commit: Blackwell optimized env"

    git push origin main
