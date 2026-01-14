ColPali Research Engine (Blackwell Optimized)

A high-performance, containerized Vision-RAG (Retrieval-Augmented Generation) pipeline engineered specifically for NVIDIA Blackwell architecture (RTX 6000 / sm_120).

This project uses CUDA 12.6 and PyTorch Nightly to provide out-of-the-box compatibility with next-generation GPUs without requiring manual C++ compilation.

⸻

Overview

The ColPali Research Engine is designed for visual document retrieval rather than text-only RAG. It uses ColPali, a vision-language model that performs Late Interaction over document images, enabling accurate retrieval from:
	•	Tables
	•	Charts
	•	Diagrams
	•	Complex document layouts

This approach overcomes the limitations of traditional OCR-based RAG systems, especially for technical and scientific documents.

⸻

Project Architecture

.
├── Dockerfile              # Blackwell-optimized CUDA 12.6 environment
├── manage_env.sh           # Automation for building and volume mounting
├── index_library.py        # PDF processing and multi-vector embedding
├── research_engine.py      # Similarity search and retrieval interface
├── .dockerignore           # Prevents data bloat during Docker builds
└── README.md               # Documentation

⸻

Quick Start

Prerequisites

Host OS
	•	Linux (Ubuntu 22.04+ recommended)

GPU
	•	NVIDIA Blackwell (RTX 6000 / RTX 5090)
	•	NVIDIA Ada Lovelace (RTX 6000 Ada / RTX 4090)

Drivers
	•	NVIDIA Driver 560 or newer

Storage
	•	Local data directory (example: /mnt/nvme8tb/RAG_Clean)

⸻

Deployment

Build the Docker image and launch the research node using the provided management script.
The script automatically mounts your local NVMe directory to /app inside the container.

Commands to run on the host system:

chmod +x manage_env.sh
./manage_env.sh

⸻

Container Management

Run the following commands from the host machine:

Enter container shell
docker exec -it colpali-research-node /bin/bash

Stop container
docker stop colpali-research-node

Resume container
docker start colpali-research-node

View status
docker ps -a | grep colpali

Delete container
docker rm -f colpali-research-node

⸻

Usage Instructions

Step 1: Hardware Verification

Inside the container, verify GPU detection and required library bindings (Flash Attention, Qwen-Utils):

python3 check_hw.py

⸻

Step 2: Index Documents
	1.	Place PDF files into the mounted NVMe directory.
	2.	Generate multi-vector embeddings:

python3 index_library.py

⸻

Step 3: Research and Query

Run a visual similarity search across the indexed document library:

python3 research_engine.py “Find the data on thermal conductivity in the Blackwell whitepaper”

⸻

Hardware Requirements

Supported architectures
	•	sm_120 (Blackwell)
	•	sm_90 (Hopper)
	•	sm_89 (Ada Lovelace)

VRAM
	•	Minimum: 24 GB
	•	Recommended: 48 GB or more for large-scale indexing

Precision
	•	Uses bfloat16 for optimal performance on Blackwell tensor cores

⸻

License

This project is licensed under the MIT License.
