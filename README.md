# ColPali Research Engine (Blackwell-Optimized)

[![CUDA](https://img.shields.io/badge/CUDA-12.6.2-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-red.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Blackwell-blue.svg)](#hardware-requirements)

A high-performance Vision-RAG pipeline specifically tuned for **NVIDIA Blackwell Architecture (RTX 6000 / sm_120)**. This project leverages `ColPali` for multi-vector document retrieval without the need for manual C++ compilation of Flash Attention kernels.

---

## 📂 Project Structure

```text
.
├── Dockerfile             # Builds the CUDA 12.6 research environment
├── manage_env.sh          # Script to automate build and volume mounting
├── index_library.py       # (To be added) PDF scanner and embedding logic
├── research_engine.py      # (To be added) Query and synthesis interface
├── .dockerignore          # Prevents large data files from slowing builds
└── README.md              # Documentation

🛠️ Getting Started
1. Prerequisites

    NVIDIA Driver: Version 560+ recommended.

    Storage: 8TB NVMe mounted at /mnt/nvme8tb/RAG_Clean.

    Tools: Docker and NVIDIA Container Toolkit.

2. Quick Launch

Run the management script to build the image and launch your research container automatically:
Bash

chmod +x manage_env.sh
./manage_env.sh

🕹️ Container Management

Use these commands from your host terminal to manage the lifecycle of your research node.
Action	Command
Enter Shell	docker exec -it colpali-research-node /bin/bash
Stop Node	docker stop colpali-research-node
Resume Node	docker start colpali-research-node
View Status	docker ps -a | grep colpali
🧪 Hardware Verification

Once inside the container, verify that your Blackwell card is correctly identified and the vision stack is functional:
Bash

python3 check_hw.py

⚖️ Hardware Requirements

    GPU: NVIDIA Blackwell (RTX 6000, 5090) or Ada Lovelace.

    VRAM: 24GB+ (48GB+ recommended for large document batches).

    Architecture: sm_120 support provided via PyTorch Nightly cu126.

📝 License

Distributed under the MIT License. See LICENSE for more information.


---

### 💡 Why this format works:
1.  **Badges:** The little shields at the top immediately tell users that you are using cutting-edge versions (CUDA 12.6/Nightly).
2.  **File Tree:** The code block with the directory structure helps people understand where to put their data.
3.  **Command Table:** The table for Stop/Start/Enter makes it impossible to forget the specific Docker names.
4.  **Syntax Highlighting:** Using ` ```bash ` and ` ```text ` ensures GitHub renders the colors correctly for readability.



**Now that your documentation is ready, do you want the Python code for `index_library.py` to start populating your 8TB drive with embeddings?**
