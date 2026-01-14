#!/bin/bash

IMAGE_NAME="colpali-blackwell-repo"
CONTAINER_NAME="colpali-research-node"
DATA_DIR="/mnt/nvme8tb/RAG_Clean"

echo "🏗️  Building Blackwell-Optimized Image..."
docker build -t $IMAGE_NAME .

docker rm -f $CONTAINER_NAME 2>/dev/null

echo "🚀 Launching Research Node..."
docker run --gpus all -it \
  --name $CONTAINER_NAME \
  --ipc=host \
  --shm-size=16gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_DIR:/app \
  $IMAGE_NAME /bin/bash
