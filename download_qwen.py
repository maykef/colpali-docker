import os
from huggingface_hub import snapshot_download

# Use high-speed transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8"
# Point to the ROOT of your hub, not a specific model subfolder
cache_path = "/mnt/nvme8tb/huggingface_cache/hub"

print(f"Downloading {repo_id} into standard cache format...")

snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_path,
    # Do NOT use local_dir if you want the blobs/snapshots structure
    resume_download=True,
    max_workers=8
)

print("Done. Check your hub folder now.")
