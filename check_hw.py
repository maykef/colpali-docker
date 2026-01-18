#!/usr/bin/env python3
import torch
import os

print("=" * 70)
print(" ^=^v   ^o  HARDWARE VERIFICATION")
print("=" * 70)

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f" ^|^e GPU: {name}")
    print(f" ^|^e VRAM: {vram:.1f} GB")
else:
    print(" ^z   ^o  No GPU detected")

print(f" ^|^e CUDA: {torch.version.cuda}")
print(f" ^|^e PyTorch: {torch.__version__}")
print("=" * 70)
print()
print(" ^=^s  CHECKING CACHED MODELS")
print("=" * 70)

cache_dir = "/root/.cache/huggingface/hub"
if os.path.exists(cache_dir):
    models = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
    if models:
        print(f" ^|^e Found {len(models)} cached models:")
        for model in sorted(models):
            model_name = model.replace("models--", "").replace("--", "/")
            print(f"    ^`  {model_name}")
    else:
        print(" ^z   ^o  No models found in cache!")
        print("   Run download_models_host.py on HOST first")
else:
    print(" ^}^l Cache directory not mounted!")
    print("   Check manage_env.sh HF_CACHE_DIR mount")

print("=" * 70)
print()
print(" ^=^r  NEXT STEPS")
print("=" * 70)
print("1. Index documents: python3 index_library.py")
print("2. Run queries: python3 research_engine.py")
print("=" * 70)
