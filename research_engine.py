import torch
from transformers import AutoProcessor
from colpali_engine.models import ColPali
from colpali_engine.utils.torch_utils import get_device
from PIL import Image

def run_query(query_text, index_path="embeddings.pt"):
    # 1. Setup Device
    device = get_device()
    print(f"Searching using device: {device}")

    # 2. Load Model & Processor (Must match indexer)
    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_name)

    # 3. Load the Index from NVMe
    print(f"Loading index from {index_path}...")
    saved_data = torch.load(index_path)
    
    # Extract embeddings and move to GPU for fast search
    # ColPali embeddings are typically (batch, tokens, dim)
    index_embeddings = torch.cat([d["embedding"] for d in saved_data]).to(device)

    # 4. Process Query
    with torch.no_grad():
        batch_query = processor(text=[query_text], return_tensors="pt").to(device)
        query_embedding = model.get_query_embeddings(**batch_query)

    # 5. Late Interaction Scoring (MaxSim)
    # This is the "magic" of ColPali that allows visual retrieval
    scores = model.score_multi_vector(query_embedding, index_embeddings)
    
    # 6. Get Top Results
    top_k = 5
    values, indices = torch.topk(scores[0], k=top_k)

    print(f"\nTop {top_k} Results for: '{query_text}'")
    print("-" * 50)
    for i in range(top_k):
        idx = indices[i].item()
        score = values[i].item()
        result = saved_data[idx]
        print(f"Rank {i+1}: Score {score:.4f} | File: {result['file']} | Page: {result['page']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your research query: ")
    
    run_query(user_query)
