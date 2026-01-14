import torch
import os
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path

# Settings for your 96GB VRAM
MODEL_NAME = "vidore/colpali-v1.2"
BATCH_SIZE = 16  # Your RTX 6000 can likely handle 32+, but 16 is a safe start
DPI = 150        # High enough for academic text, low enough for speed

# 1. Load Model (Optimized for Blackwell sm_120)
print(f"🚀 Loading {MODEL_NAME} into Blackwell GPU...")
model = ColPali.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2"
).eval()
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

# 2. Find all PDFs in your RAG_Clean folder
pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
print(f"📚 Found {len(pdf_files)} papers on NVMe. Starting Indexing...")

for pdf_file in pdf_files:
    print(f"\n📄 Processing: {pdf_file}")
    
    # Use Threadripper cores to render PDF to images
    images = convert_from_path(pdf_file, dpi=DPI, thread_count=16)
    
    all_page_embeddings = []
    
    # Process in batches to stay within VRAM limits
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        batch_images = images[i : i + BATCH_SIZE]
        
        with torch.no_grad():
            # Prep tensors
            batch_input = processor.process_images(batch_images).to(model.device)
            # Forward pass (Late Interaction / Multi-Vector)
            embeddings = model(**batch_input)
            all_page_embeddings.append(embeddings.cpu())

    # Save embeddings next to the PDF so you never have to process it again
    output_name = f"{pdf_file}.pt"
    torch.save(torch.cat(all_page_embeddings, dim=0), output_name)
    print(f"✅ Saved embeddings to {output_name}")

print("\n🔥 All academic papers indexed successfully!")
