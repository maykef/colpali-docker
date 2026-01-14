import os
import torch
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor
from colpali_engine.models import ColPali
from colpali_engine.utils.torch_utils import get_device

def index_pdfs(directory_path, output_path="embeddings.pt"):
    # 1. Setup Device (Optimized for Blackwell)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Model & Processor
    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_name)

    # 3. Find PDFs in the mounted NVMe
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {directory_path}")
        return

    all_embeddings = []

    print(f"Found {len(pdf_files)} PDFs. Starting indexing...")

    # 4. Processing Loop
    for pdf_file in tqdm(pdf_files, desc="Processing Documents"):
        path = os.path.join(directory_path, pdf_file)
        
        try:
            # Convert PDF to Images
            images = convert_from_path(path)
            
            # Process each page
            for i, image in enumerate(images):
                with torch.no_grad():
                    batch_images = processor(images=[image], return_tensors="pt").to(device)
                    embeddings = model(**batch_images)
                    all_embeddings.append({
                        "file": pdf_file,
                        "page": i,
                        "embedding": embeddings.cpu()
                    })
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    # 5. Save results to the NVMe
    torch.save(all_embeddings, output_path)
    print(f"✅ Indexing Complete. Saved to {output_path}")

if __name__ == "__main__":
    # Ensure this points to your mount inside the container
    DATA_DIR = "/app" 
    index_pdfs(DATA_DIR)
