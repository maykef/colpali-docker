#!/usr/bin/env python3
"""
Optimized PDF Indexing Pipeline
- Caches rendered images to disk
- Skips already-processed documents
- Builds centralized metadata index
- 200 DPI for optimal OCR quality
"""

import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path
from PIL import Image

# Configuration
MODEL_NAME = "vidore/colpali-v1.2"
BATCH_SIZE = 16
DPI = 200  # FIXED: 200 DPI for better OCR quality
INDEX_DIR = Path("./index_cache")
METADATA_FILE = "index_metadata.json"

# Create directory structure
INDEX_DIR.mkdir(exist_ok=True)
(INDEX_DIR / "embeddings").mkdir(exist_ok=True)
(INDEX_DIR / "images").mkdir(exist_ok=True)


def load_or_create_metadata():
    """Load existing metadata or create new"""
    if Path(METADATA_FILE).exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    """Persist metadata to disk"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def render_and_cache_pdf(pdf_path, paper_id):
    """Convert PDF to images and cache to disk at 200 DPI"""
    print(f"   🖼️  Rendering PDF at {DPI} DPI...")
    images = convert_from_path(pdf_path, dpi=DPI, thread_count=16)
    
    image_dir = INDEX_DIR / "images" / paper_id
    image_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i, img in enumerate(images):
        img_path = image_dir / f"page_{i:04d}.jpg"
        img.save(img_path, "JPEG", quality=95, optimize=True)
        image_paths.append(str(img_path))
    
    return images, image_paths


def load_cached_images(image_paths):
    """Load pre-rendered images from disk"""
    return [Image.open(p) for p in image_paths]


def embed_images(model, processor, images):
    """Generate multi-vector embeddings in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="   ⚡ Embedding"):
        batch = images[i:i + BATCH_SIZE]
        
        with torch.no_grad():
            inputs = processor.process_images(batch).to(model.device)
            embeddings = model(**inputs)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def load_model_safe():
    """Load ColPali model - try local cache first, download if needed"""
    print(f"🚀 Loading {MODEL_NAME}...")
    
    # Try local cache first (fast path)
    try:
        print("   Attempting to load from local cache...")
        model = ColPali.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            local_files_only=True
        ).eval()
        processor = ColPaliProcessor.from_pretrained(
            MODEL_NAME,
            local_files_only=True
        )
        print("   ✅ Loaded from local cache")
        return model, processor
        
    except (OSError, ValueError) as e:
        # Local cache doesn't exist, download from HuggingFace
        print("   Local cache not found, downloading from HuggingFace Hub...")
        print("   (This only happens once - subsequent runs use cached version)")
        
        model = ColPali.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()
        processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
        
        print("   ✅ Downloaded and cached for future use")
        return model, processor


def main():
    # Load model once (with smart caching)
    model, processor = load_model_safe()
    
    # Load existing metadata
    metadata = load_or_create_metadata()
    
    # Find PDFs
    pdf_files = sorted([f for f in os.listdir(".") if f.endswith(".pdf")])
    
    if not pdf_files:
        print("❌ No PDF files found in current directory")
        print("💡 Make sure you're running this inside the container with mounted data")
        return
    
    print(f"📚 Found {len(pdf_files)} PDFs in directory")
    
    # Check how many are already indexed
    already_indexed = [f for f in pdf_files if Path(f).stem in metadata]
    new_to_process = [f for f in pdf_files if Path(f).stem not in metadata]
    
    if already_indexed:
        print(f"✅ {len(already_indexed)} PDFs already indexed (skipping)")
    if new_to_process:
        print(f"🆕 {len(new_to_process)} new PDFs to process\n")
    else:
        print("\n✨ All PDFs already indexed! Nothing to do.")
        print(f"📊 Total indexed papers: {len(metadata)}")
        return
    
    # Process new PDFs only
    for pdf_file in new_to_process:
        paper_id = Path(pdf_file).stem
        
        print(f"📄 Processing: {pdf_file}")
        
        # Render and cache images at 200 DPI
        images, image_paths = render_and_cache_pdf(pdf_file, paper_id)
        
        # Generate embeddings
        embeddings = embed_images(model, processor, images)
        
        # Save embeddings
        emb_path = INDEX_DIR / "embeddings" / f"{paper_id}.pt"
        torch.save(embeddings, emb_path)
        
        # Update metadata
        metadata[paper_id] = {
            "pdf_filename": pdf_file,
            "page_count": len(images),
            "embedding_path": str(emb_path),
            "image_paths": image_paths,
            "dpi": DPI
        }
        
        print(f"   ✅ Indexed {len(images)} pages → {emb_path}\n")
    
    # Save metadata index
    save_metadata(metadata)
    
    print("="*60)
    print("🔥 INDEXING COMPLETE")
    print("="*60)
    print(f"📊 Total papers in index: {len(metadata)}")
    print(f"🆕 Newly processed: {len(new_to_process)}")
    print(f"💾 Index metadata: {METADATA_FILE}")
    print(f"📁 Cached data: {INDEX_DIR}/")
    print("\n💡 Next step: Run research_engine.py to query the index")


if __name__ == "__main__":
    main()
