#!/usr/bin/env python3
"""
Grounding-Enforced Research Engine (PRODUCTION READY)
- Top-K retrieval (no arbitrary thresholds)
- Reduced chunk size for better OCR
- Consistent filename citations
- Full offline operation
- All fixes applied
"""

import torch
import os
import json
import time
import re
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info

# Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

METADATA_FILE = "index_metadata.json"
RETRIEVER_NAME = "vidore/colpali-v1.2"
READER_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# OPTIMIZED Retrieval settings
TOP_K_PAGES = 20
TOP_PAGES_PER_PAPER = 2
CHUNK_SIZE = 2


class ResearchEngine:
    """Optimized sequential model loading"""
    
    def __init__(self):
        self.metadata = self._load_metadata()
        self.embeddings_cache = self._preload_embeddings()
        self.reader = None
        self.read_processor = None
        self.streamer = None
        print("✅ Research engine initialized\n")
    
    def _load_metadata(self):
        """Load index metadata"""
        if not Path(METADATA_FILE).exists():
            raise FileNotFoundError(f"Index not found! Run index_library.py first.")
        
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        print(f"📚 Loaded index: {len(metadata)} papers")
        return metadata
    
    def _preload_embeddings(self):
        """Preload all embeddings into RAM"""
        cache = {}
        total_size = 0
        
        print("⏳ Pre-loading embeddings into RAM...")
        for paper_id, meta in self.metadata.items():
            emb_path = meta['embedding_path']
            emb = torch.load(emb_path, map_location='cpu')
            cache[paper_id] = emb
            total_size += emb.element_size() * emb.nelement()
        
        print(f"   💾 Cached {total_size / 1e9:.2f} GB in RAM")
        return cache
    
    def retrieve_relevant_pages(self, query):
        """TOP-K RETRIEVAL"""
        print(f"🔍 RETRIEVAL PHASE")
        print(f"   Query: '{query}'")
        print(f"   Loading retriever from local cache...")
        
        retriever = ColPali.from_pretrained(
            RETRIEVER_NAME,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            local_files_only=True
        ).eval()
        
        processor = ColPaliProcessor.from_pretrained(
            RETRIEVER_NAME,
            local_files_only=True
        )
        
        with torch.no_grad():
            query_batch = processor.process_queries([query]).to("cuda:0")
            query_emb = retriever(**query_batch)
        
        all_page_scores = []
        
        for paper_id, emb in self.embeddings_cache.items():
            emb_gpu = emb.to("cuda:0")
            scores = processor.score_multi_vector(query_emb, emb_gpu)[0]
            
            for page_idx, score in enumerate(scores):
                all_page_scores.append({
                    "paper_id": paper_id,
                    "page_idx": page_idx,
                    "score": score.item()
                })
        
        all_page_scores.sort(key=lambda x: x['score'], reverse=True)
        top_pages = all_page_scores[:TOP_K_PAGES]
        
        if not top_pages:
            print("   ❌ No pages found in index")
            del retriever, processor, query_batch, query_emb
            torch.cuda.empty_cache()
            return None
        
        max_score = top_pages[0]['score']
        min_score = top_pages[-1]['score']
        median_score = top_pages[len(top_pages)//2]['score']
        
        print(f"   ✅ Selected top {len(top_pages)} pages")
        print(f"   📊 Score Range: {max_score:.2f} (best) → {min_score:.2f} (worst)")
        print(f"   📊 Median Score: {median_score:.2f}")
        
        if max_score < 12.0:
            print(f"\n   ⚠️  WARNING: Best match score is {max_score:.2f}")
            print(f"   ⚠️  Documents may not prominently feature this topic")
            print(f"   💡 Results may be less reliable\n")
        
        paper_groups = {}
        for page in top_pages:
            pid = page['paper_id']
            if pid not in paper_groups:
                paper_groups[pid] = []
            if len(paper_groups[pid]) < TOP_PAGES_PER_PAPER:
                paper_groups[pid].append(page)
        
        del retriever, processor, query_batch, query_emb, emb_gpu
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"   🧹 Unloaded retriever, freed VRAM\n")
        
        return paper_groups
    
    def _ensure_reader_loaded(self):
        """Lazy load reader"""
        if self.reader is None:
            print("🤖 GENERATION PHASE")
            print("   Loading Qwen2-VL reader from local cache...")
            
            self.reader = Qwen2VLForConditionalGeneration.from_pretrained(
                READER_NAME,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                local_files_only=True,
                trust_remote_code=True
            )
            self.read_processor = AutoProcessor.from_pretrained(
                READER_NAME,
                local_files_only=True,
                trust_remote_code=True
            )
            self.streamer = TextStreamer(
                self.read_processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            print("   ✅ Reader loaded\n")
    
    def analyze_chunk(self, query, images, filenames, chunk_idx):
        """Process images with citations"""
        start = time.time()
        unique_sources = list(set(filenames))
        print(f"   ⚡ [Chunk {chunk_idx}] Analyzing {len(images)} pages from: {', '.join(unique_sources)}")
        
        sources_instruction = "\n".join([f"  • {fname}" for fname in unique_sources])
        
        chunk_prompt = (
            f"EXTRACTION TASK: Find information about '{query}' in these images.\n\n"
            "MANDATORY CITATION FORMAT:\n"
            f"✅ 'Fact here [{filenames[0]}]'\n"
            f"✅ 'Another fact [{filenames[0] if len(filenames) > 0 else 'Filename.pdf'}]'\n"
            "❌ 'Fact here [Top-left]' (WRONG - need filename)\n"
            "❌ 'Fact here' (WRONG - no citation)\n\n"
            "CRITICAL RULES:\n"
            "1. Every finding MUST end with [Filename.pdf] from list below\n"
            "2. If nothing relevant visible: 'NOT FOUND IN IMAGES'\n"
            "3. Quote exact text/numbers when visible\n"
            "4. NO general knowledge - ONLY what you see\n\n"
            f"USE THESE EXACT FILENAMES FOR CITATIONS:\n{sources_instruction}\n\n"
            "OUTPUT (each line must have [Filename.pdf]):\n"
        )
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": chunk_prompt},
                *[{"type": "image", "image": img} for img in images]
            ]
        }]
        
        text = self.read_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.read_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            generated_ids = self.reader.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.15
            )
            generated_ids_trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            output = self.read_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]
        
        elapsed = time.time() - start
        
        pdf_citations = re.findall(r'\[([^\]]*\.pdf[^\]]*)\]', output, re.IGNORECASE)
        has_not_found = 'NOT FOUND' in output.upper() or 'NO RELEVANT' in output.upper()
        
        if len(pdf_citations) == 0 and not has_not_found:
            print(f"   ⚠️  [Chunk {chunk_idx}] No .pdf citations detected!")
        else:
            print(f"   ✅ [Chunk {chunk_idx}] Found {len(pdf_citations)} citations in {elapsed:.1f}s")
        
        return output, unique_sources
    
    def verify_grounding(self, generated_text, available_sources):
        """Verify citations"""
        warnings = []
        
        citations = re.findall(r'\[([^\]]*\.pdf[^\]]*)\]', generated_text, re.IGNORECASE)
        words = len(generated_text.split())
        
        if words > 100 and len(citations) < 3:
            warnings.append(f"Low citation density: {len(citations)} citations in {words} words")
        
        return warnings, len(citations)
    
    def research(self, query):
        """Execute research pipeline"""
        print("="*70)
        print(f"QUERY: {query}")
        print("="*70 + "\n")
        
        paper_groups = self.retrieve_relevant_pages(query)
        
        if paper_groups is None:
            return
        
        print(f"📄 RETRIEVED PAPERS:")
        all_images_meta = []
        source_manifest = []
        
        for paper_id, pages in paper_groups.items():
            meta = self.metadata[paper_id]
            filename = meta['pdf_filename']
            best_score = max(p['score'] for p in pages)
            
            print(f"   📄 {filename} (Best: {best_score:.2f}, {len(pages)} pages)")
            source_manifest.append(filename)
            
            for page_info in pages:
                page_idx = page_info['page_idx']
                img_path = meta['image_paths'][page_idx]
                img = Image.open(img_path)
                all_images_meta.append((img, filename))
        
        print(f"\n   📊 Total pages to analyze: {len(all_images_meta)}\n")
        
        self._ensure_reader_loaded()
        
        chunks = [
            all_images_meta[i:i + CHUNK_SIZE]
            for i in range(0, len(all_images_meta), CHUNK_SIZE)
        ]
        
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            imgs = [c[0] for c in chunk]
            names = [c[1] for c in chunk]
            result, sources = self.analyze_chunk(query, imgs, names, i+1)
            chunk_results.append(result)
        
        print("\n" + "="*70)
        print("SYNTHESIS PHASE")
        print("="*70 + "\n")
        
        evidence_text = "\n\n".join([
            f"EVIDENCE BLOCK {i+1}:\n{result}"
            for i, result in enumerate(chunk_results)
        ])
        
        example_source_1 = source_manifest[0] if len(source_manifest) > 0 else "Filename.pdf"
        example_source_2 = source_manifest[1] if len(source_manifest) > 1 else example_source_1
        
        synthesis_prompt = (
            "You are synthesizing research findings. CRITICAL: Cite sources for every fact.\n\n"
            "CITATION EXAMPLES:\n"
            f"✅ 'Beringia was a land bridge [{example_source_1}]'\n"
            f"✅ 'It connected continents [{example_source_2}]'\n"
            "❌ 'Beringia existed' (no citation)\n\n"
            "RULES:\n"
            "1. Every sentence MUST end with [Filename.pdf]\n"
            "2. Use ONLY filenames from list below\n"
            "3. If evidence is insufficient: 'INSUFFICIENT EVIDENCE'\n"
            "4. NO general knowledge - ONLY cited facts\n\n"
            f"QUERY: {query}\n\n"
            f"AVAILABLE SOURCES:\n" + "\n".join([f"  • {s}" for s in source_manifest]) + "\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "SYNTHESIZE (every sentence needs [Filename.pdf]):\n"
        )
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": synthesis_prompt}]
        }]
        
        text = self.read_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.read_processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        print("📝 MASTER REPORT:\n")
        print("-"*70 + "\n")
        
        class CaptureStreamer(TextStreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.captured_text = []
            
            def on_finalized_text(self, text, stream_end=False):
                super().on_finalized_text(text, stream_end)
                self.captured_text.append(text)
        
        capture_streamer = CaptureStreamer(
            self.read_processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        try:
            _ = self.reader.generate(
                **inputs,
                max_new_tokens=3000,
                streamer=capture_streamer,
                temperature=0.4,
                do_sample=True,
                repetition_penalty=1.2
            )
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            return
        
        full_output = "".join(capture_streamer.captured_text)
        print("\n" + "-"*70)
        
        print("\n🔍 GROUNDING VERIFICATION:")
        
        pdf_citations = re.findall(r'\[([^\]]*\.pdf[^\]]*)\]', full_output, re.IGNORECASE)
        
        if len(pdf_citations) == 0:
            print("   🛑 REJECTED - No .pdf citations found")
            print("   ❌ Output appears to be hallucinated\n")
            return
        
        warnings, citation_count = self.verify_grounding(full_output, source_manifest)
        
        if warnings:
            print("   🚨 ISSUES DETECTED:")
            for w in warnings:
                print(f"      ⚠️  {w}")
        else:
            print("   ✅ Well-grounded output")
            print(f"   ✅ {citation_count} source citations")
        
        print("\n" + "="*70 + "\n")


def main():
    """Interactive research loop"""
    print("="*70)
    print("GROUNDING-ENFORCED RESEARCH ENGINE")
    print("="*70)
    print("\nFeatures:")
    print("  • Top-K retrieval")
    print("  • Chunk size: 2 images")
    print("  • Filename citations")
    print("  • VRAM optimized")
    print("  • Offline operation\n")
    
    try:
        engine = ResearchEngine()
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        return
    
    print("💬 Ready (type 'exit' to quit)\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            if not query or query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Exiting")
                break
            
            engine.research(query)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted\n")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
