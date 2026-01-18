#!/usr/bin/env python3
"""
Grounding-Enforced Research Engine (PRODUCTION READY)
- Top-K retrieval (no arbitrary thresholds)
- Multi-Model Comparison (7B vs 72B 8-bit)
- Reduced chunk size for better OCR
- Consistent filename citations
- Full offline operation with initial cache check
- RESTORED: Detailed ColPali scoring and paper identification logging
"""

import torch
import os
import json
import time
import re
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextStreamer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initial state: allow online check to ensure volume mount is populated
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

METADATA_FILE = "index_metadata.json"
RETRIEVER_NAME = "vidore/colpali-v1.2"
MODEL_7B = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_72B = "Qwen/Qwen2-VL-72B-Instruct"

# OPTIMIZED Retrieval settings
TOP_K_PAGES = 20
TOP_PAGES_PER_PAPER = 2
CHUNK_SIZE = 2


class ResearchEngine:
    def __init__(self):
        # 1. Verification Step (Solves the Volume Mount Issue)
        print(f"📡 Verifying local cache for {RETRIEVER_NAME}...")
        ColPali.from_pretrained(RETRIEVER_NAME, dtype=torch.bfloat16, device_map="cpu")
        ColPaliProcessor.from_pretrained(RETRIEVER_NAME)

        # 2. Strict Offline Enforcement
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.metadata = self._load_metadata()
        self.embeddings_cache = self._preload_embeddings()

        self.reader = None
        self.read_processor = None
        self.current_model_id = None
        print("✅ Research engine initialized\n")

    def _load_metadata(self):
        if not Path(METADATA_FILE).exists():
            raise FileNotFoundError(f"Index not found! Run index_library.py first.")
        with open(METADATA_FILE, 'r') as f:
            meta = json.load(f)
        print(f"📚 Loaded index: {len(meta)} papers")
        return meta

    def _preload_embeddings(self):
        cache = {}
        total_size = 0
        print("⏳ Pre-loading embeddings into RAM...")
        for paper_id, meta in self.metadata.items():
            emb = torch.load(meta['embedding_path'], map_location='cpu')
            cache[paper_id] = emb
            total_size += emb.element_size() * emb.nelement()
        print(f"   💾 Cached {total_size / 1e9:.2f} GB in RAM")
        return cache

    def unload_reader(self):
        """Explicitly unloads the reader model from GPU memory."""
        if self.reader is not None:
            print(f"   🧹 Offloading model to free VRAM...")
            del self.reader
            del self.read_processor
            self.reader = None
            self.read_processor = None
            self.current_model_id = None
            torch.cuda.empty_cache()

    def _ensure_reader_loaded(self, choice):
        """Switches models and applies quantization if Choice 2 (72B) is selected."""
        target_id = MODEL_7B if choice == "1" else MODEL_72B

        if self.current_model_id == target_id and self.reader is not None:
            return 0.0

        start_load = time.perf_counter()
        if self.reader is not None:
            self.unload_reader()

        print(f"🤖 Loading {target_id}...")
        os.environ["HF_HUB_OFFLINE"] = "0"

        load_params = {
            "pretrained_model_name_or_path": target_id,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True
        }

        if choice == "2":
            print("   📦 Applying 8-bit quantization for 72B model...")
            load_params["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_params["dtype"] = torch.bfloat16

        self.reader = Qwen2VLForConditionalGeneration.from_pretrained(**load_params)
        self.read_processor = AutoProcessor.from_pretrained(target_id, trust_remote_code=True)

        os.environ["HF_HUB_OFFLINE"] = "1"
        self.current_model_id = target_id
        print(f"   ✅ {target_id} Ready\n")
        return time.perf_counter() - start_load

    def retrieve_relevant_pages(self, query):
        """TOP-K RETRIEVAL WITH SCORE REPORTING"""
        print(f"🔍 RETRIEVAL PHASE")
        print(f"   Query: '{query}'")
        print(f"   Loading retriever from local cache...")

        retriever = ColPali.from_pretrained(
            RETRIEVER_NAME, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="flash_attention_2",
            local_files_only=True
        ).eval()
        processor = ColPaliProcessor.from_pretrained(RETRIEVER_NAME, local_files_only=True)

        with torch.no_grad():
            query_batch = processor.process_queries([query]).to("cuda:0")
            query_emb = retriever(**query_batch)

        all_page_scores = []
        for paper_id, emb in self.embeddings_cache.items():
            scores = processor.score_multi_vector(query_emb, emb.to("cuda:0"))[0]
            for idx, s in enumerate(scores):
                all_page_scores.append({"paper_id": paper_id, "page_idx": idx, "score": s.item()})

        all_page_scores.sort(key=lambda x: x['score'], reverse=True)
        top_pages = all_page_scores[:TOP_K_PAGES]

        if not top_pages:
            print("   ❌ No pages found in index")
            return None

        # Statistics logging matching research_engine_2.py
        max_score = top_pages[0]['score']
        min_score = top_pages[-1]['score']
        median_score = top_pages[len(top_pages) // 2]['score']

        print(f"   ✅ Selected top {len(top_pages)} pages")
        print(f"   📊 Score Range: {max_score:.2f} (best) → {min_score:.2f} (worst)")
        print(f"   📊 Median Score: {median_score:.2f}")

        if max_score < 12.0:
            print(f"\n   ⚠️  WARNING: Best match score is {max_score:.2f}")
            print(f"   ⚠️  Documents may not prominently feature this topic")
            print(f"   💡 Results may be less reliable\n")

        paper_groups = {}
        for p in top_pages:
            pid = p['paper_id']
            if pid not in paper_groups: paper_groups[pid] = []
            if len(paper_groups[pid]) < TOP_PAGES_PER_PAPER:
                paper_groups[pid].append(p)

        del retriever, processor
        torch.cuda.empty_cache()
        return paper_groups

    def analyze_chunk(self, query, images, filenames, chunk_idx):
        unique_sources = list(set(filenames))
        print(f"   ⚡ [Chunk {chunk_idx}] Analyzing {len(images)} pages from: {', '.join(unique_sources)}")

        sources_list = "\n".join([f"  • {f}" for f in unique_sources])

        prompt = (
            f"EXTRACTION TASK: Find details about '{query}' in these images.\n\n"
            "MANDATORY CITATION FORMAT:\n"
            f"✅ 'Fact here [{filenames[0]}]'\n"
            "❌ 'Fact here [Top-left]' (WRONG)\n\n"
            f"SOURCES:\n{sources_list}\n\n"
            "OUTPUT (Every sentence must end with [Filename.pdf]):"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})

        text = self.read_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.read_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            gen_ids = self.reader.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
            output = self.read_processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[
                0]

        return output

    def verify_grounding(self, text, sources):
        warnings = []
        citations = re.findall(r'\[([^\]]*\.pdf[^\]]*)\]', text, re.IGNORECASE)
        words = len(text.split())

        if words > 100 and len(citations) < 3:
            warnings.append(f"Low citation density: {len(citations)} citations in {words} words")

        if not citations:
            warnings.append("No citations found in final output")

        return warnings, len(citations)

    def research(self, query, model_choice):
        print("\n" + "=" * 70)
        print(f"QUERY: {query} (Using Model {model_choice})")
        print("=" * 70 + "\n")

        # 1. RETRIEVAL TIMING
        t_start_retrieval = time.perf_counter()
        paper_groups = self.retrieve_relevant_pages(query)
        t_end_retrieval = time.perf_counter()

        if not paper_groups: return

        print(f"📄 RETRIEVED PAPERS:")
        all_images_meta = []
        source_manifest = []
        for pid, pages in paper_groups.items():
            meta = self.metadata[pid]
            fname = meta['pdf_filename']
            source_manifest.append(fname)
            best_s = max(p['score'] for p in pages)
            print(f"   📄 {fname} (Best: {best_s:.2f}, {len(pages)} pages)")
            for p in pages:
                all_images_meta.append((Image.open(meta['image_paths'][p['page_idx']]), fname))

        print(f"\n   📊 Total pages to analyze: {len(all_images_meta)}\n")

        # 2. MODEL LOADING TIMING
        load_time = self._ensure_reader_loaded(model_choice)

        # 3. CHUNK PROCESSING TIMING
        t_start_chunks = time.perf_counter()
        chunk_results = []
        chunks = [all_images_meta[i:i + CHUNK_SIZE] for i in range(0, len(all_images_meta), CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            res = self.analyze_chunk(query, [c[0] for c in chunk], [c[1] for c in chunk], i + 1)
            chunk_results.append(res)
        t_end_chunks = time.perf_counter()

        print("\n" + "=" * 70 + "\nSYNTHESIS PHASE\n" + "=" * 70)
        evidence = "\n\n".join(chunk_results)

        # 4. SYNTHESIS/RESPONSE TIMING
        t_start_synth = time.perf_counter()
        synth_prompt = (
                f"Synthesize the findings for '{query}' using ONLY the evidence below.\n"
                "CRITICAL: Every sentence MUST end with [Filename.pdf].\n"
                "NO general knowledge.\n\n"
                f"AVAILABLE SOURCES:\n" + "\n".join([f"  • {s}" for s in source_manifest]) + "\n\n"
                                                                                             f"EVIDENCE:\n{evidence}"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": synth_prompt}]}]
        text = self.read_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.read_processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        print(f"\n📝 MASTER REPORT:\n" + "-" * 70)
        with torch.no_grad():
            gen_ids = self.reader.generate(
                **inputs, max_new_tokens=1024,
                streamer=TextStreamer(self.read_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            )
            full_output = \
                self.read_processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        t_end_synth = time.perf_counter()

        print("-" * 70)
        print("\n🔍 GROUNDING VERIFICATION:")
        warnings, citation_count = self.verify_grounding(full_output, source_manifest)
        if citation_count == 0:
            print("   🛑 REJECTED - No .pdf citations found")
        elif warnings:
            print("   🚨 ISSUES DETECTED:")
            for w in warnings: print(f"      ⚠️  {w}")
        else:
            print("   ✅ Well-grounded output")
            print(f"   ✅ {citation_count} source citations")

        # PRINT PERFORMANCE STATISTICS
        print("\n" + "⏱️  PERFORMANCE STATISTICS:")
        print(f"   • Model Load/Switch:  {load_time:.2f}s")
        print(f"   • Retrieval:          {t_end_retrieval - t_start_retrieval:.2f}s")
        print(f"   • Paper Analysis:     {t_end_chunks - t_start_chunks:.2f}s")
        print(f"   • Report Synthesis:   {t_end_synth - t_start_synth:.2f}s")
        print(
            f"   • Total Task Time:    {load_time + (t_end_retrieval - t_start_retrieval) + (t_end_chunks - t_start_chunks) + (t_end_synth - t_start_synth):.2f}s")
        print("=" * 70 + "\n")


def main():
    engine = ResearchEngine()
    print("💬 Ready (type 'exit' to quit)\n")

    while True:
        print("CHOOSE MODEL:")
        print("  1) Qwen2-VL-7B")
        print("  2) Qwen2-VL-72B (8-bit)")
        m_choice = input("Choice (1 or 2): ").strip()
        if m_choice.lower() in ['exit', 'quit']: break
        if m_choice not in ["1", "2"]: continue

        query = input("Query: ").strip()
        if not query or query.lower() in ['exit', 'quit']: break

        try:
            engine.research(query, m_choice)
            engine.unload_reader()
        except Exception as e:
            print(f"❌ Error: {e}")
            engine.unload_reader()


if __name__ == "__main__":
    main()
