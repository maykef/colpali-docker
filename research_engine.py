import torch
import os
import time
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextStreamer
from qwen_vl_utils import process_vision_info

# --- HARDWARE & KERNEL TWEAKS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PAPER_DIR = "/app"
READER_NAME = "Qwen/Qwen2-VL-7B-Instruct"
RETRIEVER_NAME = "vidore/colpali-v1.2"

print("🧠 Initializing Blackwell Research Node...")
retriever = ColPali.from_pretrained(RETRIEVER_NAME, dtype=torch.bfloat16, device_map="cuda:0").eval()
ret_processor = ColPaliProcessor.from_pretrained(RETRIEVER_NAME)
reader = Qwen2VLForConditionalGeneration.from_pretrained(READER_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
read_processor = AutoProcessor.from_pretrained(READER_NAME, use_fast=True)
streamer = TextStreamer(read_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

def process_chunk(query, images, chunk_idx, filenames):
    """Visual Analysis with real-time logging."""
    start_time = time.time()
    print(f"   🚀 [CHUNK {chunk_idx}] Analyzing {len(images)} pages from: {set(filenames)}")
    
    chunk_prompt = (
        f"EXTRACTIVE TASK: Find every technical mention of '{query}' in these images.\n"
        "FOCUS ON: Radiocarbon dates (cal BP), projectile point measurements, site names, and stratigraphic layers.\n"
        "OUTPUT: Technical bullet points only. No fluff."
    )
    
    messages = [{"role": "user", "content": [{"type": "text", "text": chunk_prompt}, *([{"type": "image", "image": img} for img in images])]}]
    text = read_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = read_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = reader.generate(**inputs, max_new_tokens=1500)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        elapsed = time.time() - start_time
        print(f"   ✅ [CHUNK {chunk_idx}] Completed in {elapsed:.1f}s")
        return read_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

def master_researcher_v6(query):
    # 1. RETRIEVAL
    with torch.no_grad():
        query_batch = ret_processor.process_queries([query]).to(retriever.device)
        query_emb = retriever(**query_batch)

    all_hits = []
    for file in os.listdir(PAPER_DIR):
        if file.endswith(".pt"):
            emb = torch.load(os.path.join(PAPER_DIR, file)).to(retriever.device)
            scores = ret_processor.score_multi_vector(query_emb, emb)[0]
            relevant_pages = torch.where(scores > 10.3)[0].tolist()
            if relevant_pages:
                all_hits.append({"pdf": file.replace(".pt", ""), "pages": sorted(relevant_pages), "max": scores.max().item()})

    top_papers = sorted(all_hits, key=lambda x: x['max'], reverse=True)

    # 2. SOURCE REPORTING
    print(f"\n📑 IDENTIFIED {len(top_papers)} TARGET PAPERS:")
    all_images_meta = []
    source_manifest = ""
    for hit in top_papers:
        line = f"   • {hit['pdf']} (Max Score: {hit['max']:.2f})"
        print(line)
        source_manifest += line + "\n"
        pdf_path = os.path.join(PAPER_DIR, hit['pdf'])
        for p in hit['pages'][:2]:
            img = convert_from_path(pdf_path, first_page=p+1, last_page=p+1, dpi=180)[0]
            all_images_meta.append((img, hit['pdf']))

    # 3. CHUNKING
    chunk_size = 4
    chunks = [all_images_meta[i:i + chunk_size] for i in range(0, len(all_images_meta), chunk_size)]
    chunk_summaries = []

    print(f"\n⚙️  Processing {len(chunks)} Chunks on RTX 6000 Blackwell...")
    for i, chunk in enumerate(chunks):
        imgs = [c[0] for c in chunk]
        names = [c[1] for c in chunk]
        summary = process_chunk(query, imgs, i+1, names)
        chunk_summaries.append(f"DATA FROM {set(names)}:\n{summary}")

    # 4. FINAL MASTER REPORT (No Summary Policy)
    master_instruction = (
        "You are a Lead Archaeologist. Below is raw evidence from the library.\n\n"
        f"RESEARCH QUESTION: {query}\n\n"
        "SOURCE MANIFEST:\n" + source_manifest + "\n"
        "DATA POINTS EXTRACTED:\n" + "\n\n".join(chunk_summaries) + "\n\n"
        "FINAL TASK:\n"
        "Synthesize this data into an EXHAUSTIVE TECHNICAL REPORT.\n"
        "1. Start immediately with Chronological and Site-Specific data.\n"
        "2. List Lithic Measurements and Tool Classes found.\n"
        "3. Explicitly cite the [Filename] for every fact.\n"
        "4. STRICT RULE: DO NOT provide a summary, abstract, or introduction. Provide ONLY the dense technical findings."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": master_instruction}]}]
    text = read_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = read_processor(text=[text], padding=True, return_tensors="pt").to("cuda")

    print("\n🤖 GENERATING MASTER REPORT (Live Stream):\n" + "—"*50)
    try:
        _ = reader.generate(**inputs, max_new_tokens=4096, streamer=streamer, temperature=0.7, do_sample=True, repetition_penalty=1.1)
    except Exception as e:
        print(f"\n❌ ERROR DURING GENERATION: {e}")

if __name__ == "__main__":
    while True:
        try:
            q = input("\n💬 Research Question: ")
            if q.lower() in ['exit', 'quit']: break
            master_researcher_v6(q)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user. Cleaning VRAM...")
            torch.cuda.empty_cache()
            continue
