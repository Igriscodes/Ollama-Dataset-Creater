import os
import json
import asyncio
import re
import random
import fitz
import matplotlib.pyplot as plt
from collections import Counter
from ollama import AsyncClient
from tqdm.asyncio import tqdm

INPUT_FOLDER = "./textbooks_pdf"
DATASET_FILE = "dataset.jsonl"
CHECKPOINT_FILE = "advanced_checkpoint.json"
MODEL_NAME = "exaone3.5:2.4b"
CONCURRENCY_LIMIT = 4
SENTENCES_PER_PARA = 18
INSPECTION_NUM_SAMPLES = 3

TASK_TYPES = [
    "Logical Reasoning: Ask a question that requires connecting two separate ideas from the text.",
    "Counterfactual: Ask what would happen if a key process described in the text were reversed.",
    "Analogy: Ask to explain a complex concept using an everyday analogy.",
    "Applied Knowledge: Create a real‑world problem that requires this knowledge to solve.",
    "Concept Extraction: Identify the most important definition for a 10‑year‑old.",
    "Comparative Analysis: Compare two different mechanisms mentioned in the text.",
    "Multi-step Tutorial: Write a 'How‑to' guide based on the process.",
    "Criticism/Nuance: Ask about limitations of the theory described.",
    "Data Interpretation: Ask for a trend analysis or a 'Why' question.",
    "Dialogue: Create a conversation between a student and a professor."
]

def clean_academic_text(text: str) -> str:
    text = re.sub(r'\([A-Z][a-z]+, \d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+ et al\., \d{4}\)', '', text)
    text = re.sub(r'\[\d+(?:,\s*\d+|-?\d+)*\]', '', text)
    text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
    text = re.sub(r'http\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def split_into_sentences(text: str):
    sentence_list = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +', text)
    return [s.strip() for s in sentence_list if len(s.strip()) > 15]

def sanitize_content(text: str) -> str:
    return text.replace("> ", "").strip()

def convert_json_output_to_text(output_data):
    try:
        if isinstance(output_data, str) and output_data.strip().startswith(("{", "[")):
            parsed = json.loads(output_data)
            if isinstance(parsed, dict):
                lines = []
                for k, v in parsed.items():
                    formatted_key = k.replace('_', ' ').title()
                    lines.append(f"**{formatted_key}**: {v}")
                return "\n\n".join(lines)
        return output_data
    except Exception:
        return output_data

def classify_task(instruction: str) -> str:
    txt = instruction.lower()
    if "what would happen if" in txt or ("if the" in txt and "were reversed" in txt):
        return "Counterfactual"
    if any(w in txt for w in ["analogy", "imagine", "like a"]):
        return "Analogy"
    if any(w in txt for w in ["compare", "difference between", "versus"]):
        return "Comparative Analysis"
    if any(w in txt for w in ["how‑to", "tutorial", "steps to"]):
        return "Multi‑step Tutorial"
    if any(w in txt for w in ["dialogue", "conversation", "professor"]):
        return "Dialogue"
    if any(w in txt for w in ["solve", "real‑world", "scenario"]):
        return "Applied Knowledge"
    if "why" in txt and "because" not in txt:
        return "Logical Reasoning"
    if any(w in txt for w in ["define", "10‑year‑old", "meaning of"]):
        return "Concept Extraction"
    return "General/Other"

async def process_chunk(client, semaphore, context, para, chunk_id, file_name, pbar):
    async with semaphore:
        task_instruction = random.choice(TASK_TYPES)
        prompt = (
            "### SYSTEM ROLE\n"
            "You are a Senior AI Data Engineer. Your mission is to generate 'Gold Standard' instruct data. "
            "NEVER reference 'the provided text' or 'the snippet' in your output.\n\n"
            "### DIVERSITY TASK\n"
            f"{task_instruction}\n\n"
            "### CONSTRAINTS\n"
            "- REMOVE all academic noise (citations, page numbers, figure refs).\n"
            "- The 'instruction' must be standalone.\n"
            "- Use Markdown for structure. Use bolding for key terms.\n\n"
            f"### CONTEXT (For continuity): {context}\n"
            f"### SOURCE TEXT: {para}\n\n"
            'RESPONSE FORMAT (Strict JSON): {"instruction": "...", "output": "..."}'
        )
        try:
            resp = await client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                format="json",
                options={"num_ctx": 4096}
            )
            tokens = resp.get('eval_count', 0)
            duration_ns = resp.get('eval_duration', 1)
            tps = tokens / (duration_ns / 1e9)

            result = json.loads(resp['response'])
            if len(result['output']) < 60 or "snippet" in result['instruction'].lower():
                pbar.update(1)
                return None

            result['metadata'] = {"file": file_name, "chunk": chunk_id, "tps": round(tps, 2)}
            with open(DATASET_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')

            pbar.set_postfix({"TPS": f"{tps:.2f}", "Chunk": chunk_id})
            pbar.update(1)
            return chunk_id
        except Exception:
            pbar.update(1)
            return None

async def run_generation_stage():
    print("\n" + "="*30 + "\nSTAGE 1: DATA GENERATION\n" + "="*30)
    client = AsyncClient()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    if not os.path.isdir(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created {INPUT_FOLDER}. Add PDF files and rerun.")
        return

    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)

    pdfs = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    if not pdfs:
        print(f"No PDFs in {INPUT_FOLDER}.")
        return

    for file_name in pdfs:
        text = ""
        with fitz.open(os.path.join(INPUT_FOLDER, file_name)) as doc:
            for page in doc:
                text += page.get_text("text") + " "

        sentences = split_into_sentences(clean_academic_text(text))
        last_idx = checkpoint.get(file_name, -1)

        chunks = []
        for i in range(0, len(sentences), SENTENCES_PER_PARA):
            cid = i // SENTENCES_PER_PARA
            if cid <= last_idx:
                continue
            para = " ".join(sentences[i:i+SENTENCES_PER_PARA])
            ctx = " ".join(sentences[max(0, i-5):i])
            chunks.append((ctx, para, cid))

        if not chunks:
            continue

        pbar = tqdm(total=len(chunks), desc=f"File: {file_name[:20]}", unit="chunk")
        tasks = [
            process_chunk(client, semaphore, ctx, txt, cid, file_name, pbar)
            for ctx, txt, cid in chunks
        ]

        for fut in asyncio.as_completed(tasks):
            finished = await fut
            if finished is not None:
                checkpoint[file_name] = max(checkpoint.get(file_name, -1), finished)
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)
        pbar.close()

def run_clean_and_sanitize_stage():
    print("\n" + "="*30 + "\nSTAGE 2: CLEAN & SANITIZE\n" + "="*30)
    if not os.path.exists(DATASET_FILE):
        print(f"{DATASET_FILE} not found.")
        return

    temp_file = DATASET_FILE + ".tmp"
    kept = 0
    with open(DATASET_FILE, 'r', encoding='utf-8') as src, \
         open(temp_file, 'w', encoding='utf-8') as dst:
        for line in src:
            try:
                data = json.loads(line)
                cleaned = {
                    "instruction": sanitize_content(data.get("instruction", "")),
                    "output": convert_json_output_to_text(data.get("output", ""))
                }
                dst.write(json.dumps(cleaned) + '\n')
                kept += 1
            except Exception:
                continue

    os.replace(temp_file, DATASET_FILE)
    print(f"Sanitized dataset saved to {DATASET_FILE} ({kept} records).")

def run_inspection_stage():
    print("\n" + "="*30 + "\nSTAGE 3: INSPECTION\n" + "="*30)
    if not os.path.exists(DATASET_FILE):
        print(f"{DATASET_FILE} not found.")
        return

    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            print("File is empty.")
            return

        task_counts = Counter()
        inst_lens = []
        out_lens = []
        data = []

        for line in lines:
            entry = json.loads(line)
            data.append(entry)
            inst = entry.get("instruction", "")
            out = entry.get("output", "")
            task_counts[classify_task(inst)] += 1
            inst_lens.append(len(inst.split()))
            out_lens.append(len(out.split()))

        tasks, counts = zip(*task_counts.most_common())
        plt.figure(figsize=(10, 6))
        plt.bar(tasks, counts, color='skyblue', edgecolor='navy')
        plt.title("Estimated Task Type Distribution")
        plt.xlabel("Task Category")
        plt.ylabel("Samples")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("task_distribution.png")

        print(f"{'='*20} DATASET SUMMARY {'='*20}")
        print(f"Total samples: {len(data)}")
        if inst_lens:
            print(f"Avg instruction length: {sum(inst_lens)/len(inst_lens):.1f} words")
            print(f"Avg output length: {sum(out_lens)/len(out_lens):.1f} words")
        print("\nTask breakdown:")
        for t, c in task_counts.most_common():
            print(f"- {t}: {c}")

        print("\nRandom samples:")
        for i, sample in enumerate(random.sample(data, min(INSPECTION_NUM_SAMPLES, len(data))), 1):
            print(f"\n--- SAMPLE #{i} ---")
            print(f"INSTRUCTION:\n{sample.get('instruction')}")
            print("-" * 15)
            print(f"OUTPUT:\n{sample.get('output')}")
    except Exception as e:
        print(f"Inspection error: {e}")

async def main():
    await run_generation_stage()
    run_clean_and_sanitize_stage()
    run_inspection_stage()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
