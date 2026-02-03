import os
import json
import asyncio
import re
import random
import time
import fitz
import matplotlib.pyplot as plt
from collections import Counter
from ollama import AsyncClient
from tqdm.asyncio import tqdm

INPUT_FOLDER = "./textbooks_pdf"
RAW_OUTPUT_FILE = "advanced_instruct_dataset.jsonl"
CHECKPOINT_FILE = "advanced_checkpoint.json"
MODEL_NAME = "exaone3.5:2.4b"
CONCURRENCY_LIMIT = 4
SENTENCES_PER_PARA = 18

CLEAN_FILE = "final_training_data.jsonl"
SANITIZED_FILE = "sanitized_instruct_dataset.jsonl"

INSPECTION_NUM_SAMPLES = 3

TASK_TYPES = [
    "Logical Reasoning: Ask a question that requires connecting two separate ideas from the text.",
    "Counterfactual: Ask what would happen if a key process described in the text were reversed.",
    "Analogy: Ask to explain a complex concept using an everyday analogy.",
    "Applied Knowledge: Create a real-world problem that requires this knowledge to solve.",
    "Concept Extraction: Identify the most important definition for a 10-year-old.",
    "Comparative Analysis: Compare two different mechanisms mentioned in the text.",
    "Multi-step Tutorial: Write a 'How-to' guide based on the process.",
    "Criticism/Nuance: Ask about limitations of the theory described.",
    "Data Interpretation: Ask for a trend analysis or a 'Why' question.",
    "Dialogue: Create a conversation between a student and a professor."
]

def clean_academic_text(text):
    text = re.sub(r'\([A-Z][a-z]+, \d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+ et al\., \d{4}\)', '', text)
    text = re.sub(r'\[\d+(?:,\s*\d+|-?\d+)*\]', '', text)
    text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    sentence_list = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +', text)
    return [s.strip() for s in sentence_list if len(s.strip()) > 15]

async def process_chunk(client, semaphore, context, current_para, chunk_id, file_name, pbar):
    async with semaphore:
        task_instruction = random.choice(TASK_TYPES)
        prompt = (
            "### SYSTEM ROLE\n"
            "You are a Senior AI Data Engineer. Your mission is to generate 'Gold Standard' instruct data. NEVER reference 'the provided text' or 'the snippet' in your output.\n\n"
            "### DIVERSITY TASK\n"
            f"{task_instruction}\n\n"
            "### CONSTRAINTS\n"
            "- REMOVE all academic noise (citations, page numbers, figure refs).\n"
            "- The 'instruction' must be standalone (a user shouldn't need the textbook to understand the question).\n"
            "- Use Markdown for structure. Use bolding for key terms.\n\n"
            f"### CONTEXT (For continuity): {context}\n"
            f"### SOURCE TEXT: {current_para}\n\n"
            'RESPONSE FORMAT (Strict JSON): {"instruction": "...", "output": "..."}'
        )
        try:
            response = await client.generate(model=MODEL_NAME, prompt=prompt, format="json", options={"num_ctx": 4096})
            tokens = response.get('eval_count', 0)
            duration_ns = response.get('eval_duration', 1)
            tps = tokens / (duration_ns / 1e9)
            result = json.loads(response['response'])
            if len(result['output']) < 60 or "snippet" in result['instruction'].lower():
                pbar.update(1)
                return None
            result['metadata'] = {"file": file_name, "chunk": chunk_id, "tps": round(tps, 2)}
            with open(RAW_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            pbar.set_postfix({"TPS": f"{tps:.2f}", "Last Chunk": chunk_id})
            pbar.update(1)
            return chunk_id
        except Exception:
            pbar.update(1)
            return None

async def run_generation_stage():
    print("\n" + "="*30 + "\nSTAGE 1: DATA GENERATION\n" + "="*30)
    client = AsyncClient()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created {INPUT_FOLDER}. Please add PDF files and run again.")
        return
    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {INPUT_FOLDER}. Skipping generation.")
        return
    for file_name in pdf_files:
        full_text = ""
        with fitz.open(os.path.join(INPUT_FOLDER, file_name)) as doc:
            for page in doc:
                full_text += page.get_text("text") + " "
        sentences = split_into_sentences(clean_academic_text(full_text))
        last_idx = checkpoint.get(file_name, -1)
        all_chunks = []
        for i in range(0, len(sentences), SENTENCES_PER_PARA):
            chunk_id = i // SENTENCES_PER_PARA
            if chunk_id <= last_idx:
                continue
            current_para = " ".join(sentences[i:i+SENTENCES_PER_PARA])
            context_para = " ".join(sentences[max(0, i-5):i])
            all_chunks.append((context_para, current_para, chunk_id))
        if not all_chunks:
            continue
        pbar = tqdm(total=len(all_chunks), desc=f"Processing {file_name[:20]}", unit="chunk")
        tasks = [
            process_chunk(client, semaphore, ctx, txt, cid, file_name, pbar)
            for ctx, txt, cid in all_chunks
        ]
        for completed_task in asyncio.as_completed(tasks):
            finished_id = await completed_task
            if finished_id is not None:
                checkpoint[file_name] = max(checkpoint.get(file_name, -1), finished_id)
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)
        pbar.close()

def run_cleaning_stage():
    print("\n" + "="*30 + "\nSTAGE 2: CLEANING FIELDS\n" + "="*30)
    if not os.path.exists(RAW_OUTPUT_FILE):
        print(f"Input file {RAW_OUTPUT_FILE} not found. Skipping.")
        return
    with open(RAW_OUTPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(CLEAN_FILE, 'w', encoding='utf-8') as f_out:
        count = 0
        for line in f_in:
            try:
                data = json.loads(line)
                clean_obj = {"instruction": data.get("instruction"), "output": data.get("output")}
                f_out.write(json.dumps(clean_obj) + '\n')
                count += 1
            except:
                continue
    print(f"Cleaned dataset saved to {CLEAN_FILE} ({count} records).")

def sanitize_content(text):
    return text.replace("> ", "").strip()

def convert_json_output_to_text(output_data):
    try:
        if isinstance(output_data, str) and (output_data.strip().startswith('{') or output_data.strip().startswith('[')):
            parsed = json.loads(output_data)
            if isinstance(parsed, dict):
                lines = []
                for key, value in parsed.items():
                    formatted_key = key.replace('_', ' ').title()
                    lines.append(f"**{formatted_key}**: {value}")
                return "\n\n".join(lines)
        return output_data
    except:
        return output_data

def run_sanitizer_stage():
    print("\n" + "="*30 + "\nSTAGE 3: SANITIZATION\n" + "="*30)
    if not os.path.exists(CLEAN_FILE):
        print(f"Error: {CLEAN_FILE} not found.")
        return
    count = 0
    with open(CLEAN_FILE, 'r', encoding='utf-8') as f_in, \
         open(SANITIZED_FILE, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                data['instruction'] = sanitize_content(data.get('instruction', ""))
                data['output'] = convert_json_output_to_text(data.get('output', ""))
                if 'metadata' in data:
                    del data['metadata']
                f_out.write(json.dumps(data) + '\n')
                count += 1
            except Exception as e:
                print(f"Skipping a malformed line: {e}")
    print(f"Sanitization complete! {count} lines processed.")
    print(f"Cleaned file saved as: {SANITIZED_FILE}")

def classify_task(instruction):
    text = instruction.lower()
    if "what would happen if" in text or ("if the" in text and "were reversed" in text):
        return "Counterfactual"
    if any(word in text for word in ["analogy", "imagine", "like a"]):
        return "Analogy"
    if any(word in text for word in ["compare", "difference between", "versus"]):
        return "Comparative Analysis"
    if any(word in text for word in ["how-to", "tutorial", "steps to"]):
        return "Multi-step Tutorial"
    if any(word in text for word in ["dialogue", "conversation", "professor"]):
        return "Dialogue"
    if any(word in text for word in ["solve", "real-world", "scenario"]):
        return "Applied Knowledge"
    if "why" in text and "because" not in text:
        return "Logical Reasoning"
    if any(word in text for word in ["define", "10-year-old", "meaning of"]):
        return "Concept Extraction"
    return "General/Other"

def run_inspection_stage():
    print("\n" + "="*30 + "\nSTAGE 4: DATA AUDIT\n" + "="*30)
    if not os.path.exists(SANITIZED_FILE):
        print(f"Error: {SANITIZED_FILE} not found.")
        return
    try:
        with open(SANITIZED_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            print("The file is empty.")
            return
        task_counts = Counter()
        inst_lengths = []
        out_lengths = []
        all_data = []
        for line in lines:
            data = json.loads(line)
            all_data.append(data)
            inst = data.get("instruction", "")
            out = data.get("output", "")
            category = classify_task(inst)
            task_counts[category] += 1
            inst_lengths.append(len(inst.split()))
            out_lengths.append(len(out.split()))
        tasks, counts = zip(*task_counts.most_common())
        plt.figure(figsize=(10, 6))
        plt.bar(tasks, counts, color='skyblue', edgecolor='navy')
        plt.title("Estimated Task Type Distribution")
        plt.xlabel("Task Category")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("task_distribution.png")
        print(f"{'='*20} DATASET SUMMARY {'='*20}")
        print(f"File: {SANITIZED_FILE}")
        print(f"Total Samples: {len(all_data)}")
        if inst_lengths:
            print(f"Avg Instruction Length: {sum(inst_lengths)/len(inst_lengths):.1f} words")
            print(f"Avg Output Length: {sum(out_lengths)/len(out_lengths):.1f} words")
        print("\nTask Breakdown:")
        for task, count in task_counts.most_common():
            print(f"- {task}: {count}")
        print(f"\nChart saved as 'task_distribution.png'\n")
        print(f"{'='*20} RANDOM SAMPLES ({INSPECTION_NUM_SAMPLES}) {'='*20}\n")
        samples = random.sample(all_data, min(INSPECTION_NUM_SAMPLES, len(all_data)))
        for i, data in enumerate(samples, 1):
            print(f"--- SAMPLE #{i} ---")
            print(f"INSTRUCTION:\n{data.get('instruction', 'N/A')}")
            print("-" * 15)
            print(f"OUTPUT:\n{data.get('output', 'N/A')}")
            print(f"{'='*60}\n")
    except Exception as e:
        print(f"An error occurred during inspection: {e}")

async def main():
    await run_generation_stage()
    run_cleaning_stage()
    run_sanitizer_stage()
    run_inspection_stage()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
