#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hadith sanad extraction tester

HOW TO RUN
----------
# Basic run (random seed, default sample size):
python3 test.py

# Specific seed and sample size, save a snapshot to JSON, and write human log file:
python3 test.py --seed 120 --sample_size 50 \
  --output test_runs/seed-120-sample-50.txt \
  --save-json json_snapshots/seed120_s50.json

# Rerun evaluation later from saved JSON (skips API entirely):
python3 test.py --load-json json_snapshots/seed120_s50.json \
  --output test_runs/rerun-seed120-s50.txt

WHAT THIS SCRIPT DOES
---------------------
1) Loads Kaggle hadith + rawi CSVs
2) Samples N rows from hadiths (seeded)
3) Calls your FastAPI endpoint in batches of 10 (retries + timers)
4) Optionally saves the results + truth to a JSON snapshot (or loads from one)
5) Normalizes keys, compares extracted sanad vs ground truth
6) Uses deterministic token-subset matching, then LLM fallback ("YES/NO/PROPHET")
7) Ignores Prophet mentions in metrics, logs unmatched, prints precision/recall/F1
"""

import argparse
import io
import json
import math
import os
import re
import time
from datetime import datetime
from random import randrange
from typing import List

import dotenv
import pandas as pd
import requests
from openai import OpenAI


# ============================= Helpers & Utilities =============================

def calculate_sample_size(population_size: int, confidence_level: float = 1.96, margin_of_error: float = 0.05) -> int:
    """Cochran formula with finite population correction (defaults ~95% CL, 5% MoE)."""
    p = 0.5
    n = (confidence_level**2 * p * (1 - p)) / (margin_of_error**2)
    n = n / (1 + (n - 1) / population_size)
    return math.ceil(n)


def is_missing(x) -> bool:
    """True if value is None or NaN."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def safe_str(x) -> str:
    """Convert any value to string safely; NaN/None -> ''."""
    if is_missing(x):
        return ""
    return str(x)


def safe_tokenize(x) -> List[str]:
    """Whitespace-split a string safely (handles NaN/None)."""
    s = safe_str(x).strip()
    return s.split() if s else []


def robust_int_from_any(x, default=None):
    """Extract the first digit run from anything like ': 5411,', '*.5079', ' 0007 ', '.111', etc."""
    if x is None:
        return default
    if isinstance(x, int):
        return x
    s = str(x).strip()
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else default


def normalize_api_key(api_result):
    """
    Normalize a single API result into (source:str, chapter:int, hadith:int).
    Returns None if normalization fails.
    """
    src = safe_str(api_result.get("source")).strip()
    ch = robust_int_from_any(api_result.get("chapter_no"))
    hn = robust_int_from_any(api_result.get("hadith_no"))
    if not src or ch is None or hn is None:
        return None
    return (src, ch, hn)


# ============================= CLI / Setup =============================

POPULATION = 34441
DEFAULT_SAMPLE = calculate_sample_size(POPULATION)
print(f"Sample size needed: {DEFAULT_SAMPLE}")

dotenv.load_dotenv()  # load .env for OPENAI_API_KEY, etc.

parser = argparse.ArgumentParser(description="Hadith API tester")
parser.add_argument(
    "--output",
    type=str,
    default="test_runs/hadith_api_comparison_output.txt",
    help="Output filename for results (default: test_runs/hadith_api_comparison_output.txt)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for sampling hadiths",
)
parser.add_argument(
    "--sample_size",
    type=int,
    default=DEFAULT_SAMPLE,
    help=f"Number of hadiths to sample (default: {DEFAULT_SAMPLE})",
)
parser.add_argument(
    "--save-json",
    type=str,
    default=None,
    help="If provided, save aggregated API results + truth to this JSON",
)
parser.add_argument(
    "--load-json",
    type=str,
    default=None,
    help="If provided, load results + truth from JSON and skip API calls",
)
args = parser.parse_args()

# Normalize and ensure the main log exists
output_file_1 = args.output if args.output.startswith("test_runs/") else os.path.join("test_runs", args.output)
os.makedirs(os.path.dirname(output_file_1), exist_ok=True)

rand_seed = args.seed if args.seed is not None else randrange(100)
sample_size = args.sample_size
print("Random seed =", rand_seed)
print("Sample size =", sample_size)

# OpenAI client (optional)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment. LLM fallback will be skipped.")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# ============================= Paths =============================

hadith_path = "datasets/kaggle/kaggle_hadiths_clean.csv"
rawis_path = "datasets/kaggle/kaggle_rawis.csv"


# ============================= LLM Fallback =============================

def llm_name_match(extracted: str, true_names: List[str]) -> str:
    """
    Fuzzy check using LLM. Returns one of: "YES", "PROPHET", "NO".
    - YES: extracted matches any true name (partial, alias, ibn/bin spelling, etc.)
    - PROPHET: extracted refers to the Prophet Muhammad (ﷺ) (RasulAllah, Nabi, etc.)
    - NO: does not match.
    """
    if openai_client is None:
        return "NO"  # no key -> skip LLM

    print('Inside LLM check for the following:\n')
    print("Extracted sanad:", extracted)
    print("true-names:", true_names)

    prompt = (
        "You are an expert in Arabic names, specifically in the context of Arabic Ahadith narrators (sanad). "
        "Compare the following extracted narrator name with the list of ground truth narrator names."
        "Only answer 'YES' if the extracted name is a valid reference to any of the names in the list."
        "(e.g., partial, nickname, spelling, bin/ibn variations). "
        "Sometimes the last name of a person might also be included in the extracted name but not in the Ground truth list (or vice versa) but that is fine, answer with 'YES'"
        "It could also be that the person has another valid nickname or title; use your knowledge to judge accordingly."
        "Only answer 'YES' or 'NO'.\n\n"
        "Special case: if the narrator name is the Prophet Muhammad (may peace be upon him)—including common titles "
        "like RasulAllah, Nabi—always answer 'PROPHET'.\n\n"
        f"Extracted name: {extracted}\n"
        f"Ground truth list: {true_names}\n\n"
        "Is the extracted name a valid match for any in the list? Respond with exactly one word: YES, NO, or PROPHET."
    )
    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini-2024-07-18",
            input=prompt,
            temperature=0,
        )
        ans = (response.output_text or "").strip().upper()
        print('LLM check said:', ans)
        if "PROPHET" in ans:
            return "PROPHET"
        if "YES" in ans:
            return "YES"
        return "NO"
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "NO"  # Fail safe


# ============================= Load CSVs =============================

try:
    hadith_df = pd.read_csv(hadith_path)
    print(f"DataFrame 1 loaded successfully from {hadith_path}")
    print("First 5 rows of hadith_df:")
    print(hadith_df.head())
except FileNotFoundError:
    raise SystemExit(f"Error: {hadith_path} not found. Please ensure the file exists and the path is correct.")
except Exception as e:
    raise SystemExit(f"An error occurred while reading {hadith_path}: {e}")

try:
    rawis_df = pd.read_csv(rawis_path)
    print(f"\nDataFrame 2 loaded successfully from {rawis_path}")
    print("First 5 rows of rawis_df:")
    print(rawis_df.head())
except FileNotFoundError:
    raise SystemExit(f"Error: {rawis_path} not found. Please ensure the file exists and the path is correct.")
except Exception as e:
    raise SystemExit(f"An error occurred while reading {rawis_path}: {e}")

# Sample hadiths
sample_data = hadith_df[['chain_indx', 'text_ar', 'source', 'chapter_no', 'hadith_no']].sample(
    n=sample_size, random_state=rand_seed
)
print("Total rows in dataset =", len(hadith_df))
print("Sample size           =", sample_size)

# Build JSON-friendly truth table with normalized ints
truth = [] 
for _, row in sample_data.iterrows():
    truth.append({
        "source": safe_str(row["source"]).strip(),
        "chapter_no": int(row["chapter_no"]),
        "hadith_no": robust_int_from_any(row["hadith_no"], default=None),
        "chain_indx": safe_str(row["chain_indx"]),
    })


# ============================= API Calls (or Load Snapshot) =============================

results = []
if args.load_json:
    # Load a previous snapshot (skips API)
    with open(args.load_json, "r", encoding="utf-8") as jf:
        blob = json.load(jf)
    results = blob.get("results", [])
    loaded_truth = blob.get("truth", [])
    if loaded_truth:
        truth = loaded_truth  # use snapshot truth for consistency
    print(f"Loaded {len(results)} results from {args.load_json}. Skipping API calls.")
else:
    # Batched calls with retries + timers
    BATCH_SIZE = 10
    MAX_RETRIES = 3
    BACKOFF_BASE = 2  # seconds, exponential

    num_rows = len(sample_data)
    num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nWill send {num_rows} rows in {num_batches} batch(es) of up to {BATCH_SIZE}.")
    api_total_start = time.perf_counter()

    for b_idx, start in enumerate(range(0, num_rows, BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, num_rows)
        batch_df = sample_data.iloc[start:end]

        # CSV for this batch
        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        batch_start = time.perf_counter()
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    "http://localhost:8000/upload-csv/",
                    files={"file": (f"sample_batch_{b_idx}.csv", csv_buffer, "text/csv")},
                    timeout=180,
                )
                break  # success, get out of retry loop
            except requests.RequestException as e:
                last_exc = e
                wait = BACKOFF_BASE ** (attempt - 1)
                print(f"Batch {b_idx}/{num_batches} attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            # exhausted retries
            print(f"Batch {b_idx}/{num_batches} failed after {MAX_RETRIES} attempts: {last_exc}")
            continue

        batch_elapsed = time.perf_counter() - batch_start

        if response.ok:
            batch_results = response.json().get("results", [])
            results.extend(batch_results)
            done = end
            percent = (done / num_rows) * 100
            print(
                f"Batch {b_idx}/{num_batches} OK in {batch_elapsed:.2f}s — "
                f"+{len(batch_results)} results (progress: {done}/{num_rows} = {percent:.1f}%)"
            )
        else:
            print(f"Batch {b_idx}/{num_batches} failed with status {response.status_code} in {batch_elapsed:.2f}s")
            print(response.text)

    api_total_elapsed = time.perf_counter() - api_total_start
    print(f"\nTotal results collected: {len(results)} (for {num_rows} sampled rows)")
    print(f"Total API time: {api_total_elapsed:.2f}s ({api_total_elapsed/60:.2f} min)\n")

# Optional: save snapshot for later reuse
if args.save_json:
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    blob = {
        "meta": {
            "seed": rand_seed,
            "sample_size": int(sample_size),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "api_model": "gpt-4o-mini-2024-07-18",
        },
        "truth": truth,
        "results": results,
    }
    with open(args.save_json, "w", encoding="utf-8") as jf:
        json.dump(blob, jf, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.save_json}")


# ============================= Build chain_lookup =============================

# (source, chapter:int, hadith:int) -> chain_indx
chain_lookup = {
    (t["source"], int(t["chapter_no"]), int(t["hadith_no"])): t["chain_indx"]
    for t in truth
    if t["hadith_no"] is not None
}
print(f"Chain lookup built with {len(chain_lookup)} entries")

# Preview: how many API keys match the truth (after normalization)
normalized_hits = sum(1 for r in results if (normalize_api_key(r) in chain_lookup))
print(f"Preview: {normalized_hits}/{len(results)} API results match truth keys (after normalization).")


# ============================= Rawis cache =============================

rawi_cache: dict[int, str] = {}

def get_rawi_name(scholar_id: int) -> str:
    """Cache-based lookup of rawi name by ID."""
    if scholar_id in rawi_cache:
        return rawi_cache[scholar_id]
    result = rawis_df[rawis_df['scholar_indx'] == scholar_id]
    if not result.empty:
        name = safe_str(result.iloc[0]['name'])
        rawi_cache[scholar_id] = name
        return name
    return ""


# ============================= Evaluation =============================

total_extracted = 0
total_ground_truth = 0
total_matched = 0
exact_match_count = 0

# A second structured log with a unique filename
os.makedirs("structured_results", exist_ok=True)
file_count = sum(
    1 for p in os.listdir("structured_results")
    if os.path.isfile(os.path.join("structured_results", p))
)
filename = f"{file_count+1}_{sample_size}_{rand_seed}.txt"
output_file_2 = os.path.join("structured_results", filename)

eval_start = time.perf_counter()
with open(output_file_1, "w", encoding="utf-8") as f1, open(output_file_2, "w", encoding="utf-8") as f2:
    header = "\n=== Comparison Results ===\n"
    print(header)
    f1.write(header)
    f2.write(header)

    for api_result in results:
        key = normalize_api_key(api_result)
        if key is None:
            msg = (
                f"❌ Unusable key from API result: "
                f"{api_result.get('source')}, {api_result.get('chapter_no')}, {api_result.get('hadith_no')}"
            )
            print(msg)
            f1.write(msg + "\n")
            f2.write(msg + "\n")
            continue

        chain_str = chain_lookup.get(key)
        if is_missing(chain_str):
            msg = f"❌ Chain not found for: {key}"
            print(msg)
            f1.write(msg + "\n")
            f2.write(msg + "\n")
            continue

        # Parse ground-truth narrator IDs -> names
        try:
            narrator_ids = [int(i) for i in safe_str(chain_str).split(',') if i.strip().isdigit()]
        except Exception:
            narrator_ids = []
        true_names = [get_rawi_name(i) for i in narrator_ids]

        # Extracted sanad (ensure list)
        extracted_sanad = api_result.get("sanad", []) or []
        if not isinstance(extracted_sanad, list):
            extracted_sanad = [safe_str(extracted_sanad)] if not is_missing(extracted_sanad) else []

        matches = 0
        unmatched_names = []
        prophet_names = []
        prophet_indices = set()  # which extracted indices should be ignored for metrics

        # 1) Deterministic token-subset match
        # 2) LLM fallback ("YES"/"PROPHET")
        for idx, name in enumerate(extracted_sanad):
            ex_tokens = safe_tokenize(name)
            found = False

            for full_name in true_names:
                true_tokens = safe_tokenize(full_name)
                # Treat the extracted snippet as a subset of tokens of the true name
                if ex_tokens and all(tok in true_tokens for tok in ex_tokens):
                    matches += 1
                    found = True
                    break

            if not found:
                llm_check_result = llm_name_match(name, true_names)
                if llm_check_result == "YES":
                    matches += 1
                    found = True
                elif llm_check_result == "PROPHET":
                    # Prophet: don't count as mismatch; exclude from denominator
                    found = True
                    prophet_names.append(name)
                    prophet_indices.add(idx)

            if not found:
                unmatched_names.append(name)

        # For metrics, remove prophet names from denominator
        final_extracted = [n for i, n in enumerate(extracted_sanad) if i not in prophet_indices]

        # Metrics accumulation
        total_extracted += len(final_extracted)
        unique_true_names = list({safe_str(n).strip() for n in true_names if safe_str(n).strip()})
        total_ground_truth += len(unique_true_names)
        total_matched += matches

        if matches == len(unique_true_names) and len(final_extracted) == len(unique_true_names):
            exact_match_count += 1

        # Per-hadith log entry
        output = (
            f"\n🔹 Hadith: {key}\n"
            f"Extracted sanad: {extracted_sanad}\n"
            f"Ground truth:    {true_names}\n"
            f"✅ Matched {matches}/{len(final_extracted)} narrators\n"
        )
        if prophet_names:
            output += f"ℹ️ Prophet name(s) detected & ignored: {prophet_names}\n"
        if unmatched_names:
            output += f"❌ Unmatched extracted names: {unmatched_names}\n"

        print(output)
        f1.write(output)
        f2.write(output)

    # Final metrics
    precision = total_matched / total_extracted if total_extracted else 0.0
    recall = total_matched / total_ground_truth if total_ground_truth else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match_rate = (exact_match_count / len(results)) if results else 0.0

    summary = (
        "\n=== Evaluation Metrics ===\n"
        f"🔸 Precision:         {precision:.2f}\n"
        f"🔸 Recall:            {recall:.2f}\n"
        f"🔸 F1 Score:          {f1_score:.2f}\n"
        f"🔸 Exact Match Rate:  {exact_match_rate:.2f} ({exact_match_count}/{len(results)})\n"
    )
    print(summary)
    f1.write(summary)
    f2.write(summary)

eval_elapsed = time.perf_counter() - eval_start
print(f"Evaluation time: {eval_elapsed:.2f}s")