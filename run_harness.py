"""
run_harness.py
Week 3-4: Test harness for RAG calibration project.

Runs a labeled test set through the existing rag-project pipeline,
extracts (answer, evidence, source, confidence), and logs everything
to a dataframe for later calibration analysis (Week 5).

Usage:
    python run_harness.py

Requires:
    - The existing rag-project repo cloned/available locally, with its
      src/ folder importable (adjust RAG_PROJECT_SRC below).
    - Ollama running locally (`ollama serve`) if using the local model.
    - A test_questions.json file (see load_test_questions below) with
      your labeled question set from Week 2.
"""

import sys
import re
import json
import time
from pathlib import Path

import pandas as pd

# --- CONFIG: point this at your existing rag-project's src/ folder ---
RAG_PROJECT_SRC = r"D:\project\rag-project\src"   # <-- update to your actual path
RAG_PROJECT_ROOT = r"D:\project\rag-project"
sys.path.insert(0, RAG_PROJECT_SRC)

from rag_pipeline import load_retriever, retrieve, generate_answer  # noqa: E402
from reranker import rerank  # noqa: E402
from prompt_templates import build_prompt_v2 as build_prompt  # noqa: E402


def extract_fields(raw_answer: str) -> dict:
    """Parse ANSWER / EVIDENCE / SOURCE / CONFIDENCE out of the LLM's raw text."""

    def extract(field, text):
        match = re.search(rf"{field}:\s*(.+?)(?=\n[A-Z]+:|\Z)", text, re.DOTALL)
        return match.group(1).strip() if match else None

    return {
        "raw_answer": raw_answer,
        "answer": extract("ANSWER", raw_answer),
        "evidence": extract("EVIDENCE", raw_answer),
        "source": extract("SOURCE", raw_answer),
        "confidence": extract("CONFIDENCE", raw_answer),
    }


def load_test_questions(path="test_questions.json"):
    """
    Expected format — a list of dicts, one per test question:
    [
      {
        "id": "aq_01",
        "question": "What dataset was used in the federated learning paper?",
        "category": "answerable",       # answerable | unanswerable | distractor | conflicting
        "expected_source": "2106.09592v2.pdf",  # optional, for scoring later
        "ground_truth": "..."           # optional, for scoring later
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_harness(test_questions, top_k=10, rerank_n=3):
    import os
    print("Loading retriever (embedding model + Chroma collection)...")
    original_dir = os.getcwd()
    os.chdir(RAG_PROJECT_ROOT)   # so relative "chroma_experiments" path resolves correctly
    try:
        model, collection = load_retriever()
    finally:
        os.chdir(original_dir)   # switch back so test_questions.json etc. still resolve correctly
    print(f"Loaded. Running {len(test_questions)} test questions.\n")

    results = []
    for i, item in enumerate(test_questions, 1):
        question = item["question"]
        print(f"[{i}/{len(test_questions)}] {question}")

        t0 = time.time()
        chunks = retrieve(question, model, collection, top_k=top_k)
        reranked = rerank(question, chunks, top_n=rerank_n)
        prompt = build_prompt(question, reranked)
        raw_answer = generate_answer(prompt)
        elapsed = time.time() - t0

        parsed = extract_fields(raw_answer)

        results.append({
            "id": item.get("id", f"q_{i}"),
            "question": question,
            "category": item.get("category"),
            "expected_source": item.get("expected_source"),
            "ground_truth": item.get("ground_truth"),
            "retrieved_sources": list({c["source"] for c in reranked}),
            **parsed,
            "latency_sec": round(elapsed, 2),
        })

        print(f"    -> confidence={parsed['confidence']}  ({elapsed:.1f}s)\n")

    return pd.DataFrame(results)


if __name__ == "__main__":
    test_questions = load_test_questions("test_questions.json")
    df = run_harness(test_questions)

    out_path = "results_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")
