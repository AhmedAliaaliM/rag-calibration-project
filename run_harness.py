"""
run_harness.py
Week 3-4: Test harness for RAG calibration project.

Runs a labeled test set through the existing rag-project pipeline,
extracts (answer, evidence, source, confidence), and logs everything
to a dataframe for later calibration analysis (Week 5).

Usage:
    python run_harness.py

Requires:
    - The existing rag-project repo available locally, with its
      src/ folder importable (adjust RAG_PROJECT_SRC / RAG_PROJECT_ROOT below).
    - Ollama running locally (`ollama serve`) if using the local model.
    - A test_questions.json file in the SAME folder as this script.
"""

import os
import sys
import re
import json
import time

import pandas as pd

# --- CONFIG: point these at your existing rag-project ---
RAG_PROJECT_ROOT = r"D:\project\rag-project"
RAG_PROJECT_SRC = os.path.join(RAG_PROJECT_ROOT, "src")
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


def load_test_questions(path):
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


def run_harness(test_questions, rag_project_root, top_k=10, rerank_n=3):
    """
    Runs the full retrieve -> rerank -> generate pipeline for each test question.

    IMPORTANT: we chdir into the RAG project root for the ENTIRE duration of
    this function (not just while loading the retriever), because rag_pipeline.py
    references "chroma_experiments" as a path relative to the current working
    directory. If we switch back too early, every retrieve() call afterward
    silently returns zero results instead of raising an error.
    """
    original_dir = os.getcwd()
    os.chdir(rag_project_root)

    try:
        print("Loading retriever (embedding model + Chroma collection)...")
        model, collection = load_retriever()
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
            retrieved_sources = list({c["source"] for c in reranked})

            results.append({
                "id": item.get("id", f"q_{i}"),
                "question": question,
                "category": item.get("category"),
                "expected_source": item.get("expected_source"),
                "ground_truth": item.get("ground_truth"),
                "retrieved_sources": retrieved_sources,
                **parsed,
                "latency_sec": round(elapsed, 2),
            })

            print(f"    -> retrieved={retrieved_sources}  confidence={parsed['confidence']}  ({elapsed:.1f}s)\n")

        return pd.DataFrame(results)

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    calibration_project_dir = os.getcwd()  # where this script + test_questions.json live

    test_questions_path = os.path.join(calibration_project_dir, "test_questions.json")
    test_questions = load_test_questions(test_questions_path)

    df = run_harness(test_questions, rag_project_root=RAG_PROJECT_ROOT)

    out_path = os.path.join(calibration_project_dir, "results_raw_v2_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")
