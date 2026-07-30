"""
run_harness.py
Week 3-6: Test harness for RAG calibration project. Now supports
multiple generation backends for model comparison (Week 6).

Usage:
    python run_harness.py --model ollama:llama3.2:3b
    python run_harness.py --model ollama:mistral:7b
    python run_harness.py --model groq:llama-3.1-8b-instant

Requires:
    - The existing rag-project repo available locally (adjust
      RAG_PROJECT_ROOT below).
    - Ollama running locally (`ollama serve`) for ollama: models.
    - A GROQ_API_KEY in a local .env file for groq: models.
    - test_questions.json in the SAME folder as this script.
    - model_backends.py in the SAME folder as this script.
"""

import os
import sys
import re
import json
import time
import argparse

import pandas as pd

# --- CONFIG ---
RAG_PROJECT_ROOT = r"D:\project\rag-project"
RAG_PROJECT_SRC = os.path.join(RAG_PROJECT_ROOT, "src")
sys.path.insert(0, RAG_PROJECT_SRC)

from rag_pipeline import load_retriever, retrieve  # noqa: E402
from reranker import rerank  # noqa: E402
from prompt_templates import build_prompt_v2 as build_prompt  # noqa: E402

# our own additional backends (Groq + arbitrary Ollama models)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_backends import generate_answer_ollama, generate_answer_groq  # noqa: E402


def get_generator(model_spec: str):
    """
    model_spec looks like "ollama:llama3.2:3b" or "groq:llama-3.1-8b-instant".
    Returns a function(prompt) -> answer_text, and a clean model name for logging.
    """
    backend, _, model_name = model_spec.partition(":")
    if backend == "ollama":
        return (lambda prompt: generate_answer_ollama(prompt, model_name)), model_name
    elif backend == "groq":
        return (lambda prompt: generate_answer_groq(prompt, model_name)), model_name
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'ollama:<model>' or 'groq:<model>'.")


def extract_fields(raw_answer: str) -> dict:
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_harness(test_questions, rag_project_root, generate_fn, top_k=10, rerank_n=3):
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
            raw_answer = generate_fn(prompt)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ollama:llama3.2:3b",
        help="e.g. ollama:llama3.2:3b, ollama:mistral:7b, groq:llama-3.1-8b-instant",
    )
    parser.add_argument(
        "--questions",
        default="test_questions.json",
        help="Filename of the question set to run (must be in the same folder as this script).",
    )
    args = parser.parse_args()

    generate_fn, model_name = get_generator(args.model)
    safe_model_name = model_name.replace(":", "-").replace(".", "_")

    calibration_project_dir = os.getcwd()
    test_questions_path = os.path.join(calibration_project_dir, args.questions)
    test_questions = load_test_questions(test_questions_path)

    questions_tag = os.path.splitext(args.questions)[0]

    print(f"=== Running harness with model: {args.model}, questions: {args.questions} ===\n")
    df = run_harness(test_questions, rag_project_root=RAG_PROJECT_ROOT, generate_fn=generate_fn)
    df["model"] = model_name

    out_path = os.path.join(calibration_project_dir, f"results_{safe_model_name}_{questions_tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")
