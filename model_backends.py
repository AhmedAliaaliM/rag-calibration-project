"""
model_backends.py
Week 6: Additional generation backends for multi-model comparison.

Keeps the original rag-project's generate_answer() (Ollama, llama3.2:3b)
untouched. This module adds:
  - generate_answer_ollama(prompt, model_name) -> works with ANY local
    Ollama model (e.g. "llama3.2:3b", "mistral:7b")
  - generate_answer_groq(prompt, model_name) -> calls Groq's free API
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory

OLLAMA_URL = "http://localhost:11434/api/generate"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_answer_ollama(prompt: str, model_name: str) -> str:
    """Same logic as the original project's generate_answer(), but works
    with any Ollama model name, not just the hardcoded one."""
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Run 'ollama serve' in a separate terminal."
    except Exception as e:
        return f"ERROR: {e}"


def generate_answer_groq(prompt: str, model_name: str = "llama-3.1-8b-instant") -> str:
    import time
    time.sleep(5)  # stay under Groq's free-tier rate limit
    if not GROQ_API_KEY:
        return "ERROR: GROQ_API_KEY not found. Check your .env file."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"
