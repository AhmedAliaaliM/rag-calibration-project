\# RAG Confidence Calibration \& Reliability Testing



\*\*Live dashboard:\*\* https://rag-calibration-project-dmgmgusj578cstdfigeewr.streamlit.app/



\## What this is



A testing framework built to answer one question about my \[RAG-based paper Q\&A system](https://github.com/AhmedAliaaliM/rag-project): \*\*can you trust its self-reported confidence scores?\*\*



Most RAG systems return an answer alongside a confidence label (HIGH/LOW), implying "trust me more when I say HIGH." This project builds the tooling to actually test that assumption — rather than take it on faith — using a 42-question labeled test set spanning answerable, unanswerable, and adversarial ("distractor") questions across 3 different LLMs.



\## Headline finding



\*\*Confidence is inversely related to reliability.\*\* Across all three models tested, answers marked LOW confidence were correct 87-96% of the time — while answers marked HIGH confidence were only correct 63-75% of the time.



| Model | Overall Accuracy | HIGH Confidence Accuracy | LOW Confidence Accuracy |

|---|---|---|---|

| llama3.2:3b | 76.2% | 63.2% | 87.0% |

| mistral:7b | 73.8% | 63.6% | 89.5% |

| llama-3.1-8b-instant | 85.7% | 75.0% | 95.5% |



In other words: \*\*a user who trusted only "HIGH confidence" answers would do worse than one who ignored the confidence label entirely.\*\* The pattern held consistently across every model tested — it isn't a quirk of one LLM's self-assessment, but appears to be a structural property of prompt-based, self-reported confidence in RAG pipelines.



\### The clearest example: a single fabricated source can override correct ones



A stress test injected one fake, contradictory chunk of text into the vector database alongside real paper content. When asked a question where both the real answer and the fake one were retrieved together, the system:

\- Picked the fabricated chunk over two real, correct ones

\- Reproduced the false claim word-for-word

\- Reported \*\*HIGH confidence\*\*, with no indication it noticed the contradiction



This shows the system has no mechanism for cross-checking retrieved sources against each other — it trusts whichever chunk sounds most directly relevant, real or not.



\## How this was built



1\. \*\*Test harness\*\* (`run\_harness.py`) — automates running a labeled question set through the RAG pipeline, logging answer/confidence/sources/latency per question. Supports swapping the generation backend (local Ollama models or Groq API) via a `--model` flag.

2\. \*\*Test set\*\* (`test\_questions.json`) — 42 questions: 20 answerable, 15 deliberately unanswerable (testing honest abstention), 7 "distractor" questions designed to tempt the system into conflating two different papers.

3\. \*\*Manual scoring\*\* (`scores.csv`) — each answer manually graded for correctness against its retrieved evidence, using a rubric that treats retrieval failure, incomplete answers, and honest abstention as three distinct outcomes (not all "wrong" the same way).

4\. \*\*Dashboard\*\* (`dashboard.py`) — a Streamlit app reading live from the results + scores CSVs, visualizing the confidence/accuracy relationship and surfacing specific failure examples.



\## Bugs found along the way (kept as part of the story, not cleaned up)



This repo includes the one-off diagnostic scripts used to investigate two real bugs discovered during testing, left as-is rather than removed:

\- `check\_embeddings.py` / `check\_other\_paper.py` — used to diagnose a corrupted PDF text-extraction issue (one paper's text was missing whitespace, degrading its embeddings and causing it to be spuriously retrieved for unrelated queries)

\- `delete\_broken\_chunks.py` — the fix for the above

\- `add\_poison\_chunk.py` / `remove\_poison\_chunk.py` — used for the adversarial stress test described above



\## Tech stack



Python, ChromaDB, sentence-transformers, cross-encoder reranking, Ollama (local LLMs), Groq API, Streamlit, pandas.



\## Running it locally



```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

streamlit run dashboard.py

```



Requires the underlying \[rag-project](https://github.com/AhmedAliaaliM/rag-project) repo available locally for `run\_harness.py` to generate new results (the dashboard itself only needs the CSVs already in this repo).

