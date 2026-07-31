# Can You Trust a RAG System's Confidence Score?

**A calibration and reliability study of self-reported confidence in retrieval-augmented generation**

**Live dashboard:** https://rag-calibration-project-dmgmgusj578cstdfigeewr.streamlit.app/
**Code:** https://github.com/AhmedAliaaliM/rag-calibration-project
**Underlying RAG system tested:** https://github.com/AhmedAliaaliM/rag-project

---

## 1. Motivation

Most RAG (retrieval-augmented generation) systems don't just answer questions — they also tell you how confident they are. My own RAG project, built over a summer of prior work, does exactly this: for every question, it returns an answer, the source it pulled from, and a confidence label of either **HIGH** or **LOW**.

The implicit promise of that confidence label is simple: *trust HIGH answers more than LOW ones.*

Nobody had actually tested whether that promise holds. So instead of building another RAG chatbot, I built a testing framework to answer one specific question: **when this system says HIGH confidence, is it actually more likely to be correct?**

This mattered to me for two reasons. First, most projects at this level demonstrate that you can *build* an LLM application — this project demonstrates that you can *evaluate* one rigorously, which is a different and arguably more valuable skill for an AI engineering role. Second, confidence miscalibration is a real, underexamined failure mode in production RAG systems — the kind of thing that causes real harm in domains like healthcare, law, or finance, where a user might reasonably filter for "only show me high-confidence answers."

## 2. What was tested

The system under test is a RAG pipeline with the following architecture:
- **Corpus:** 21 data science research papers (PDFs) covering varied topics — NLP, forecasting, computer vision, IoT, systems, explainability
- **Retrieval:** sentence-transformer embeddings (`all-MiniLM-L6-v2`) + ChromaDB vector store
- **Reranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) narrows candidates to the top 3 most relevant chunks
- **Generation:** a structured prompt instructs the LLM to respond with an ANSWER, EVIDENCE (a supporting quote), SOURCE, and a self-reported CONFIDENCE label (HIGH or LOW — the prompt does not offer a finer-grained scale)
- **Models tested:** `llama3.2:3b` and `mistral:7b` (both run locally via Ollama), and `llama-3.1-8b-instant` (via the Groq API)

## 3. Methodology

### 3.1 Test set design

I built a 42-question labeled test set across three categories, designed to probe different failure modes:

| Category | Count | Purpose |
|---|---|---|
| Answerable | 20 | Can the system correctly answer questions its own papers actually cover? |
| Unanswerable | 15 | Questions deliberately outside the corpus's scope. Correct behavior is honest abstention ("I don't know") — this is the category that best exposes overconfident hallucination. |
| Distractor | 7 | Questions that tempt the system into conflating two different, topically-adjacent papers (e.g., "does the concept drift paper use the same method as the IoT anomaly paper?"). |

A later, smaller batch (Section 6) added 4 more questions to probe two additional failure modes: answer *completeness* under confidence, and behavior when retrieved evidence is *internally contradictory*.

### 3.2 Test harness

I built an automated harness (`run_harness.py`) that feeds each test question through the full retrieval → rerank → generate pipeline and logs the answer, confidence, retrieved sources, and latency to a CSV — replacing what would otherwise be 42+ questions typed manually into a chat UI, one at a time. The harness supports swapping which model generates the answer via a command-line flag, so the same 42 questions could be run identically across all three models without duplicating code.

### 3.3 Scoring methodology

Each answer was manually scored for correctness against its own retrieved evidence (not against my own outside knowledge of the papers, since I hadn't read all 21 in full). This is an important methodological choice: I was scoring **faithfulness to the paper**, not just "does this sound plausible."

The rubric went through one important revision. My first scoring pass was too lenient — it credited vague, incomplete, or wrong-but-plausible-sounding answers as "correct" if they were superficially consistent with the retrieved text. On review, I tightened the rubric to distinguish three genuinely different outcomes that a looser pass had conflated:

1. **Honest abstention** — the system correctly recognized it didn't have enough information and said so. This is *correct* behavior.
2. **Retrieval failure** — the system said "insufficient information," but the answer actually existed in the corpus; the retrieval step simply failed to find the right document. This is *incorrect* — it's a different failure from genuine abstention, even though both produce the same "I don't know" output.
3. **Incomplete or partially fabricated answers** — the system produced a fluent, confident-sounding answer that was missing key details or included details not actually present in the retrieved evidence. Scored *incorrect*, even when partially right.

Re-scoring under this stricter rubric dropped overall accuracy from an initial 85.7% to 73.8% for the first model tested — but importantly, **the core finding survived the stricter scoring**, which is a stronger result than if it had only shown up under generous grading.

## 4. Two real bugs found along the way

Before any calibration analysis was possible, two genuine bugs surfaced during testing — both are documented in the repo with the diagnostic scripts used to find them.

**Bug 1: Corrupted PDF text extraction.** Early test runs showed one paper (`2601.03085v1.pdf`) being retrieved for almost every question, regardless of topic — hate speech questions, federated learning questions, SHAP questions, all pulling from the same unrelated IoT anomaly detection paper. Diagnosis (checking chunk counts, then raw embedding vectors, then raw text) revealed the paper's text had been extracted with all whitespace stripped (`©IEEE.Thisistheauthor'sacceptedmanuscript...`), degrading its embeddings into a strange, non-specific region of embedding space that falsely matched unrelated queries. The fix was to remove that paper's chunks from the vector store.

**Bug 2: A working-directory bug in my own harness script.** After the fix above, a full re-run returned **zero retrieved sources for all 42 questions** — yet the LLM still answered fluently every time, fabricating citations to papers that don't exist in the corpus ("Fake News Detection - A Survey (2020)", "spark-2014.pdf") and inventing specific false claims (a fabricated 15% stock price increase from predictive maintenance) at HIGH confidence. The root cause was a relative file path in the underlying pipeline that broke when the current working directory changed too early in my script. This was, on its own, one of the most useful accidental findings in the project: it's a clean demonstration that a RAG system with completely broken retrieval gives no external signal that anything is wrong — the fluency of the output is identical whether retrieval works or is completely dead.

## 5. Core finding: confidence is inversely related to reliability

After both bugs were fixed, a clean 42-question run was scored for each of the three models. The result was consistent, and counterintuitive:

| Model | Overall Accuracy | HIGH Confidence Accuracy | LOW Confidence Accuracy |
|---|---|---|---|
| llama3.2:3b | 76.2% | 63.2% | 87.0% |
| mistral:7b | 73.8% | 63.6% | 89.5% |
| llama-3.1-8b-instant | 85.7% | 75.0% | 95.5% |

**In every single model tested, LOW-confidence answers were meaningfully more reliable than HIGH-confidence answers** — by a margin of 20 to 26 percentage points. A well-calibrated system should show the opposite relationship (HIGH should mean "trust more"). Here, a user who filtered for only HIGH-confidence answers would systematically end up with a *less* reliable subset of answers than one who ignored the confidence label entirely.

This pattern was strongest and clearest in the **unanswerable** question category specifically. Across all models, every time the system claimed HIGH confidence on a genuinely out-of-scope question, it was wrong — it had misused an unrelated retrieved chunk to fabricate a plausible-sounding but false answer. Two representative examples:

- Asked which platform banned a fake news detection model from production use, one model retrieved a chunk that merely *mentioned* Facebook in an unrelated context, then confidently fabricated a specific claim built around that mention — HIGH confidence.
- Asked about a publication's peer-review rejection rate (a nonsensical question relative to the corpus), a model retrieved an unrelated statistic from a completely different paper (a 55.7% figure about data preprocessing) and confidently repurposed it as an answer to a different question entirely — HIGH confidence.

Every time these same models said LOW confidence on an unanswerable question, they were correct.

### Why this happens

The confidence label isn't measuring "how likely am I to be right." It's better understood as measuring something closer to "how fluent and internally consistent does my own answer sound to me." These two things usually align — but they diverge in a specific, dangerous way: when retrieval genuinely fails to find anything relevant, models tend to notice and hedge (LOW). But when retrieval returns something *topically adjacent but wrong*, models often don't detect the mismatch, and confidently construct an answer from it anyway — labeling that fluency as HIGH confidence, even though the underlying retrieval failed just as completely as in the "nothing found" case.

### Why the cross-model consistency matters

The fact that this pattern held at nearly identical magnitude across three different model families (Llama 3.2, Mistral, and Llama 3.1 via a different serving stack) is the difference between "one model is bad at self-assessment" and "this is a structural property of prompt-based, self-reported confidence in RAG systems generally." The latter is a much stronger and more generalizable claim.

One encouraging note: `llama-3.1-8b-instant` was both the most accurate model overall (85.7%) *and* the best-calibrated (the smallest HIGH/LOW gap, and the only model with zero hallucinations on unanswerable questions). This suggests model capability and calibration quality may be linked — a more capable model was less likely to confidently paper over a retrieval gap.

## 6. Stress test: a single fabricated source overrides correct ones

To probe a different, more adversarial failure mode, I ran a small follow-up test. I inserted one deliberately false, self-contradictory chunk into the real vector store (a fabricated claim that "BERTopic was introduced in 2010 by Google as an extension of Word2Vec, predating BERT itself" — notably impossible on its face, since BERTopic derives its name from BERT). I then asked a question about BERTopic's origins, removing the fake chunk immediately afterward.

The retrieval step pulled back **three chunks**: two real ones from the actual BERTopic paper and a related paper, plus the single fabricated chunk. Despite having correct information available in the same context, the system:

1. Selected the fabricated chunk over the two real ones
2. Reproduced the false claim verbatim
3. Reported **HIGH confidence**, with no indication it detected any contradiction between sources

This is arguably the single most concrete finding in the project. It demonstrates that the system has no mechanism for cross-referencing or reconciling contradictory information across its retrieved sources — it will confidently report whichever single chunk seems most directly relevant to the question, real or fabricated, with nothing resembling source-triangulation or skepticism.

A smaller companion test (3 questions explicitly demanding *complete* lists of features/constraints, rather than open-ended questions) reinforced the same theme: 2 of 3 produced incomplete or partially fabricated answers at HIGH confidence, even when the question's phrasing explicitly asked for completeness.

## 7. Deliverables

- **Test harness** (`run_harness.py`) — automated, model-agnostic runner supporting local Ollama models and the Groq API
- **42-question labeled test set** (`test_questions.json`) spanning answerable, unanswerable, and distractor categories
- **Manual scoring dataset** (`scores.csv`) with a documented, revised rubric distinguishing abstention, retrieval failure, and incompleteness as separate outcomes
- **Live interactive dashboard** (Streamlit, deployed publicly) visualizing the confidence/accuracy relationship across all three models, with a browsable failure gallery
- **A documented bug-discovery trail** — two real, non-trivial bugs found and fixed during testing, with the diagnostic scripts left in the repo

## 8. Limitations and honest caveats

- **Confidence is binary in this system** (HIGH/LOW only), which limits how fine-grained a calibration analysis can be — a system with a continuous 0-100 confidence scale would support metrics like Expected Calibration Error (ECE) and reliability diagrams. That was a deliberate scope decision given the underlying system's design, not an oversight, but it's worth naming as a constraint on what these results can say.
- **Scoring was manual**, done by one person (me), rather than by multiple independent raters or an automated LLM-as-judge system. I mitigated this by revising my own rubric after a critical second look, and by being explicit about the rubric's rules — but inter-rater reliability wasn't formally measured.
- **The test set (42 questions) is modest in size.** The consistency of the finding across 3 separate models is reassuring, but a larger test set would allow more statistically rigorous claims (e.g., confidence intervals on the accuracy gap).
- **The corpus is narrow** — 21 data science papers on a specific range of topics. Whether this finding generalizes to other domains (legal documents, medical literature, customer support corpora) is untested.

## 9. What I'd do next with more time

- Redesign the confidence-elicitation prompt to request a continuous 0-100 score rather than a binary label, and re-test whether finer granularity improves calibration or just adds false precision
- Build an automated retrieval-quality check (e.g., a lightweight relevance classifier comparing the question to retrieved chunks) as an independent signal, separate from the LLM's own self-reported confidence — directly motivated by the finding that the LLM's confidence doesn't track retrieval quality
- Expand the test set size and add inter-rater scoring to strengthen the statistical claims

## 10. Conclusion

This project set out to answer a simple, practical question about a system I had already built: can its confidence scores be trusted? The answer, tested rigorously across three different models, was no — and more specifically, the confidence label was found to be **inversely** related to reliability, a more serious and more interesting failure than simple miscalibration. The clearest demonstration of why this happens came from a small adversarial test: the system will confidently repeat a single fabricated claim over two correct ones sitting in the same context, with no mechanism to notice the contradiction.

The practical implication is direct: in a RAG system like this one, a stated HIGH confidence should not be read as a green light. If anything, this data suggests the opposite — LOW confidence responses in this system were the ones worth trusting.
