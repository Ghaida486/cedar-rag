# Cedar Lab RAG challenge

PS05 teaching project using only the six fictional handout paragraphs. Participants: Ghaida Aljuhani and Mohammad Alnasser. See SOLUTION.txt for interpretations and remaining submission steps.

## Run

From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2
.venv/bin/python embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2 --inspect
.venv/bin/python run_experiments.py
```

The first embedding run downloads the public model to `.cache/` and needs internet. Subsequent runs can reuse it. The model runs on CPU without an API key. `run_experiments.py` overwrites generated evidence and resets judgment columns in results.csv to blank. Preserve completed judgments before rerunning.

## How it works

`keyword_retriever.py` lines 9–19 load the three files and separate IDs, filenames and paragraph text. Line 23 lowercases text, extracts alphanumeric words and removes exactly the specified stopwords. Sets remove duplicates. Lines 29–30 count distinct shared words; line 31 ranks by descending score then ascending ID. Line 39 selects at most k positive-scoring passages. No stemming or synonyms are used.

Before running Q1, expected required evidence was A1 and A2. The actual keyword result selects both. A1 shares `00`, `graduate`, `trained` (score 3). A2 shares `laser`, `trained` (score 2). `student` does not match `students`; `operate` does not match `operating`. Time punctuation creates a `00` token. These are intentional consequences of the baseline.

`embedding_retriever.py` loads one pretrained model, encodes the six passage texts once, and encodes each question in the same compatible vector space. IDs and filenames are metadata, never model input. Each vector contains 384 coordinates. Normalization makes each vector approximately unit length. Multiplying matching coordinates and summing gives cosine similarity. Sorting six scores and taking the first k selects the evidence. The first 12 displayed coordinates are only an inspection window; retrieval uses all 384. A single coordinate has no reliable semantic label.

Embeddings always return top k here, even for Q4, whose answer is absent. Similarity is not a guarantee that a passage answers the question. Keyword and embedding score scales are different and should not be compared numerically.

## Results and answer audits

The evidence judgments and observations in results.csv have been filled in from the actual runs; review them against the saved evidence. For Q1, coverage counts retrieved A1 and A2 divided by two; Q2 needs A2 and Q3 needs T1. Q4 needs an absence check, not a nearest-neighbor success label.

The user completed the two fresh restricted-evidence answer chats and supplied their actual responses. Both answers were audited. Q1 needed one instruction revision to explicitly acknowledge missing access-hour evidence; the original and revised answers are preserved. Q4 passed on its first run. See the answer and audit files in evidence/.

Individual reflection files are reflection_ghaida.txt (Ghaida Aljuhani) and reflection_mohammad.txt (Mohammad Alnasser), each below 150 words. Each participant must ensure the final text is their own explanation, as required by the handout. Repository: https://github.com/Ghaida486/cedar-rag. Submission to the class signup sheet remains a separate step.

## Verified environment and actual run

Python 3.14.7; sentence-transformers 6.1.0; numpy 2.5.4; torch 2.14.1; transformers 5.20.2. Full dependency versions are pinned in requirements.txt. MiniLM loaded successfully on CPU. The actual document matrix was (6, 384), query vector (384,), and all normalization and cosine assertions passed.

Q1 embedding order was A2, T1, B1, A1, T2, B2. The k=1, k=2 and k=3 experiments all recovered only one of the two required passages. Extra passages supplied training and booking conditions without supplying the access-hours passage. These observations are recorded in results.csv and SOLUTION.txt.

## Rubric evidence map

| Criterion | Files to inspect |
| --- | --- |
| Visible embedding process (3 points) | embedding_retriever.py; evidence/embedding-inspection.txt; evidence/environment.txt |
| Retrieval comparison (3 points) | results.csv; evidence/retrieval-runs.txt; evidence/k-experiment.txt; SOLUTION.txt sections 1–3 |
| Grounded answer audit (2 points) | evidence/q1-answer-original.txt; evidence/q1-answer-revised.txt; evidence/q1-instruction-revision.txt; evidence/q1-answer-audit.txt; evidence/q4-answer-original.txt; evidence/q4-answer-audit.txt |
| Reproducibility and explanation (2 points) | docs/; keyword_retriever.py; embedding_retriever.py; run_experiments.py; requirements.txt; this README; both reflection files |

## Reproduce from GitHub

```sh
git clone https://github.com/Ghaida486/cedar-rag.git
cd cedar-rag
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Use the Run commands above to reproduce the recorded outputs. The verified environment used Python 3.14.7 on macOS ARM64; other Python versions and platforms have not been verified. Dependencies and the public model must be downloadable on the first run. The small experiment is an evidence audit, not a reliable real-world accuracy estimate.
