# legal-rag

A retrieval-augmented Q&A assistant over company ethics and compliance policy documents, built from scratch for a RAG lab: chunking, embeddings, vector search, hybrid retrieval, reranking, an evaluation harness, a data quality diagnosis, and source attribution.

## Corpus

The corpus is four publicly available Eli Lilly policy documents rather than LLM-generated ones, used with the instructor's approval.

| Document | Version | Status |
|---|---|---|
| Ethics and Compliance Committee Charter | 2025-03-01 | current |
| Red Book: Code of Business Conduct | 2026 | current |
| Lilly Supplier Code of Business Conduct | 15.0 (2025) | current |
| Lilly Supplier Code of Business Conduct | 13.0 (2021) | **superseded** |

The planted data quality issue is the outdated 2021 Supplier Code, which sits alongside the current 2025 version with conflicting details. `data/manifest.json` records each document's source file, hash, version and status.

## Pipeline

```
PDFs -> markdown + manifest -> section-aware chunks -> embeddings in Chroma
question -> vector search + keyword search -> rank fusion -> cross-encoder rerank -> Claude -> answer + sources
```

| Stage | Module | Notes |
|---|---|---|
| Ingest | `src/rag/ingest.py` | PDF to markdown, duplicate detection, lineage manifest |
| Chunking | `src/rag/chunking.py` | Never crosses a section; at most 180 words; one-block overlap. Rationale in the module docstring |
| Embedding | `src/rag/embed.py` | `BAAI/bge-large-en-v1.5` (sentence-transformers) |
| Vector store | `src/rag/store.py` | Chroma, cosine distance |
| Keyword search | `src/rag/keyword.py` | Substring match, rarer terms weigh more |
| Hybrid + rerank | `src/rag/retrieve.py`, `src/rag/rerank.py` | Reciprocal Rank Fusion, then `BAAI/bge-reranker-base`. Current documents only |
| Generation | `src/rag/generate.py` | Claude, restricted to the retrieved excerpts, with numbered citations |
| End to end | `src/rag/ask.py` | Answer plus the sources it cited |
| Evaluation | `src/rag/evaluate.py`, `data/gold_set.json` | 24 fixed questions |
| Tracing | `src/rag/trace.py` | Optional, for Arize Phoenix |

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add ANTHROPIC_API_KEY
python -m rag.index           # build the vector index (not stored in git)
```

The first run downloads the embedding and reranker models (about 2.4 GB together).

## Commands

| Command | What it shows |
|---|---|
| `python -m rag.ingest` | PDFs to markdown, with the lineage manifest |
| `python -m rag.minimal_loop` | The smallest embed-store-retrieve loop, on two known texts |
| `python -m rag.index --show 3` | Chunk, embed and store every document |
| `python -m rag.view_chunks` | Every chunk as a browsable HTML page |
| `python -m rag.ask "question"` | The full pipeline: retrieved chunks, answer, sources |
| `python -m rag.ask --include-superseded "question"` | Same, but also searching the superseded document |
| `python -m rag.compare "question"` | Vector-only next to hybrid retrieval |
| `python -m rag.evaluate --answers` | Scores every pipeline variant against the gold set |
| `python -m rag.diagnose` | The planted data issue: flawed answer, then the corrected one |
| `pytest -v` | All tests, including the evaluation harness |

## Results

Gold set of 24 questions across 12 scenario types (`python -m rag.evaluate --answers`):

| Retrieval variant | Recall@4 | Stale rate |
|---|---|---|
| Vector only | 95% | 22% |
| Hybrid (vector + keyword) | 95% | 21% |
| Hybrid + rerank | 100% | 26% |
| Hybrid + rerank, current documents only (shipped) | 100% | 0% |

Answer accuracy 100%, source attribution 100%.

- **Recall@4**: the retrieved chunks from the expected document and section contain the expected answer keywords.
- **Stale rate**: share of retrieved chunks that come from the superseded document.
- **Attribution**: a source the answer cited is one of the expected sources.

Full results, including every generated answer, are in `results/eval_results.json`.

## Data quality diagnosis

The written diagnosis is in [`docs/diagnosis.md`](docs/diagnosis.md). In short: with every document searched, the question "What is the current version number of the Supplier Code?" retrieved mostly 2021 chunks and the answer could not say which version was current. Retrieval and generation both behaved correctly; the cause was the superseded document in the index. Restricting the search to current documents fixes the answer with no change to retrieval or generation logic.

## Seeing what was retrieved (Arize Phoenix)

Every pipeline stage (vector search, keyword search, fusion, rerank, LLM call) records a span with the chunks it returned, their scores, and each chunk's document version and status. Tracing is off unless `PHOENIX_TRACING=1` is set.

```
pip install -r requirements-dev.txt        # once: adds the small tracing client
uvx arize-phoenix serve                    # in a second terminal; leave it running
PHOENIX_TRACING=1 python -m rag.ask "What is the current version number of the Supplier Code of Business Conduct?"
```

Then open http://localhost:6006 and pick the `legal-rag` project. `PHOENIX_TRACING=1 python -m rag.evaluate --answers` sends a trace for every gold question.

The Phoenix server runs through `uvx` rather than from this project's environment, because it requires `anthropic<2` and this project uses a newer SDK.

## Workflow

- `main` is only changed through pull requests.
- One branch per lab step (`step-NN-name`).
- Test-first: each step has a `(red)` commit with failing tests, then a `(green)` commit that makes them pass.
- CI (`.github/workflows/ci.yml`) runs the full test suite and the retrieval evaluation on every pull request. Tests that call the LLM run only when the `ANTHROPIC_API_KEY` repository secret is set.
