# legal-rag

A retrieval-augmented Q&A assistant over company ethics and compliance policy documents, built from scratch for a RAG lab: chunking, embeddings, vector search, hybrid retrieval, routing, reranking, an evaluation harness, source attribution, and a data-quality diagnosis.

## Corpus

The corpus is four publicly available Eli Lilly policy documents rather than LLM-generated ones, used with the instructor's approval. It contains a real data quality issue: an outdated 2021 Supplier Code sits alongside the current 2025 version.

## Workflow

- `main` is only changed through pull requests.
- One branch per lab step (`step-NN-name`).
- Test-first: each step has a `(red)` commit with failing tests, then a `(green)` commit that makes them pass.

## Seeing what was retrieved (Arize Phoenix)

Every pipeline stage (vector search, keyword search, fusion, rerank, LLM call) records a span with the chunks it returned, their scores, and each chunk's document version and status. Tracing is off unless `PHOENIX_TRACING=1` is set.

```
pip install -r requirements-dev.txt        # once: adds the small tracing client
uvx arize-phoenix serve                    # in a second terminal; leave it running
PHOENIX_TRACING=1 python -m rag.ask "What is the current version number of the Supplier Code of Business Conduct?"
```

Then open http://localhost:6006 and pick the `legal-rag` project. `PHOENIX_TRACING=1 python -m rag.evaluate --answers` sends a trace for every gold question.

The Phoenix server runs through `uvx` rather than from this project's environment, because it requires `anthropic<2` and this project uses a newer SDK.
