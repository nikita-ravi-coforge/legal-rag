# legal-rag

A retrieval-augmented Q&A assistant over company ethics and compliance policy documents, built from scratch for a RAG lab: chunking, embeddings, vector search, hybrid retrieval, routing, reranking, an evaluation harness, source attribution, and a data-quality diagnosis.

## Corpus

The corpus is four publicly available Eli Lilly policy documents rather than LLM-generated ones, used with the instructor's approval. It contains a real data quality issue: an outdated 2021 Supplier Code sits alongside the current 2025 version.

## Workflow

- `main` is only changed through pull requests.
- One branch per lab step (`step-NN-name`).
- Test-first: each step has a `(red)` commit with failing tests, then a `(green)` commit that makes them pass.
