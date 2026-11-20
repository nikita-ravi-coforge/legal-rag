"""Tracing: record each pipeline stage as a span, to view in Arize Phoenix.

A trace is one question travelling through the pipeline. Each stage (vector
search, keyword search, fusion, rerank, LLM call) is a span that records what
went in and which chunks came out, with their scores and document status.

Tracing is off by default. Without setup() the tracer below is a no-op, so the
`with span(...)` lines in the pipeline cost nothing in tests and CI.

To see traces:
  1. start Phoenix in another terminal:   uvx arize-phoenix serve
  2. run with tracing on:                 PHOENIX_TRACING=1 python -m rag.ask "..."
  3. open http://localhost:6006

Attribute names follow the OpenInference convention, which Phoenix reads to
show retrieved documents and scores.
"""
import json
import os
from contextlib import contextmanager

from opentelemetry import trace

TRACER = trace.get_tracer("rag")
PROJECT = "legal-rag"


def setup():
    """Start sending spans to Phoenix, if PHOENIX_TRACING=1. Returns whether tracing is on."""
    if os.getenv("PHOENIX_TRACING") != "1":
        return False
    from phoenix.otel import register  # from requirements-dev.txt; only needed when tracing is on
    register(project_name=PROJECT, verbose=False)
    return True


@contextmanager
def span(name, kind, question):
    """One pipeline stage. `kind` is an OpenInference span kind: CHAIN, RETRIEVER, RERANKER or LLM."""
    attributes = {"openinference.span.kind": kind, "input.value": question}
    with TRACER.start_as_current_span(name, attributes=attributes) as current:
        yield current


def documents(hits, score, prefix="retrieval.documents"):
    """Hits -> span attributes: each chunk's id, text, score and lineage (version, status)."""
    attributes = {}
    for i, hit in enumerate(hits):
        key = f"{prefix}.{i}.document"
        metadata = {k: hit["metadata"][k] for k in ("title", "section", "version", "status") if k in hit["metadata"]}
        if "found_by" in hit:
            metadata["found_by"] = " + ".join(hit["found_by"])
        attributes[f"{key}.id"] = hit["chunk_id"]
        attributes[f"{key}.content"] = hit["text"]
        attributes[f"{key}.metadata"] = json.dumps(metadata)
        if score(hit) is not None:
            attributes[f"{key}.score"] = float(score(hit))
    return attributes
