"""Step 9: tracing. Every pipeline stage becomes a span that Arize Phoenix can display.

These tests capture spans in memory; no Phoenix server is needed.
"""
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag import trace
from rag.ask import answer
from rag.retrieve import retrieve
from test_generate import FakeClient

QUESTION = "How many times a year must the ethics committee meet?"


@pytest.fixture
def spans(monkeypatch):
    """Point the pipeline's tracer at an in-memory exporter; returns a function that lists the finished spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "TRACER", provider.get_tracer("test"))
    return lambda: {s.name: s for s in exporter.get_finished_spans()}


def test_documents_lists_each_hit_with_id_text_score_and_lineage():
    hits = [{"chunk_id": "a:s:00", "text": "Some text.", "rerank_score": 0.9, "found_by": ["vector"],
             "metadata": {"title": "Policy", "section": "S", "version": "13.0", "status": "superseded"}}]
    attrs = trace.documents(hits, score=lambda h: h["rerank_score"])
    assert attrs["retrieval.documents.0.document.id"] == "a:s:00"
    assert attrs["retrieval.documents.0.document.content"] == "Some text."
    assert attrs["retrieval.documents.0.document.score"] == 0.9
    assert json.loads(attrs["retrieval.documents.0.document.metadata"]) == {
        "title": "Policy", "section": "S", "version": "13.0", "status": "superseded", "found_by": "vector"}


def test_documents_leaves_out_a_score_that_does_not_exist():
    hits = [{"chunk_id": "a", "text": "t", "metadata": {"title": "P", "section": "S"}}]
    assert "retrieval.documents.0.document.score" not in trace.documents(hits, score=lambda h: h.get("rrf_score"))


def test_setup_does_nothing_unless_switched_on(monkeypatch):
    monkeypatch.delenv("PHOENIX_TRACING", raising=False)
    assert trace.setup() is False


def test_retrieval_produces_one_span_per_stage(collection, spans):
    retrieve(collection, QUESTION, k=4)
    got = spans()
    assert {"vector_search", "keyword_search", "hybrid_search", "rerank"} <= set(got)
    kinds = {name: s.attributes["openinference.span.kind"] for name, s in got.items()}
    assert kinds["vector_search"] == kinds["keyword_search"] == kinds["hybrid_search"] == "RETRIEVER"
    assert kinds["rerank"] == "RERANKER"
    # the two searches happen inside the hybrid search
    assert got["vector_search"].parent.span_id == got["hybrid_search"].context.span_id
    assert got["keyword_search"].parent.span_id == got["hybrid_search"].context.span_id


def test_search_spans_show_the_question_and_the_chunks_with_their_status(collection, spans):
    retrieve(collection, QUESTION, k=4)
    vector = spans()["vector_search"].attributes
    assert vector["input.value"] == QUESTION
    assert vector["retrieval.documents.0.document.id"].startswith("ethics_compliance_committee_charter_2025:")
    assert "status" in json.loads(vector["retrieval.documents.0.document.metadata"])
    assert 0 < vector["retrieval.documents.0.document.score"] <= 1


def test_rerank_span_shows_candidates_in_and_top_chunks_out(collection, spans):
    hits = retrieve(collection, QUESTION, k=4)
    rerank = spans()["rerank"].attributes
    assert rerank["reranker.query"] == QUESTION and rerank["reranker.top_k"] == 4
    assert rerank["reranker.model_name"]
    assert "reranker.input_documents.4.document.id" in rerank       # more than four went in
    assert "reranker.output_documents.4.document.id" not in rerank  # exactly four came out
    assert rerank["reranker.output_documents.0.document.id"] == hits[0]["chunk_id"]
    assert rerank["reranker.output_documents.0.document.score"] == hits[0]["rerank_score"]


def test_an_answer_is_one_trace_from_question_to_llm_reply(collection, spans):
    answer(QUESTION, collection=collection, client=FakeClient("Four times a year."))
    got = spans()
    root, llm = got["rag.answer"], got["generate_answer"]
    assert root.parent is None
    assert root.attributes["input.value"] == QUESTION and root.attributes["output.value"] == "Four times a year."
    assert llm.attributes["openinference.span.kind"] == "LLM"
    assert llm.attributes["llm.model_name"] and llm.attributes["output.value"] == "Four times a year."
    assert len({s.context.trace_id for s in got.values()}) == 1
