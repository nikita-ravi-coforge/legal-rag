"""Step 10: the planted data quality issue, and its fix.

A superseded 2021 Supplier Code is indexed next to the current 2025 one. The
fix leaves retrieval and generation alone and removes the bad source: by
default only chunks from current documents are searched.
"""
from rag.ask import answer
from rag.diagnose import QUESTION, diagnose
from rag.evaluate import answer_is_correct
from rag.retrieve import hybrid_search, retrieve
from test_generate import FakeClient


def statuses(hits):
    return {h["metadata"]["status"] for h in hits}


def test_the_question_surfaces_superseded_chunks_when_every_document_is_searched(collection):
    hits = retrieve(collection, QUESTION, k=4, current_only=False)
    assert "superseded" in statuses(hits)
    assert hits[0]["metadata"]["doc_id"] == "supplier_code_2021"  # the old version outranks the current one


def test_retrieval_uses_only_current_documents_by_default(collection):
    hits = retrieve(collection, QUESTION, k=4)
    assert statuses(hits) == {"current"}
    assert hits[0]["chunk_id"] == "supplier_code_2025:document-version:00"


def test_both_halves_of_hybrid_search_are_filtered(collection):
    hits = hybrid_search(collection, QUESTION, k=20)
    assert statuses(hits) == {"current"}
    assert {"vector", "keyword"} <= {name for h in hits for name in h["found_by"]}
    assert "superseded" in statuses(hybrid_search(collection, QUESTION, k=20, current_only=False))


def test_the_answer_pipeline_can_still_reproduce_the_flaw(collection):
    client = FakeClient()
    flawed = answer(QUESTION, collection=collection, client=client, current_only=False)
    fixed = answer(QUESTION, collection=collection, client=client)
    assert "superseded" in statuses(flawed["hits"])
    assert statuses(fixed["hits"]) == {"current"}
    assert "Version 13.0" in client.calls[0]["messages"][0]["content"]      # the old text reached the LLM
    assert "Version 13.0" not in client.calls[1]["messages"][0]["content"]  # and no longer does


def test_diagnose_returns_the_before_and_after_runs(collection):
    report = diagnose(collection=collection, client=FakeClient("An answer."))
    assert "superseded" in statuses(report["before"]["hits"])
    assert statuses(report["after"]["hits"]) == {"current"}
    assert report["before"]["answer"] == report["after"]["answer"] == "An answer."


def test_an_answer_that_repeats_superseded_information_is_marked_wrong():
    q = {"expected_sources": [{}], "expected_keywords": ["15.0"], "forbidden_keywords": ["13.0"]}
    assert answer_is_correct("The current version is V15.0.", q)
    assert not answer_is_correct("V15.0 is the newest; Version 13.0 also appears.", q)
