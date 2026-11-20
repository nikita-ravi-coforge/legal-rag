"""Step 6: the gold dataset and the evaluation harness.

Three groups of tests:
  1. The gold set itself is sound: it covers every scenario and every expected
     answer really is in the documents.
  2. The scoring functions work, on small fake inputs.
  3. The pipeline is scored against the gold set: one retrieval test per
     question, recall floors per retriever, and one live answer test per question.
"""
import pytest

from rag.evaluate import (BEST, RETRIEVERS, answer_is_correct, answer_rank, evaluate_answers,
                          evaluate_retrieval, load_gold)
from rag.generate import NOT_COVERED

GOLD = load_gold()
ANSWERABLE = [q for q in GOLD if q["expected_sources"]]

SCENARIOS = {"semantic", "exact_term", "exact_phrase", "section_name", "paraphrase", "hard_paraphrase", "multi_chunk",
             "cross_document", "data_quality", "current_version", "out_of_scope", "unanswerable"}

# Recall@4 measured when each retriever was added. A change that drops below this fails.
RECALL_FLOOR = {"vector": 0.95, "hybrid": 0.95, "hybrid+rerank": 1.0, "current only": 1.0}


def source(hit_or_chunk):
    m = hit_or_chunk["metadata"]
    return {"doc_id": m["doc_id"], "section": m["section"]}


def hit(doc_id, section, text="", status="current"):
    return {"text": text, "metadata": {"doc_id": doc_id, "section": section, "status": status}}


# --- 1. the gold set is sound -------------------------------------------------

def test_gold_set_has_at_least_eight_questions_with_unique_ids():
    assert len(GOLD) >= 8
    assert len({q["id"] for q in GOLD}) == len(GOLD)


def test_gold_set_covers_every_scenario():
    assert {q["type"] for q in GOLD} == SCENARIOS


def test_gold_set_covers_every_current_document_and_never_expects_a_superseded_one(chunks):
    status = {c["metadata"]["doc_id"]: c["metadata"]["status"] for c in chunks}
    expected_docs = {s["doc_id"] for q in GOLD for s in q["expected_sources"]}
    assert expected_docs == {doc for doc, st in status.items() if st == "current"}


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q["id"])
def test_expected_answer_is_in_the_expected_source(q, chunks):
    """Checks ingest and chunking: the section exists, and the answer was not cut apart."""
    in_source = [c["text"].lower() for c in chunks if source(c) in q["expected_sources"]]
    assert in_source, "expected section does not exist"
    keywords = [k.lower() for k in q["expected_keywords"]]
    assert keywords and all(any(k in text for text in in_source) for k in keywords)
    if q["type"] != "multi_chunk":  # the whole answer must sit inside one chunk
        assert any(all(k in text for k in keywords) for text in in_source)


def test_the_multi_chunk_question_really_needs_two_chunks(chunks):
    q = next(q for q in GOLD if q["type"] == "multi_chunk")
    in_source = [c["text"].lower() for c in chunks if source(c) in q["expected_sources"]]
    assert not any(all(k.lower() in text for k in q["expected_keywords"]) for text in in_source)


# --- 2. the scoring functions ---------------------------------------------------

def test_answer_rank_is_where_the_expected_source_supplies_the_keyword():
    q = {"expected_sources": [{"doc_id": "d", "section": "S"}], "expected_keywords": ["Four"]}
    hits = [hit("x", "S", "four times"),      # has the keyword, but is the wrong document
            hit("d", "S", "meets quarterly"),  # right section, but not the chunk with the answer
            hit("d", "S", "four times")]
    assert answer_rank(hits, q) == 3
    assert answer_rank(hits[:2], q) is None


def test_answer_rank_waits_until_every_keyword_is_covered():
    q = {"expected_sources": [{"doc_id": "d", "section": "S"}], "expected_keywords": ["one day", "website"]}
    hits = [hit("d", "S", "within one day"), hit("x", "T", "other"), hit("d", "S", "use the website")]
    assert answer_rank(hits, q) == 3
    assert answer_rank(hits[:1], q) is None


def test_answer_needs_every_keyword_ignoring_case():
    q = {"expected_sources": [{}], "expected_keywords": ["Four", "per year"]}
    assert answer_is_correct("It meets four times per year.", q)
    assert not answer_is_correct("It meets four times.", q)


def test_a_question_with_no_source_must_be_declined():
    q = {"expected_sources": [], "expected_keywords": []}
    assert answer_is_correct(NOT_COVERED, q)
    assert not answer_is_correct("Employees get 20 days.", q)


def test_evaluate_retrieval_reports_recall_ranks_and_stale_rate():
    gold = [{"id": "q1", "question": "one", "expected_sources": [{"doc_id": "d", "section": "S"}], "expected_keywords": ["four"]},
            {"id": "q2", "question": "two", "expected_sources": [{"doc_id": "d", "section": "T"}], "expected_keywords": ["five"]},
            {"id": "q3", "question": "three", "expected_sources": [], "expected_keywords": []}]

    def search(collection, question, k):  # always returns the same two chunks, one of them stale
        return [hit("old", "S", "four", status="superseded"), hit("d", "S", "four")]

    report = evaluate_retrieval(None, search, gold, k=2)
    assert report["ranks"] == {"q1": 2, "q2": None}  # q3 has no source, so it is not scored
    assert report["recall"] == 0.5
    assert report["stale_rate"] == 0.5


def test_evaluate_answers_reports_accuracy():
    gold = [{"id": "q1", "question": "one", "expected_sources": [{}], "expected_keywords": ["four"]},
            {"id": "q2", "question": "two", "expected_sources": [], "expected_keywords": []}]
    report = evaluate_answers(None, gold, ask=lambda question, collection: {"answer": "Four times."})
    assert report["correct"] == {"q1": True, "q2": False}
    assert report["accuracy"] == 0.5


# --- 3. the pipeline against the gold set ---------------------------------------

@pytest.fixture(scope="module")
def retrieval(collection):
    return {name: evaluate_retrieval(collection, search, GOLD) for name, search in RETRIEVERS.items()}


@pytest.mark.parametrize("name", list(RECALL_FLOOR))
def test_recall_does_not_drop_below_the_measured_floor(name, retrieval):
    assert retrieval[name]["recall"] >= RECALL_FLOOR[name]


def test_the_final_pipeline_never_retrieves_a_superseded_chunk(retrieval):
    assert BEST == "current only"
    assert retrieval[BEST]["stale_rate"] == 0
    assert retrieval["hybrid+rerank"]["stale_rate"] > 0  # the same pipeline, before the source data was fixed


def gold_cases(questions, known_failure):
    """A question with a known-failure note in the gold file runs as a strict xfail,
    so the note has to be removed as soon as a later step fixes it."""
    return [pytest.param(q, id=q["id"], marks=[pytest.mark.xfail(reason=q[known_failure], strict=True)]
                         if known_failure in q else [])
            for q in questions]


@pytest.mark.parametrize("q", gold_cases(ANSWERABLE, "known_miss"))
def test_best_retriever_finds_the_chunk_with_the_answer(q, retrieval):
    assert retrieval[BEST]["ranks"][q["id"]] is not None


@pytest.fixture(scope="module")
def answers(collection):
    return evaluate_answers(collection, GOLD)


@pytest.mark.live
@pytest.mark.parametrize("q", gold_cases(GOLD, "known_wrong_answer"))
def test_live_answer_contains_the_expected_information(q, answers):
    assert answers["correct"][q["id"]], answers["answers"][q["id"]]
