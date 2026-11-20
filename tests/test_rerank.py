"""Step 8: reranking. A cross-encoder re-scores the hybrid candidates; the best few go to the LLM."""
from rag.rerank import rerank
from rag.retrieve import retrieve


def hit(chunk_id, text, section="General"):
    return {"chunk_id": chunk_id, "text": text, "metadata": {"title": "Policy", "section": section}}


HITS = [
    hit("gifts", "We do not accept gifts or entertainment from suppliers."),
    hit("meetings", "The committee shall meet at least four times per year."),
    hit("privacy", "We protect personal information against loss and misuse."),
]


def test_rerank_puts_the_chunk_that_answers_the_question_first():
    ranked = rerank("How often does the committee meet?", HITS, top_n=3)
    assert [h["chunk_id"] for h in ranked][0] == "meetings"


def test_rerank_adds_a_score_and_sorts_by_it():
    scores = [h["rerank_score"] for h in rerank("How often does the committee meet?", HITS, top_n=3)]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) for s in scores)


def test_rerank_keeps_only_top_n_and_leaves_the_input_alone():
    ranked = rerank("Can we accept gifts?", HITS, top_n=1)
    assert [h["chunk_id"] for h in ranked] == ["gifts"]
    assert "rerank_score" not in HITS[0]


def test_rerank_of_nothing_is_nothing():
    assert rerank("anything", [], top_n=4) == []


def test_retrieve_returns_k_reranked_hits_from_the_real_index(collection):
    hits = retrieve(collection, "How many times a year must the ethics committee meet?", k=4)
    assert len(hits) == 4
    assert hits[0]["metadata"]["section"] == "Administrative Matters"
    assert all("rerank_score" in h and "found_by" in h for h in hits)


def test_retrieve_fixes_the_question_hybrid_alone_missed(collection):
    """Hybrid pushed this paraphrase's answer out of the top 4; the reranker brings it back."""
    hits = retrieve(collection, "Is it acceptable to show up to the job drunk or high?", k=4)
    assert any("impaired" in h["text"] and h["metadata"]["status"] == "current" for h in hits)
