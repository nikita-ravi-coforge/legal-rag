"""Step 5: hybrid retrieval. Keyword search runs next to vector search; RRF merges the two."""
from rag.keyword import keyword_search, terms
from rag.retrieve import hybrid_search, rrf_merge
from rag.store import all_chunks, vector_search


def chunk(chunk_id, text, section="General"):
    return {"chunk_id": chunk_id, "text": text, "metadata": {"section": section}}


CHUNKS = [
    chunk("a", "Suppliers must report theft to Lilly security."),
    chunk("b", "Suppliers must obtain AAALAC accreditation for animal research."),
    chunk("c", "Suppliers must keep accurate records.", section="Transactions"),
    chunk("d", "The committee meets four times per year."),
]


def ids(hits):
    return [h["chunk_id"] for h in hits]


# --- keyword search ---------------------------------------------------------

def test_terms_are_lowercased_deduplicated_and_stripped_of_stopwords():
    assert terms("What is the AAALAC rule, and is AAALAC required?") == ["aaalac", "rule", "required"]


def test_terms_keep_codes_and_numbers_whole():
    assert terms("V15.0 hotline 1-800-815-2481") == ["v15.0", "hotline", "1-800-815-2481"]


def test_keyword_search_finds_the_chunk_with_the_exact_term():
    hits = keyword_search(CHUNKS, "What is AAALAC?", k=3)
    assert ids(hits) == ["b"]
    assert hits[0]["text"] == CHUNKS[1]["text"] and hits[0]["keyword_score"] > 0


def test_a_rare_term_outweighs_a_common_one():
    # "suppliers" is in three chunks, "theft" in one: the theft chunk must come first.
    assert ids(keyword_search(CHUNKS, "suppliers theft", k=3))[0] == "a"


def test_keyword_search_also_matches_the_section_name():
    assert ids(keyword_search(CHUNKS, "transactions", k=3)) == ["c"]


def test_keyword_search_returns_nothing_when_no_term_matches():
    assert keyword_search(CHUNKS, "vacation days", k=3) == []


# --- reciprocal rank fusion -------------------------------------------------

def test_rrf_puts_a_chunk_found_by_both_searches_first():
    merged = rrf_merge({"vector": [chunk("a", ""), chunk("b", "")],
                        "keyword": [chunk("c", ""), chunk("b", "")]})
    assert ids(merged)[0] == "b"
    assert merged[0]["found_by"] == ["vector", "keyword"]
    assert abs(merged[0]["rrf_score"] - 2 / 62) < 1e-9  # rank 2 in both lists: 1/(60+2) twice


def test_rrf_lists_every_chunk_once():
    merged = rrf_merge({"vector": [chunk("a", ""), chunk("b", "")],
                        "keyword": [chunk("b", ""), chunk("c", "")]})
    assert sorted(ids(merged)) == ["a", "b", "c"]
    assert [h["found_by"] for h in merged if h["chunk_id"] == "a"] == [["vector"]]


# --- on the real index ------------------------------------------------------

def test_all_chunks_returns_every_stored_chunk(chunks, collection):
    stored = all_chunks(collection)
    assert len(stored) == len(chunks)
    assert set(stored[0]) == {"chunk_id", "text", "metadata"}


def test_hybrid_search_returns_k_hits_best_first(collection):
    hits = hybrid_search(collection, "How often must the ethics committee meet?", k=4)
    assert len(hits) == 4
    scores = [h["rrf_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_finds_an_exact_phrase_that_vector_search_misses(collection):
    """The demo query: 'cooperative learning programs' appears only in the Supplier Code's Workplace section."""
    question = "Which policy covers cooperative learning programs?"

    def has_the_phrase(hits):
        return any("cooperative learning programs" in h["text"] for h in hits)

    assert not has_the_phrase(vector_search(collection, question, k=4))
    hybrid = hybrid_search(collection, question, k=4)
    assert "cooperative learning programs" in hybrid[0]["text"]  # hybrid ranks it first
    assert "keyword" in hybrid[0]["found_by"]


def test_hybrid_ranks_an_exact_section_name_first(collection):
    assert hybrid_search(collection, "Composition and Term", k=4)[0]["metadata"]["section"] == "Composition and Term"
