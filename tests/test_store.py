"""Step 3b: embed every chunk and store it in Chroma, then prove it can be searched."""
from rag.embed import embed_documents, embed_query
from rag.store import vector_search


def test_embeddings_are_1024_dim_unit_vectors():
    for vec in embed_documents(["a policy sentence"]) + [embed_query("a question?")]:
        assert len(vec) == 1024
        assert abs(sum(x * x for x in vec) - 1.0) < 1e-3


def test_every_chunk_is_stored(chunks, collection):
    assert collection.count() == len(chunks)


def test_text_and_metadata_round_trip(chunks, collection):
    original = next(c for c in chunks if c["chunk_id"] == "supplier_code_2021:transactions:00")
    got = collection.get(ids=[original["chunk_id"]])
    assert got["documents"][0] == original["text"]
    assert got["metadatas"][0] == original["metadata"]
    assert got["metadatas"][0]["status"] == "superseded"


def test_semantic_query_finds_the_right_section(collection):
    hits = vector_search(collection, "How many times a year must the ethics committee meet?", k=3)
    assert len(hits) == 3
    top = hits[0]
    assert top["metadata"]["doc_id"] == "ethics_compliance_committee_charter_2025"
    assert top["metadata"]["section"] == "Administrative Matters"
    assert hits[0]["distance"] <= hits[1]["distance"] <= hits[2]["distance"]


def test_where_filter_restricts_results_to_one_family(collection):
    hits = vector_search(collection, "reporting a concern", k=5, where={"family": "employee"})
    assert hits and all(h["metadata"]["family"] == "employee" for h in hits)
