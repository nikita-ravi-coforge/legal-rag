"""Stage 4: retrieval. Vector search and keyword search, merged into one ranking, then reranked.

The two searches score on different scales (cosine distance vs keyword score),
so the scores cannot be added. Reciprocal Rank Fusion uses only the rank:
a chunk at rank r in a list earns 1 / (RRF_K + r), summed over the lists it is
in. A chunk both searches found beats one that only a single search found.
"""
from rag.keyword import keyword_search
from rag.rerank import rerank
from rag.store import all_chunks, vector_search
from rag.trace import documents, span

CANDIDATES = 10  # how many hits each search contributes
RRF_K = 60       # standard constant; it stops rank 1 from dominating everything else


def rrf_merge(rankings):
    """{"vector": hits, "keyword": hits} -> one list, best first.

    Each hit gains `rrf_score` and `found_by` (the searches that returned it).
    """
    merged = {}
    for name, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            entry = merged.setdefault(hit["chunk_id"], {"rrf_score": 0.0, "found_by": []})
            entry.update(hit)
            entry["rrf_score"] += 1 / (RRF_K + rank)
            entry["found_by"].append(name)
    return sorted(merged.values(), key=lambda h: h["rrf_score"], reverse=True)


def hybrid_search(collection, question, k=5):
    """Top-k chunks from vector search and keyword search combined."""
    with span("hybrid_search", "RETRIEVER", question) as fused_span:
        with span("vector_search", "RETRIEVER", question) as s:
            vector = vector_search(collection, question, k=CANDIDATES)
            s.set_attributes(documents(vector, score=lambda h: 1 - h["distance"]))  # cosine similarity
        with span("keyword_search", "RETRIEVER", question) as s:
            keyword = keyword_search(all_chunks(collection), question, k=CANDIDATES)
            s.set_attributes(documents(keyword, score=lambda h: h["keyword_score"]))
        fused = rrf_merge({"vector": vector, "keyword": keyword})[:k]
        fused_span.set_attributes(documents(fused, score=lambda h: h["rrf_score"]))
        return fused


def retrieve(collection, question, k=4):
    """The full retrieval pipeline: every hybrid candidate, reranked, top k kept."""
    candidates = hybrid_search(collection, question, k=2 * CANDIDATES)
    return rerank(question, candidates, top_n=k)
