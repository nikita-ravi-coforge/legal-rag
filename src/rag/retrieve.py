"""Stage 4: hybrid retrieval. Vector search and keyword search, merged into one ranking.

The two searches score on different scales (cosine distance vs keyword score),
so the scores cannot be added. Reciprocal Rank Fusion uses only the rank:
a chunk at rank r in a list earns 1 / (RRF_K + r), summed over the lists it is
in. A chunk both searches found beats one that only a single search found.
"""
from rag.keyword import keyword_search
from rag.store import all_chunks, vector_search

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
    return rrf_merge({
        "vector": vector_search(collection, question, k=CANDIDATES),
        "keyword": keyword_search(all_chunks(collection), question, k=CANDIDATES),
    })[:k]
