"""Stage 5: reranking. A cross-encoder re-scores the retrieved candidates.

Vector search embeds the question and each chunk separately and compares the
two vectors. A cross-encoder reads the question and one chunk together and
outputs a single relevance score. That is more accurate but far slower, so it
is only run on the few candidates the first-stage searches return.
"""
from functools import lru_cache

from rag.config import RERANK_MODEL
from rag.trace import documents, span


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def passage(hit):
    """What the reranker reads for a chunk: its document and section, then its text (same as the embedder)."""
    return f"{hit['metadata']['title']} | {hit['metadata']['section']}\n{hit['text']}"


def rerank(question, hits, top_n=4):
    """Score every hit against the question and return the top_n, best first, each with a `rerank_score`."""
    if not hits:
        return []
    with span("rerank", "RERANKER", question) as s:
        scores = model().predict([(question, passage(h)) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        top = [{**hit, "rerank_score": float(score)} for hit, score in ranked[:top_n]]
        s.set_attributes({
            "reranker.query": question, "reranker.model_name": RERANK_MODEL, "reranker.top_k": top_n,
            **documents(hits, score=lambda h: h.get("rrf_score"), prefix="reranker.input_documents"),
            **documents(top, score=lambda h: h["rerank_score"], prefix="reranker.output_documents"),
        })
        return top
