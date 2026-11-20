"""Vector-only search next to hybrid search, for one question.

Run:  python -m rag.compare "Which policy covers cooperative learning programs?"
"""
import sys

from rag.retrieve import hybrid_search
from rag.store import get_collection, vector_search

TOP_K = 4


def show(title, hits, note):
    print(f"\n{title}")
    for rank, h in enumerate(hits, start=1):
        print(f"  {rank}. {h['chunk_id']:<70} {note(h)}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m rag.compare "your question"')
    question = " ".join(sys.argv[1:])
    collection = get_collection()
    vector = vector_search(collection, question, k=TOP_K)
    hybrid = hybrid_search(collection, question, k=TOP_K)

    print(f"\nQuestion: {question}")
    show(f"Vector only, top {TOP_K}:", vector, lambda h: f"distance {h['distance']:.3f}")
    show(f"Hybrid (vector + keyword, RRF), top {TOP_K}:", hybrid, lambda h: "found by " + " + ".join(h["found_by"]))

    vector_ids = {h["chunk_id"] for h in vector}
    added = [h["chunk_id"] for h in hybrid if h["chunk_id"] not in vector_ids]
    print(f"\nChunks hybrid found that vector-only missed: {', '.join(added) or 'none'}\n")


if __name__ == "__main__":
    main()
