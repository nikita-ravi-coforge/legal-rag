"""The RAG pipeline end to end.

Run:  python -m rag.ask "How often must the ethics committee meet?"

Requires the index (python -m rag.index) and ANTHROPIC_API_KEY in .env.
"""
import sys

from rag.generate import ANSWER_MODEL, generate_answer
from rag.store import get_collection, vector_search

TOP_K = 4


def answer(question, k=TOP_K, collection=None, client=None):
    """Embed the question, retrieve the top-k chunks, and have Claude answer from them."""
    collection = collection or get_collection()
    hits = vector_search(collection, question, k=k)
    return {"question": question, "answer": generate_answer(question, hits, client=client), "hits": hits}


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m rag.ask "your question"')
    result = answer(" ".join(sys.argv[1:]))

    print(f"\nQuestion: {result['question']}\n")
    print(f"Retrieved top {len(result['hits'])} chunks (vector search):")
    for rank, h in enumerate(result["hits"], start=1):
        print(f"  {rank}. distance {h['distance']:.3f}  {h['chunk_id']}")
    print(f"\nAnswer ({ANSWER_MODEL}):\n{result['answer']}\n")


if __name__ == "__main__":
    main()
