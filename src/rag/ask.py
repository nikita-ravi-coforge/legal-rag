"""The RAG pipeline end to end.

Run:  python -m rag.ask "How often must the ethics committee meet?"

Requires the index (python -m rag.index) and ANTHROPIC_API_KEY in .env.
"""
import sys

from rag.generate import ANSWER_MODEL, generate_answer
from rag.retrieve import retrieve
from rag.store import get_collection
from rag.trace import setup, span

TOP_K = 4


def answer(question, k=TOP_K, collection=None, client=None):
    """Retrieve the top-k chunks (hybrid search + rerank) and have Claude answer from them."""
    collection = collection or get_collection()
    with span("rag.answer", "CHAIN", question) as s:
        hits = retrieve(collection, question, k=k)
        text = generate_answer(question, hits, client=client)
        s.set_attribute("output.value", text)
        return {"question": question, "answer": text, "hits": hits}


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m rag.ask "your question"')
    setup()  # tracing to Phoenix, only if PHOENIX_TRACING=1
    result = answer(" ".join(sys.argv[1:]))

    print(f"\nQuestion: {result['question']}\n")
    print(f"Retrieved top {len(result['hits'])} chunks (hybrid search, then reranked):")
    for rank, h in enumerate(result["hits"], start=1):
        print(f"  {rank}. rerank score {h['rerank_score']:.3f}  {h['chunk_id']}  (found by {' + '.join(h['found_by'])})")
    print(f"\nAnswer ({ANSWER_MODEL}):\n{result['answer']}\n")


if __name__ == "__main__":
    main()
