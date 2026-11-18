"""Step 2: the smallest possible embed -> store -> retrieve loop.

Run:  python -m rag.minimal_loop

Two known texts go into an in-memory Chroma collection. A question about
one of them must bring that one back first. Nothing else in the pipeline is
built until this works.
"""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL

TEXTS = {
    "gifts": "We do not accept gifts, entertainment, or favors from suppliers "
             "that could affect or appear to affect our business judgment.",
    "meetings": "The ethics and compliance committee shall meet at least four times per year.",
}


@lru_cache(maxsize=1)
def model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts):
    """Text -> unit-length vectors, so cosine distance is meaningful."""
    return model().encode(list(texts), normalize_embeddings=True).tolist()


def build_collection():
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("minimal_loop")
    except Exception:
        pass
    collection = client.create_collection("minimal_loop", metadata={"hnsw:space": "cosine"})
    collection.add(ids=list(TEXTS), documents=list(TEXTS.values()),
                   embeddings=embed(TEXTS.values()))
    return collection


def query(collection, question, k=2):
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return [{"id": i, "text": t, "distance": d}
            for i, t, d in zip(res["ids"][0], res["documents"][0], res["distances"][0])]


def main():
    print(f"Embedding model: {EMBEDDING_MODEL}")
    vec = embed(["hello"])[0]
    print(f"One embedding = {len(vec)} numbers, e.g. {[round(x, 3) for x in vec[:5]]} ...\n")

    collection = build_collection()
    print(f"Stored {collection.count()} texts in Chroma:")
    for key, text in TEXTS.items():
        print(f"  [{key}] {text}")

    ok = True
    for question, expected in [("Can I accept a present from a vendor?", "gifts"),
                               ("How often does the committee meet?", "meetings")]:
        hits = query(collection, question)
        print(f"\nQuery: {question}")
        for rank, h in enumerate(hits, 1):
            print(f"  {rank}. [{h['id']}] cosine distance {h['distance']:.3f}")
        passed = hits[0]["id"] == expected
        ok = ok and passed
        print(f"  -> expected [{expected}] first: {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
