"""Stage 3: the vector store. Chunks go in with their embeddings and metadata."""
import chromadb

from rag.config import CHROMA_DIR
from rag.embed import embed_documents, embed_query

COLLECTION = "policy_chunks"


def build_index(chunks, path=CHROMA_DIR):
    """Embed every chunk and (re)build the collection from scratch."""
    client = chromadb.PersistentClient(path=str(path))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed_documents(c["embed_text"] for c in chunks),
    )
    return collection


def get_collection(path=CHROMA_DIR):
    return chromadb.PersistentClient(path=str(path)).get_collection(COLLECTION)


def all_chunks(collection, where=None):
    """Stored chunks with their text and metadata (no embeddings), for keyword search. `where` filters on metadata."""
    got = collection.get(where=where, include=["documents", "metadatas"])
    return [{"chunk_id": i, "text": t, "metadata": m}
            for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]


def vector_search(collection, question, k=5, where=None):
    """Top-k chunks by cosine distance. `where` is a Chroma metadata filter."""
    res = collection.query(query_embeddings=[embed_query(question)], n_results=k, where=where)
    return [{"chunk_id": i, "text": t, "metadata": m, "distance": d}
            for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                                  res["metadatas"][0], res["distances"][0])]
