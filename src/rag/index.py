"""Chunk every document, embed the chunks, and store them in Chroma.

Run:  python -m rag.index            build the index and print a summary
      python -m rag.index --show N   also print N sample chunks with metadata
"""
import json
import sys
from collections import Counter

from rag.chunking import MAX_WORDS, build_chunks
from rag.config import CHROMA_DIR, EMBEDDING_MODEL, ROOT
from rag.embed import MAX_TOKENS
from rag.store import build_index

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"


def main():
    chunks = build_chunks()
    CHUNKS_PATH.write_text("".join(
        json.dumps({"chunk_id": c["chunk_id"], "metadata": c["metadata"], "text": c["text"]},
                   ensure_ascii=False) + "\n" for c in chunks))
    collection = build_index(chunks)

    per_doc = Counter(c["metadata"]["doc_id"] for c in chunks)
    print(f"{'document':<44}{'family':<10}{'version':<12}{'status':<12}chunks")
    for doc_id, n in per_doc.items():
        m = next(c["metadata"] for c in chunks if c["metadata"]["doc_id"] == doc_id)
        print(f"{doc_id:<44}{m['family']:<10}{m['version']:<12}{m['status']:<12}{n}")

    words = [c["metadata"]["word_count"] for c in chunks]
    tokens = [c["metadata"]["token_count"] for c in chunks]
    print(f"\n{len(chunks)} chunks | words: min {min(words)}, avg {sum(words) // len(words)}, "
          f"max {max(words)} (limit {MAX_WORDS}) | tokens: max {max(tokens)} (model limit {MAX_TOKENS})")
    print(f"Embedded with {EMBEDDING_MODEL}; {collection.count()} vectors stored in "
          f"{CHROMA_DIR.relative_to(ROOT)}/")
    print(f"Chunks written to {CHUNKS_PATH.relative_to(ROOT)}")

    if "--show" in sys.argv:
        n = int(sys.argv[sys.argv.index("--show") + 1])
        for c in chunks[:: max(1, len(chunks) // n)][:n]:
            print(f"\n--- {c['chunk_id']}")
            print(json.dumps(c["metadata"], indent=2))
            print(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
