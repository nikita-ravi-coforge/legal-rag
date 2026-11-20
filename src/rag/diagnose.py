"""Part 6: reproduce and diagnose the planted data quality issue.

Run:  python -m rag.diagnose

The same question goes through the same pipeline twice. The only difference
is the source data: first every indexed document, then current documents only.
The written diagnosis is in docs/diagnosis.md.
"""
import json

from rag.ask import answer
from rag.config import MANIFEST_PATH
from rag.generate import ANSWER_MODEL

QUESTION = "What is the current version number of the Supplier Code of Business Conduct?"


def diagnose(question=QUESTION, collection=None, client=None):
    """Run the question with and without the superseded document in the search."""
    return {"before": answer(question, collection=collection, client=client, current_only=False),
            "after": answer(question, collection=collection, client=client, current_only=True)}


def show(title, result):
    print(f"\n{title}")
    for rank, h in enumerate(result["hits"], start=1):
        m = h["metadata"]
        print(f"  [{rank}] {h['chunk_id']:<42} version {m['version']:<6} {m['status']}")
    print(f"\n  Answer ({ANSWER_MODEL}):\n  {result['answer']}")


def main():
    print("\nLineage of the Supplier Code in data/manifest.json:")
    for doc in json.loads(MANIFEST_PATH.read_text()):
        if doc.get("family") == "supplier":
            print(f"  {doc['doc_id']:<20} version {doc['version']:<6} {doc['status']:<11}"
                  f" superseded by: {doc['superseded_by'] or '-'}   ({doc['source_file']})")

    report = diagnose()
    print(f"\nQuestion: {QUESTION}")
    show("BEFORE - every indexed document is searched. Chunks sent to the LLM:", report["before"])
    show("AFTER - only current documents are searched. Chunks sent to the LLM:", report["after"])
    print()


if __name__ == "__main__":
    main()
