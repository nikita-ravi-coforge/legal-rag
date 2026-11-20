"""Evaluation harness: score the pipeline against the gold dataset.

Run:  python -m rag.evaluate             retrieval only (no API calls)
      python -m rag.evaluate --answers   also generate and score the answers

data/gold_set.json holds fixed questions, each with the (document, section)
that contains the answer and the keywords a correct answer must include.

Metrics
  recall@k      share of questions whose answer is in the top k chunks: the chunks
                from the expected source contain every expected keyword
  stale rate    share of retrieved chunks that come from a superseded document
  accuracy      share of answers that contain every expected keyword; a question
                with no source in the corpus must get the "not covered" reply
"""
import json
import sys

from rag.ask import answer
from rag.config import RESULTS_DIR, ROOT
from rag.generate import NOT_COVERED
from rag.retrieve import hybrid_search, retrieve
from rag.store import get_collection, vector_search
from rag.trace import setup

GOLD_PATH = ROOT / "data" / "gold_set.json"
RESULTS_PATH = RESULTS_DIR / "eval_results.json"
TOP_K = 4

# Every retrieval variant built so far. Each takes (collection, question, k).
RETRIEVERS = {"vector": vector_search, "hybrid": hybrid_search, "hybrid+rerank": retrieve}
BEST = "hybrid+rerank"  # the variant the per-question gold tests hold to account


def load_gold():
    return json.loads(GOLD_PATH.read_text())


def answer_rank(hits, q):
    """How far down the hits you must read to have the whole answer, or None if it is not there.

    Only chunks from an expected (document, section) count, and only for the
    expected keywords they contain. So the right section with the wrong chunk,
    or the right words from a superseded document, is not a hit.
    """
    missing = {keyword.lower() for keyword in q["expected_keywords"]}
    for rank, hit in enumerate(hits, start=1):
        m = hit["metadata"]
        if {"doc_id": m["doc_id"], "section": m["section"]} in q["expected_sources"]:
            missing -= {keyword for keyword in missing if keyword in hit["text"].lower()}
            if not missing:
                return rank
    return None


def evaluate_retrieval(collection, search, gold, k=TOP_K):
    """Run every gold question through one search function."""
    ranks, retrieved, stale = {}, 0, 0
    for q in gold:
        hits = search(collection, q["question"], k=k)
        retrieved += len(hits)
        stale += sum(h["metadata"]["status"] == "superseded" for h in hits)
        if q["expected_sources"]:  # out-of-scope questions have no source to find
            ranks[q["id"]] = answer_rank(hits, q)
    found = sum(rank is not None for rank in ranks.values())
    return {"recall": found / len(ranks), "stale_rate": stale / retrieved, "ranks": ranks}


def answer_is_correct(text, q):
    if not q["expected_sources"]:
        return NOT_COVERED.lower() in text.lower()
    return all(keyword.lower() in text.lower() for keyword in q["expected_keywords"])


def evaluate_answers(collection, gold, ask=answer):
    """Generate an answer for every gold question (one LLM call each) and score it."""
    answers = {q["id"]: ask(q["question"], collection=collection)["answer"] for q in gold}
    correct = {q["id"]: answer_is_correct(answers[q["id"]], q) for q in gold}
    return {"accuracy": sum(correct.values()) / len(gold), "correct": correct, "answers": answers}


def main():
    setup()  # tracing to Phoenix, only if PHOENIX_TRACING=1
    collection, gold = get_collection(), load_gold()
    results = {"k": TOP_K, "questions": len(gold),
               "retrieval": {name: evaluate_retrieval(collection, search, gold)
                             for name, search in RETRIEVERS.items()}}
    if "--answers" in sys.argv:
        results["answers"] = evaluate_answers(collection, gold)

    columns = list(RETRIEVERS) + (["answer"] if "answers" in results else [])
    print(f"\n{'question':<30}{'type':<17}" + "".join(f"{c:>15}" for c in columns))
    for q in gold:
        cells = [results["retrieval"][name]["ranks"].get(q["id"], "n/a") or "miss" for name in RETRIEVERS]
        if "answers" in results:
            cells.append("ok" if results["answers"]["correct"][q["id"]] else "WRONG")
        print(f"{q['id']:<30}{q['type']:<17}" + "".join(f"{c!s:>15}" for c in cells))

    print(f"\nRank at which the answer is fully retrieved, top {TOP_K} (miss = not retrieved, n/a = no source exists)\n")
    for name, report in results["retrieval"].items():
        print(f"{name:<14} recall@{TOP_K} {report['recall']:.0%}   stale rate {report['stale_rate']:.0%}")
    if "answers" in results:
        print(f"{'answers':<14} accuracy {results['answers']['accuracy']:.0%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
