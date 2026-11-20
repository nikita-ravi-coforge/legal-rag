"""Keyword search: a grep-style match of the question's words against chunk text.

Vector search matches meaning, so it can miss an exact word, code or section
name. This search does the opposite: it only looks for the literal terms.

Scoring: each question term found in a chunk adds log(N / n) to that chunk's
score, where N is the number of chunks and n is how many contain the term. A
term found in one chunk counts a lot; a term found in every chunk counts zero.
"""
import math
import re

# Question words carry no topic, but a few appear literally in the policies
# ("What are the facts?"), so they are removed before matching.
STOPWORDS = frozenset(
    "a about an and are as at be by can do does for from how i in is it must my of on or "
    "our say says should that the their they this to we what when where which who why with you".split())

# A term is a run of letters and digits; dots and hyphens inside it are kept,
# so "v15.0" and "1-800-815-2481" stay whole.
TERM = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def terms(question):
    """Question -> its distinct lower-case terms, in order, without stopwords."""
    words = TERM.findall(question.lower())
    return [w for w in dict.fromkeys(words) if w not in STOPWORDS]


def keyword_search(chunks, question, k=5):
    """Top-k chunks by keyword score. A chunk that matches no term is not returned."""
    texts = [f"{c['metadata']['section']}\n{c['text']}".lower() for c in chunks]
    scores = [0.0] * len(chunks)
    for term in terms(question):
        matches = [i for i, text in enumerate(texts) if term in text]
        for i in matches:
            scores[i] += math.log(len(chunks) / len(matches))
    ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True)
    return [{**chunks[i], "keyword_score": round(scores[i], 3)} for i in ranked[:k]]
