"""Stage 5: generation. Retrieved chunks + question -> prompt -> Claude -> answer."""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

from rag.config import ROOT
from rag.trace import span

load_dotenv(ROOT / ".env")

# Pinned so answers do not change when a newer model is released.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-5-5")
MAX_ANSWER_TOKENS = 500

NOT_COVERED = "This is not covered in the provided policy documents."

SYSTEM = f"""You are an internal assistant that answers questions about company ethics and compliance policies.

Answer using only the numbered policy excerpts in the user's message. Do not use outside knowledge and do not guess.
If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: {NOT_COVERED}
Keep the answer short and factual: one to four sentences, using the policy's own terms.
Cite the excerpt each statement comes from by its number in square brackets, like [1]."""


def build_prompt(question, hits):
    """Returns (system, user). Each chunk is numbered and labelled with its document and section."""
    excerpts = "\n\n".join(
        f"[{i}] {h['metadata']['title']} | {h['metadata']['section']}\n{h['text']}"
        for i, h in enumerate(hits, start=1))
    return SYSTEM, f"Policy excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def cited_sources(text, hits):
    """The chunks an answer cites, as source records (document, version, section), in the order cited.

    The LLM cites excerpts by number, and excerpt [n] is hits[n-1]. An answer
    with no citation is attributed to every chunk it was given. A declined
    answer has no sources.
    """
    if NOT_COVERED.lower() in text.lower():
        return []
    cited = [int(n) for n in dict.fromkeys(re.findall(r"\[(\d+)\]", text))]
    numbers = [n for n in cited if 1 <= n <= len(hits)] or range(1, len(hits) + 1)
    return [{"number": n,
             "document": hits[n - 1]["metadata"]["title"],
             "version": hits[n - 1]["metadata"]["version"],
             "section": hits[n - 1]["metadata"]["section"],
             "source_file": hits[n - 1]["metadata"]["source_file"],
             "chunk_id": hits[n - 1]["chunk_id"]} for n in numbers]


@lru_cache(maxsize=1)
def default_client():
    import anthropic
    return anthropic.Anthropic(max_retries=3)  # reads ANTHROPIC_API_KEY; retries rate limits


def generate_answer(question, hits, client=None, model=None):
    system, user = build_prompt(question, hits)
    model = model or ANSWER_MODEL
    with span("generate_answer", "LLM", user) as s:
        response = (client or default_client()).messages.create(
            model=model,
            max_tokens=MAX_ANSWER_TOKENS,
            # No temperature: the API rejects it for this model ("Extra inputs are not
            # permitted"), so output is not guaranteed identical between runs. The
            # strict system prompt keeps answers stable; tests match key phrases.
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        s.set_attributes({"llm.model_name": model, "output.value": text})
        return text
