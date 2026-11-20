"""Stage 5: generation. Retrieved chunks + question -> prompt -> Claude -> answer."""
import os
from functools import lru_cache

from dotenv import load_dotenv

from rag.config import ROOT

load_dotenv(ROOT / ".env")

# Pinned so answers do not change when a newer model is released.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-5-5")
MAX_ANSWER_TOKENS = 500

NOT_COVERED = "This is not covered in the provided policy documents."

SYSTEM = f"""You are an internal assistant that answers questions about company ethics and compliance policies.

Answer using only the numbered policy excerpts in the user's message. Do not use outside knowledge and do not guess.
If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: {NOT_COVERED}
Keep the answer short and factual: one to four sentences, using the policy's own terms."""


def build_prompt(question, hits):
    """Returns (system, user). Each chunk is numbered and labelled with its document and section."""
    excerpts = "\n\n".join(
        f"[{i}] {h['metadata']['title']} | {h['metadata']['section']}\n{h['text']}"
        for i, h in enumerate(hits, start=1))
    return SYSTEM, f"Policy excerpts:\n\n{excerpts}\n\nQuestion: {question}"


@lru_cache(maxsize=1)
def default_client():
    import anthropic
    return anthropic.Anthropic(max_retries=3)  # reads ANTHROPIC_API_KEY; retries rate limits


def generate_answer(question, hits, client=None, model=None):
    system, user = build_prompt(question, hits)
    response = (client or default_client()).messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        # No temperature: the API rejects it for this model ("Extra inputs are not
        # permitted"), so output is not guaranteed identical between runs. The
        # strict system prompt keeps answers stable; tests match key phrases.
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
