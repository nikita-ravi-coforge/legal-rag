"""Embedding model wrapper: one place that knows which model is used and its limits."""
import os
from functools import lru_cache

from rag.config import EMBEDDING_MODEL

# bge-small-en-v1.5 reads at most 512 tokens; anything longer is silently cut off.
MAX_TOKENS = 512

# The bge models are trained with this instruction on the *query* side only.
# Documents are embedded as-is. Set QUERY_PREFIX="" to turn it off.
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def count_tokens(text):
    """Token count as the model sees it, including its two special tokens, with no truncation."""
    return len(tokenizer()(text, truncation=False)["input_ids"])


def embed_documents(texts):
    return model().encode(list(texts), normalize_embeddings=True).tolist()


def embed_query(question):
    return model().encode([QUERY_PREFIX + question], normalize_embeddings=True)[0].tolist()
