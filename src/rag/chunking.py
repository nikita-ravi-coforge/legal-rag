"""Stage 2: section-aware chunking.

Strategy
  1. Split each document on its section headings. A chunk never crosses a
     section, so every chunk has exactly one (document, section) to cite.
  2. Inside a section the natural unit is a "block": one bullet, one numbered
     item or one paragraph. In these policies each bullet is a complete rule.
  3. Pack blocks into a chunk until the next one would pass MAX_WORDS.
  4. Start the next chunk by repeating the previous chunk's last block.

Sizes, and why
  MAX_WORDS = 180   About 250 tokens of policy text, half of the embedding
                    model's 512-token limit, so nothing is truncated. Small
                    enough that a chunk is about one or two topics; large
                    enough to keep a rule together with its neighbours.
  Overlap = 1 block Bullets are self-contained, so the only context that can
                    be lost at a boundary is the neighbouring bullet. The
                    overlap is skipped when that block is longer than
                    OVERLAP_MAX_WORDS, where it would mostly duplicate text.
  A single block longer than MAX_WORDS is split on sentence boundaries with a
  one-sentence overlap.
"""
import json
import re

from rag.config import MANIFEST_PATH, PROCESSED_DIR
from rag.embed import MAX_TOKENS, count_tokens

MAX_WORDS = 180
OVERLAP_MAX_WORDS = MAX_WORDS // 2

LINEAGE_FIELDS = ["doc_id", "title", "source_file", "family", "version", "year", "status"]


def n_words(text):
    return len(text.split())


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])", text.strip()) if s]


def load_documents():
    """Manifest entries for ingested documents, each with its markdown attached."""
    docs = []
    for entry in json.loads(MANIFEST_PATH.read_text()):
        if entry.get("ingested"):
            docs.append({**entry, "markdown": (PROCESSED_DIR / f"{entry['doc_id']}.md").read_text()})
    return docs


def parse_sections(markdown):
    """Markdown -> [(section name, [block, ...]), ...] in document order."""
    body = markdown.split("\n---\n", 1)[1]  # drop the metadata header
    sections = []
    for part in re.split(r"(?m)^## ", body)[1:]:
        name, _, rest = part.partition("\n")
        blocks = [b.strip() for b in rest.split("\n\n") if b.strip()]
        sections.append((name.strip(), blocks))
    return sections


def pack_blocks(blocks, max_words=MAX_WORDS, overlap_max=OVERLAP_MAX_WORDS):
    """Greedily pack blocks into chunks of at most max_words, with a one-block overlap."""
    chunks, cur = [], []
    for block in blocks:
        if cur and sum(map(n_words, cur)) + n_words(block) > max_words:
            chunks.append(cur)
            last = cur[-1]
            fits = n_words(last) <= overlap_max and n_words(last) + n_words(block) <= max_words
            cur = [last] if fits else []
        cur.append(block)
    if cur:
        chunks.append(cur)
    return chunks


def chunk_section(blocks):
    """Blocks of one section -> list of chunk texts."""
    texts, run = [], []

    def flush():
        texts.extend("\n\n".join(group) for group in pack_blocks(run))
        run.clear()

    for block in blocks:
        if n_words(block) <= MAX_WORDS:
            run.append(block)
        else:  # one paragraph too long for a chunk: split it on sentences
            flush()
            texts.extend(" ".join(group) for group in pack_blocks(split_sentences(block)))
    flush()
    return texts


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_chunks():
    chunks = []
    for doc in load_documents():
        for section, blocks in parse_sections(doc["markdown"]):
            for i, text in enumerate(chunk_section(blocks)):
                # The model embeds the text with its document and section in front, so a
                # short chunk still says what it is about. The stored text stays clean.
                embed_text = f"{doc['title']} | {section}\n{text}"
                tokens = count_tokens(embed_text)
                chunk_id = f"{doc['doc_id']}:{slug(section)}:{i:02d}"
                if tokens > MAX_TOKENS:
                    raise ValueError(f"{chunk_id} is {tokens} tokens; the model truncates at {MAX_TOKENS}")
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "embed_text": embed_text,
                    "metadata": {
                        **{k: doc[k] for k in LINEAGE_FIELDS},
                        "section": section,
                        "chunk_index": i,
                        "word_count": n_words(text),
                        "token_count": tokens,
                    },
                })
    return chunks
