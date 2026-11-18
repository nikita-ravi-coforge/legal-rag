"""Step 3a: section-aware chunking."""
import re

import pytest

from rag.chunking import MAX_WORDS, load_documents, pack_blocks, parse_sections, split_sentences
from rag.embed import MAX_TOKENS, count_tokens

REQUIRED = {"doc_id", "title", "source_file", "section", "family", "version", "year",
            "status", "chunk_index", "word_count", "token_count"}


def words(text):
    return len(text.split())


# ---- pure packing logic, on synthetic blocks --------------------------------

def block(letter, n):
    return " ".join([letter] * n)


def test_pack_keeps_small_section_as_one_chunk():
    assert pack_blocks([block("a", 50), block("b", 50)], max_words=180) == [[block("a", 50), block("b", 50)]]


def test_pack_starts_new_chunk_at_limit_and_repeats_last_block_as_overlap():
    a, b, c, d = block("a", 80), block("b", 40), block("c", 80), block("d", 30)
    assert pack_blocks([a, b, c, d], max_words=180, overlap_max=60) == [[a, b], [b, c, d]]


def test_pack_skips_overlap_when_the_boundary_block_is_too_long():
    a, b = block("a", 100), block("b", 100)
    assert pack_blocks([a, b], max_words=180, overlap_max=60) == [[a], [b]]


def test_pack_never_exceeds_the_limit():
    blocks = [block(ch, n) for ch, n in zip("abcdefgh", [70, 20, 90, 55, 10, 120, 60, 30])]
    for chunk in pack_blocks(blocks, max_words=180, overlap_max=60):
        assert sum(words(b) for b in chunk) <= 180


def test_split_sentences():
    assert split_sentences("First one. Second one! Is this third? Yes.") == \
        ["First one.", "Second one!", "Is this third?", "Yes."]


# ---- the real corpus --------------------------------------------------------

def test_corpus_produces_a_sensible_number_of_chunks(chunks):
    assert 40 <= len(chunks) <= 120


def test_no_chunk_exceeds_the_word_limit(chunks):
    over = [(c["chunk_id"], words(c["text"])) for c in chunks if words(c["text"]) > MAX_WORDS]
    assert over == []


def test_no_chunk_exceeds_the_embedding_models_token_limit(chunks):
    """Anything past the limit is silently truncated by the model, so fail loudly here."""
    for c in chunks:
        actual = count_tokens(c["embed_text"])
        assert actual == c["metadata"]["token_count"]
        assert actual <= MAX_TOKENS, c["chunk_id"]


def test_every_chunk_has_complete_metadata(chunks):
    for c in chunks:
        assert REQUIRED <= set(c["metadata"]), c["chunk_id"]
        assert all(v not in ("", None) for v in c["metadata"].values()), c["chunk_id"]


def test_chunk_ids_are_unique_and_readable(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert "supplier_code_2025:transactions:00" in ids


def test_embed_text_carries_document_and_section_context(chunks):
    c = next(c for c in chunks if c["chunk_id"] == "supplier_code_2025:transactions:00")
    assert c["embed_text"].startswith("Lilly Supplier Code of Business Conduct | Transactions\n")
    assert c["embed_text"].endswith(c["text"])


def test_chunks_never_cross_a_section_boundary(chunks):
    section_text = {}
    for doc in load_documents():
        for section, blocks in parse_sections(doc["markdown"]):
            section_text[(doc["doc_id"], section)] = " ".join(blocks)
    for c in chunks:
        body = section_text[(c["metadata"]["doc_id"], c["metadata"]["section"])]
        for sentence in split_sentences(c["text"].replace("\n\n", " ")):
            assert sentence in body, (c["chunk_id"], sentence)


def test_no_source_text_is_lost(chunks):
    for doc in load_documents():
        for section, blocks in parse_sections(doc["markdown"]):
            mine = " ".join(c["text"].replace("\n\n", " ") for c in chunks
                            if (c["metadata"]["doc_id"], c["metadata"]["section"]) == (doc["doc_id"], section))
            for b in blocks:
                for sentence in split_sentences(b):
                    assert sentence in mine, (doc["doc_id"], section, sentence)


def test_consecutive_chunks_in_a_bulleted_section_overlap_by_one_bullet(chunks):
    section = [c for c in chunks if c["chunk_id"].startswith("supplier_code_2025:transactions:")]
    assert len(section) >= 2
    for prev, nxt in zip(section, section[1:]):
        assert nxt["text"].split("\n\n")[0] == prev["text"].split("\n\n")[-1]


def test_an_oversized_paragraph_is_split_on_sentences(chunks):
    section = [c for c in chunks
               if c["chunk_id"].startswith("ethics_compliance_committee_charter_2025:administrative-matters:")]
    assert len(section) >= 2
    for c in section:
        assert re.search(r"[.;]$", c["text"].strip())


@pytest.mark.parametrize("doc_id,family,version,status", [
    ("supplier_code_2021", "supplier", "13.0", "superseded"),
    ("supplier_code_2025", "supplier", "15.0", "current"),
    ("red_book_2026", "employee", "2026", "current"),
    ("ethics_compliance_committee_charter_2025", "board", "2025-03-01", "current"),
])
def test_chunks_inherit_lineage_from_the_manifest(chunks, doc_id, family, version, status):
    mine = [c["metadata"] for c in chunks if c["metadata"]["doc_id"] == doc_id]
    assert mine
    assert {(m["family"], m["version"], m["status"]) for m in mine} == {(family, version, status)}
