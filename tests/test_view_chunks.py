"""Chunk viewer: one self-contained HTML page showing every chunk."""
from rag.view_chunks import overlap_block, render_html


def test_every_chunk_appears_in_the_page(chunks):
    html = render_html(chunks)
    assert html.startswith("<!doctype html>")
    for c in chunks:
        assert c["chunk_id"] in html


def test_overlap_block_is_detected(chunks):
    by_id = {c["chunk_id"]: c for c in chunks}
    prev = by_id["supplier_code_2025:transactions:00"]
    nxt = by_id["supplier_code_2025:transactions:01"]
    assert overlap_block(prev, nxt) == prev["text"].split("\n\n")[-1]
    assert overlap_block(None, prev) is None


def test_overlap_is_not_reported_across_sections(chunks):
    by_id = {c["chunk_id"]: c for c in chunks}
    assert overlap_block(by_id["supplier_code_2025:interactions:00"],
                         by_id["supplier_code_2025:workplace:00"]) is None


def test_page_marks_overlap_and_superseded_chunks(chunks):
    html = render_html(chunks)
    assert 'class="block overlap"' in html
    assert html.count('class="badge superseded"') >= 14  # every 2021 chunk


def test_chunk_text_is_html_escaped():
    chunk = {"chunk_id": "d:s:00", "text": "a <b> & c",
             "metadata": {"doc_id": "d", "title": "T", "section": "S", "family": "board",
                          "version": "1", "status": "current", "word_count": 4, "token_count": 9}}
    html = render_html([chunk])
    assert "a &lt;b&gt; &amp; c" in html and "a <b> & c" not in html
