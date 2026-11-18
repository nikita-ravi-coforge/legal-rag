"""Render every chunk as a card in one self-contained HTML page.

Run:  python -m rag.view_chunks     writes results/chunks.html and opens it
"""
import sys
import webbrowser
from html import escape
from itertools import groupby

from rag.chunking import MAX_WORDS, build_chunks
from rag.config import RESULTS_DIR
from rag.embed import MAX_TOKENS

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--muted:#667085;--line:#dfe3ea;
--accent:#2456c9;--overlap:#fff3c4;--overlap-line:#e0b400;--old:#b42318;--old-bg:#fdecea;--bar:#cfd8ea}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--muted:#98a2b3;
--line:#2c3340;--accent:#8fb0ff;--overlap:#4a3f10;--overlap-line:#c9a300;--old:#ff8a80;--old-bg:#3a1d1a;--bar:#333c4d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:0}
h3{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:26px 0 8px}
.sub,.meta{color:var(--muted);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
nav a{color:var(--accent);text-decoration:none;border:1px solid var(--line);background:var(--card);
border-radius:999px;padding:4px 12px;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--muted);margin-bottom:8px}
.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px;
background:var(--overlap);border-left:3px solid var(--overlap-line)}
details.doc{margin-top:28px;border-top:2px solid var(--line);padding-top:16px}
summary{cursor:pointer}
summary h2{display:inline}
.chunk{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.chunk.superseded{border-left:4px solid var(--old)}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
.id{font:12.5px ui-monospace,Menlo,monospace;color:var(--accent);overflow-wrap:anywhere}
.badge{font-size:11.5px;border-radius:999px;padding:1px 9px;border:1px solid var(--line);color:var(--muted)}
.badge.superseded{color:var(--old);background:var(--old-bg);border-color:var(--old)}
.size{margin-left:auto;font-size:12.5px;color:var(--muted);white-space:nowrap}
.bar{height:4px;background:var(--bar);border-radius:2px;margin-bottom:10px}
.bar i{display:block;height:4px;background:var(--accent);border-radius:2px}
.block{margin:0 0 8px;padding:2px 8px;white-space:pre-wrap;overflow-wrap:anywhere}
.block.overlap{background:var(--overlap);border-left:3px solid var(--overlap-line);border-radius:3px}
.block:last-child{margin-bottom:0}
"""


def overlap_block(prev, chunk):
    """The block `chunk` repeats from the end of `prev`, or None."""
    if prev is None:
        return None
    same_section = all(prev["metadata"][k] == chunk["metadata"][k] for k in ("doc_id", "section"))
    first = chunk["text"].split("\n\n")[0]
    return first if same_section and first == prev["text"].split("\n\n")[-1] else None


def render_chunk(chunk, prev):
    m = chunk["metadata"]
    repeated = overlap_block(prev, chunk)
    blocks = []
    for i, block in enumerate(chunk["text"].split("\n\n")):
        cls = "block overlap" if i == 0 and repeated else "block"
        title = ' title="Repeated from the previous chunk (overlap)"' if cls.endswith("overlap") else ""
        blocks.append(f'<p class="{cls}"{title}>{escape(block)}</p>')
    old = m["status"] == "superseded"
    pct = min(100, round(100 * m["word_count"] / MAX_WORDS))
    return (
        f'<div class="chunk{" superseded" if old else ""}"><div class="head">'
        f'<span class="id">{escape(chunk["chunk_id"])}</span>'
        f'<span class="badge">{escape(m["family"])}</span>'
        f'<span class="badge">v{escape(str(m["version"]))}</span>'
        + (f'<span class="badge superseded">superseded</span>' if old else "")
        + f'<span class="size">{m["word_count"]} / {MAX_WORDS} words &middot; '
        f'{m["token_count"]} / {MAX_TOKENS} tokens</span></div>'
        f'<div class="bar"><i style="width:{pct}%"></i></div>{"".join(blocks)}</div>'
    )


def render_html(chunks):
    nav, body = [], []
    for doc_id, doc_chunks in groupby(chunks, key=lambda c: c["metadata"]["doc_id"]):
        doc_chunks = list(doc_chunks)
        m = doc_chunks[0]["metadata"]
        nav.append(f'<a href="#{escape(doc_id)}">{escape(m["title"])} v{escape(str(m["version"]))} '
                   f'({len(doc_chunks)})</a>')
        parts, prev = [], None
        for section, section_chunks in groupby(doc_chunks, key=lambda c: c["metadata"]["section"]):
            parts.append(f"<h3>{escape(section)}</h3>")
            for c in section_chunks:
                parts.append(render_chunk(c, prev))
                prev = c
        body.append(
            f'<details class="doc" id="{escape(doc_id)}" open><summary><h2>{escape(m["title"])}</h2> '
            f'<span class="meta">{escape(doc_id)} &middot; {escape(m["family"])} &middot; '
            f'v{escape(str(m["version"]))} &middot; {escape(m["status"])} &middot; '
            f'{len(doc_chunks)} chunks</span></summary>{"".join(parts)}</details>')
    words = [c["metadata"]["word_count"] for c in chunks]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Chunk Viewer</title><style>{CSS}</style></head><body><main>"
        f'<h1>Chunk Viewer</h1><div class="sub">{len(chunks)} chunks &middot; '
        f'{min(words)} to {max(words)} words, average {sum(words) // len(words)} &middot; '
        f'limit {MAX_WORDS} words per chunk</div>'
        f'<nav>{"".join(nav)}</nav>'
        '<div class="legend"><span><span class="swatch"></span>Overlap: block repeated from the '
        'previous chunk</span><span>Bar: how full the chunk is against the word limit</span>'
        '<span>Red edge: chunk from a superseded document</span></div>'
        f'{"".join(body)}</main></body></html>'
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "chunks.html"
    out.write_text(render_html(build_chunks()))
    print(f"Wrote {out}")
    if "--no-open" not in sys.argv:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
