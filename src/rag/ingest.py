"""Stage 1: raw PDFs -> clean markdown + a lineage manifest.

Run:  python -m rag.ingest

For every PDF in data/raw this records a sha256, drops byte-identical
duplicates, extracts the text, strips page furniture, marks section headings,
and writes data/processed/<doc_id>.md. data/manifest.json is the lineage
record: which file each document came from, its version, and its status.
"""
import hashlib
import json
import re

from pypdf import PdfReader

from rag.config import MANIFEST_PATH, OVERRIDES_DIR, PROCESSED_DIR, RAW_DIR

SUPPLIER_HEADINGS = {h: h for h in
                     ["Interactions", "Workplace", "Information", "Transactions", "Reporting"]}
SUPPLIER_DROP = [
    r"^Lilly Supplier Code of Business Conduct$",
    r"^page \d of \d$",
]
# Decorative pull-quotes that pypdf drops into the middle of the body text.
SUPPLIER_CUT = [
    r"Lilly unites caring with\s+discovery\s+to make life better\s+for people\s+around the world\.",
    r"Together, we act with transparency,\s+honesty,\s+and integrity, knowing that\s+every interaction "
    r"is an\s+opportunity to\s+improve patient care and build trust\s+with those we serve\.",
]

RED_BOOK_RESPONSIBILITIES = [
    "Respecting People",
    "Protecting People, the Environment, and Our Assets",
    "Conducting Research and Development",
    "Assuring Quality",
    "Preventing Corruption",
    "Communicating Honestly",
    "Marketing Responsibly",
    "Managing and Protecting Information",
    "Respecting Personal Information and Privacy",
    "Using Artificial Intelligence Responsibly",
    "Maintaining Financial Integrity",
    "Speaking Up",
]

DOCS = {
    "Final_Ethics_and_Compliance_Committee_Charter_-_3.1.2025.pdf": {
        "doc_id": "ethics_compliance_committee_charter_2025",
        "title": "Ethics and Compliance Committee Charter",
        "family": "board",
        "version": "2025-03-01",
        "year": 2025,
        "status": "current",
        "headings": {h: h for h in ["Purpose", "Composition and Term",
                                    "Administrative Matters", "Duties and Responsibilities"]},
        "drop": [r"^Ethics and Compliance Committee Charter$"],
        # The date/page stamp sits in the middle of a sentence at the page break.
        "subs": [(r"\n*March 1, 2025 \d\n*", "\n")],
    },
    "LSCoBC_2025_EN.pdf": {
        "doc_id": "supplier_code_2025",
        "title": "Lilly Supplier Code of Business Conduct",
        "family": "supplier",
        "version": "15.0",
        "year": 2025,
        "status": "current",
        "headings": SUPPLIER_HEADINGS,
        "drop": SUPPLIER_DROP,
        "cut": SUPPLIER_CUT,
        "first_section": "Introduction",
    },
    "Lilly_SCoBC_2021_EN.pdf": {
        "doc_id": "supplier_code_2021",
        "title": "Lilly Supplier Code of Business Conduct",
        "family": "supplier",
        "version": "13.0",
        "year": 2021,
        # Superseded by supplier_code_2025. This is the planted data quality issue.
        "status": "superseded",
        "superseded_by": "supplier_code_2025",
        "headings": SUPPLIER_HEADINGS,
        "drop": SUPPLIER_DROP,
        "cut": SUPPLIER_CUT,
        "first_section": "Introduction",
    },
    "Red_Book_English.pdf": {
        "doc_id": "red_book_2026",
        "title": "Red Book: Code of Business Conduct",
        "family": "employee",
        "version": "2026",
        "year": 2026,
        "status": "current",
        "skip_pages": [1, 11],  # front and back cover
        "page_overrides": {8: "red_book_p08.md"},
        "headings": {
            **{h: f"Our Responsibilities: {h}" for h in RED_BOOK_RESPONSIBILITIES},
            "Investigations Overview": "Speaking Up: Investigations Overview",
        },
        "drop": [
            r"^\d{2,4}$",
            r"^RED BOOK \| ELI LILLY AND COMPANY$",
            r"^OUR$", r"^RESPONSIBILITIES$", r"^CONTINUED$",
        ],
        "subs": [
            (r"(?m)^WHO WE ARE ", "## Who We Are\n"),
            (r"(?m)^OUR PURPOSE ", "## Our Purpose\n"),
            (r"(?m)^OUR VALUES ", "## Our Values\n"),
            (r"(?m)^RED BOOK CODE OF\nBUSINESS CONDUCT$", "## About the Red Book"),
            (r"(?m)^QUOTE FROM\nOUR CEO$", "## Quote from Our CEO"),
            (r"(?m)^DAVE RICKS$", "\nDave Ricks, Chief Executive Officer"),
            (r"(?m)^SPEAKING UP How to Report a Concern$", "## Speaking Up: How to Report a Concern"),
            (r"(?m)^QUOTE FROM OUR CHIEF\nETHICS AND COMPLIANCE\nOFFICER$",
             "## Quote from Our Chief Ethics and Compliance Officer"),
            (r"(?m)^STEPHANIE MCCALLUM$", "\nStephanie McCallum, Chief Ethics and Compliance Officer"),
        ],
    },
}

VERSION_LINE = re.compile(r"(?m)^(?:V|Version )\d+\.\d+ ©\d{4}, Eli Lilly and Company\. ALL RIGHTS RESERVED\.$")
STANDALONE = {"In the US:", "Outside the US:"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(line):
    return re.sub(r"\s+", " ", line).strip()


def extract(path, cfg):
    """PDF -> one cleaned string; newlines are still the PDF's line wraps."""
    reader = PdfReader(str(path))
    pages = []
    for num, page in enumerate(reader.pages, start=1):
        if num in cfg.get("skip_pages", []):
            continue
        if num in cfg.get("page_overrides", {}):
            pages.append((OVERRIDES_DIR / cfg["page_overrides"][num]).read_text().strip())
            continue
        lines = [norm(l) for l in page.extract_text().splitlines()]
        lines = [l for l in lines if not any(re.match(p, l) for p in cfg.get("drop", []))]
        pages.append("\n".join(lines))
    text = "\n".join(pages)
    for pattern, repl in cfg.get("subs", []):
        text = re.sub(pattern, repl, text)
    for pattern in cfg.get("cut", []):
        text = re.sub(pattern, "", text)
    return text


def to_markdown(text, cfg):
    """Join wrapped lines into blocks; mark headings, bullets and numbered items."""
    tail = []
    m = VERSION_LINE.search(text)
    if m:  # footer line carries the version number; keep it, but at the end
        tail = ["## Document Version", m.group(0)]
        text = VERSION_LINE.sub("", text)

    blocks, cur = [], ""

    def flush():
        nonlocal cur
        if cur:
            blocks.append(cur)
        cur = ""

    if cfg.get("first_section"):
        blocks.append(f"## {cfg['first_section']}")
    for line in text.split("\n"):
        if not line:
            flush()
        elif line.startswith("## "):
            flush(); blocks.append(line)
        elif line in cfg["headings"]:
            flush(); blocks.append(f"## {cfg['headings'][line]}")
        elif line in STANDALONE:
            flush(); blocks.append(line)
        elif line.startswith("» "):
            flush(); cur = "- " + line[2:]
        elif line.startswith("• "):
            flush(); cur = "  - " + line[2:]
        elif re.match(r"\d+\. ", line):
            flush(); cur = line
        elif cur.endswith("-"):
            cur += line  # word hyphenated across a line wrap
        else:
            cur = f"{cur} {line}" if cur else line
    flush()
    return "\n\n".join(blocks + tail) + "\n"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest, seen = [], {}
    # Configured documents first, so an unknown copy is the one flagged as the duplicate.
    for path in sorted(RAW_DIR.glob("*.pdf"), key=lambda p: (p.name not in DOCS, p.name)):
        digest = sha256(path)
        if digest in seen:
            manifest.append({"source_file": path.name, "sha256": digest, "status": "duplicate",
                             "duplicate_of": seen[digest], "ingested": False})
            print(f"SKIP  {path.name}: byte-identical duplicate of {seen[digest]}")
            continue
        seen[digest] = path.name
        cfg = DOCS.get(path.name)
        if cfg is None:
            manifest.append({"source_file": path.name, "sha256": digest, "status": "unknown",
                             "ingested": False})
            print(f"SKIP  {path.name}: no entry in DOCS")
            continue

        body = to_markdown(extract(path, cfg), cfg)
        entry = {
            "doc_id": cfg["doc_id"], "title": cfg["title"], "source_file": path.name,
            "sha256": digest, "family": cfg["family"], "version": cfg["version"],
            "year": cfg["year"], "status": cfg["status"],
            "superseded_by": cfg.get("superseded_by"),
            "manual_overrides": [f"page {p}: {f}" for p, f in cfg.get("page_overrides", {}).items()],
            "word_count": len(body.split()), "ingested": True,
        }
        header = "\n".join(f"{k}: {entry[k]}" for k in
                           ["doc_id", "title", "source_file", "family", "version", "year", "status"])
        (PROCESSED_DIR / f"{cfg['doc_id']}.md").write_text(f"---\n{header}\n---\n\n# {cfg['title']}\n\n{body}")
        manifest.append(entry)
        print(f"OK    {path.name} -> {cfg['doc_id']}.md  ({entry['word_count']} words, "
              f"{body.count('## ')} sections, v{cfg['version']}, {cfg['status']})")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {MANIFEST_PATH.relative_to(MANIFEST_PATH.parents[1])}")


if __name__ == "__main__":
    main()
