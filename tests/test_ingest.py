"""Step 1: ingest. Runs the real PDFs through ingest into a temp directory."""
import json
import re

import pytest

from rag import ingest


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    out = tmp_path_factory.mktemp("ingest")
    mp = pytest.MonkeyPatch()
    mp.setattr(ingest, "PROCESSED_DIR", out / "processed")
    mp.setattr(ingest, "MANIFEST_PATH", out / "manifest.json")
    ingest.main()
    mp.undo()
    manifest = json.loads((out / "manifest.json").read_text())
    docs = {p.stem: p.read_text() for p in (out / "processed").glob("*.md")}
    return manifest, docs


def by_id(manifest, doc_id):
    return next(e for e in manifest if e.get("doc_id") == doc_id)


def sections(text):
    return re.findall(r"(?m)^## (.+)$", text)


def test_four_distinct_documents_are_ingested(ingested):
    _, docs = ingested
    assert set(docs) == {
        "ethics_compliance_committee_charter_2025",
        "supplier_code_2025",
        "supplier_code_2021",
        "red_book_2026",
    }


def test_exact_duplicate_is_skipped_and_logged(ingested):
    manifest, _ = ingested
    dupes = [e for e in manifest if e["status"] == "duplicate"]
    assert len(dupes) == 1
    assert dupes[0]["source_file"] == "Lilly_SCoBC_2021_EN-2.pdf"
    assert dupes[0]["duplicate_of"] == "Lilly_SCoBC_2021_EN.pdf"
    assert dupes[0]["ingested"] is False


def test_2021_supplier_code_is_marked_superseded(ingested):
    manifest, _ = ingested
    old = by_id(manifest, "supplier_code_2021")
    assert (old["status"], old["version"], old["superseded_by"]) == \
        ("superseded", "13.0", "supplier_code_2025")
    new = by_id(manifest, "supplier_code_2025")
    assert (new["status"], new["version"]) == ("current", "15.0")


def test_each_document_has_an_audience_family(ingested):
    manifest, _ = ingested
    families = {e["doc_id"]: e["family"] for e in manifest if e["ingested"]}
    assert families == {
        "ethics_compliance_committee_charter_2025": "board",
        "supplier_code_2025": "supplier",
        "supplier_code_2021": "supplier",
        "red_book_2026": "employee",
    }


@pytest.mark.parametrize("doc_id", ["supplier_code_2025", "supplier_code_2021"])
def test_supplier_code_sections(ingested, doc_id):
    _, docs = ingested
    assert sections(docs[doc_id]) == [
        "Introduction", "Interactions", "Workplace", "Information",
        "Transactions", "Reporting", "Document Version",
    ]


def test_charter_sections(ingested):
    _, docs = ingested
    assert sections(docs["ethics_compliance_committee_charter_2025"]) == [
        "Purpose", "Composition and Term", "Administrative Matters", "Duties and Responsibilities",
    ]


def test_red_book_has_all_twelve_responsibilities(ingested):
    _, docs = ingested
    resp = [s for s in sections(docs["red_book_2026"]) if s.startswith("Our Responsibilities: ")]
    assert len(resp) == 12
    assert "Our Responsibilities: Using Artificial Intelligence Responsibly" in resp


def test_page_furniture_is_removed(ingested):
    _, docs = ingested
    for text in docs.values():
        assert not re.search(r"page \d of \d", text)
        assert "RED BOOK | ELI LILLY AND COMPANY" not in text
        assert not re.search(r"(?m)^\d{2,4}$", text)
        assert not re.search(r"March 1, 2025 \d", text)


def test_sentence_spanning_charter_page_break_is_rejoined(ingested):
    _, docs = ingested
    assert ("recommending to the directors and corporate governance committee any changes"
            in docs["ethics_compliance_committee_charter_2025"])


def test_garbled_red_book_page_uses_the_manual_override(ingested):
    manifest, docs = ingested
    assert "EXERCISE GOOD JUDGMENTEXERCISE" not in docs["red_book_2026"]
    assert "Step 3: Consider the Impact." in docs["red_book_2026"]
    assert by_id(manifest, "red_book_2026")["manual_overrides"] == ["page 8: red_book_p08.md"]


def test_version_differences_survive_extraction(ingested):
    """The planted issue: 2025 adds requirements that 2021 does not have."""
    _, docs = ingested
    assert "assessment by third party risk management" in docs["supplier_code_2025"]
    assert "third party risk management" not in docs["supplier_code_2021"]
    assert "AAALAC" in docs["supplier_code_2025"]
    assert "AAALAC" not in docs["supplier_code_2021"]
