# Diagnosis: the planted data quality issue

Reproduce everything below with `python -m rag.diagnose`.

## The planted issue

The corpus holds two versions of the same policy, the Lilly Supplier Code of Business Conduct:

| Document | Version | Status in `data/manifest.json` |
|---|---|---|
| `supplier_code_2025` | 15.0 (2025) | current |
| `supplier_code_2021` | 13.0 (2021) | superseded by `supplier_code_2025` |

The 2021 version is an outdated duplicate. Most of its text is identical to the 2025 version, but some details conflict. For example, the 2025 version requires "assessment by third party risk management" before purchasing goods or services, and the 2021 version does not.

Both versions were chunked, embedded and indexed.

## The question that surfaced it

> What is the current version number of the Supplier Code of Business Conduct?

The correct answer is 15.0.

## The flawed answer

Chunks sent to the LLM, with every indexed document searched:

| Rank | Chunk | Version | Status |
|---|---|---|---|
| 1 | `supplier_code_2021:document-version:00` | 13.0 | superseded |
| 2 | `supplier_code_2025:document-version:00` | 15.0 | current |
| 3 | `supplier_code_2021:reporting:00` | 13.0 | superseded |
| 4 | `supplier_code_2021:workplace:01` | 13.0 | superseded |

Answer:

> The excerpts show two versions: Version 13.0 (©2021) and V15.0 (©2025). V15.0 is the more recent, but the excerpts do not explicitly identify either one as the current version.

The answer is flawed in two ways. It does not commit to which version is current, and it presents the superseded version 13.0 as a valid source.

A second question shows the same fault with a conflict in substance. Asked "What is required before a supplier purchases goods or services?", the system answered that "the excerpts give two versions of this requirement" and listed the 2021 wording first, without the third party risk management assessment.

## Two-question debugging

### 1. Was it retrieval?

No. Retrieval did what it is designed to do.

- The question asks about the Supplier Code's version. The top two chunks are the "Document Version" sections of the Supplier Code. They are the most relevant chunks in the index.
- The correct chunk, `supplier_code_2025:document-version:00`, was retrieved at rank 2. Nothing was missed.
- The 2021 chunk at rank 1 is not a ranking mistake. Its text answers the question as well as the 2025 chunk does; the reranker scored the two almost the same. Nothing in a chunk's text says it is out of date.

Generation was not the cause either. The prompt contained two version statements with the same document title and nothing marking one as superseded. The model reported exactly that and declined to guess, which is the behaviour the system prompt asks for.

### 2. Was it the source data?

Yes. The index contained a document that should not have been searchable.

- `data/manifest.json` records `supplier_code_2021` as `superseded`, with `superseded_by: supplier_code_2025`. The lineage was known at ingest time.
- Every chunk carries that status in its metadata, but retrieval did not use it, so superseded chunks competed with current ones on equal terms.
- Across the 24-question gold set, 26% of all chunks retrieved by the hybrid + rerank pipeline came from the superseded document.

## Confirming the root cause

The test: change only the source data and leave retrieval and generation untouched. If the answer becomes correct, the source data was the cause.

The fix restricts the search to chunks whose document status is `current`. It is a metadata filter (`CURRENT` in `src/rag/retrieve.py`). The embedding model, the keyword search, the fusion, the reranker, the prompt and the LLM are all unchanged.

Chunks sent to the LLM for the same question, after the fix:

| Rank | Chunk | Version | Status |
|---|---|---|---|
| 1 | `supplier_code_2025:document-version:00` | 15.0 | current |
| 2 | `supplier_code_2025:information:01` | 15.0 | current |
| 3 | `supplier_code_2025:workplace:00` | 15.0 | current |
| 4 | `supplier_code_2025:reporting:00` | 15.0 | current |

Answer:

> The current version of the Lilly Supplier Code of Business Conduct is V15.0, ©2025, Eli Lilly and Company [1].

The purchasing question now answers with the 2025 requirement only: authorization from procurement, assessment by third party risk management, and supporting contractual documentation.

Gold set results (`python -m rag.evaluate --answers`):

| Pipeline | Recall@4 | Stale rate |
|---|---|---|
| Hybrid + rerank, every document searched | 100% | 26% |
| Hybrid + rerank, current documents only | 100% | 0% |

Recall did not change, because retrieval was never the problem. The stale rate fell to zero, because the bad source is no longer searched.

## Conclusion

The flawed answer traces to the source data: an outdated duplicate of a policy was indexed alongside the current version. Retrieval returned the most relevant chunks, and generation reported faithfully what it was given. Removing the superseded document from the search corrects the answer with no change to retrieval or generation logic.

The flaw can still be reproduced on demand with `python -m rag.ask --include-superseded "..."`.

## Limits of this diagnosis

- The fix depends on the manifest being right. A superseded document wrongly marked `current` would still be searched.
- The filter hides the old version completely, so a question about what the 2021 version said can no longer be answered.
- The gold set's keyword scoring originally marked the flawed answer as correct, because it contained "15.0". The version question now also forbids "13.0" in the answer.
