"""Shared fixtures. Chunks and the index are built once per test session."""
import pytest


@pytest.fixture(scope="session")
def chunks():
    from rag.chunking import build_chunks
    return build_chunks()


@pytest.fixture(scope="session")
def collection(chunks, tmp_path_factory):
    from rag.store import build_index
    return build_index(chunks, path=tmp_path_factory.mktemp("chroma"))
