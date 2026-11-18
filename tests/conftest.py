"""Shared fixtures. Chunks and the index are built once per test session."""
import os

import pytest
from dotenv import load_dotenv

from rag.config import ROOT

load_dotenv(ROOT / ".env")


def has_api_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key) and "paste-your-key" not in key


def pytest_collection_modifyitems(config, items):
    """Tests marked `live` call the real LLM; skip them when there is no key."""
    if has_api_key():
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def chunks():
    from rag.chunking import build_chunks
    return build_chunks()


@pytest.fixture(scope="session")
def collection(chunks, tmp_path_factory):
    from rag.store import build_index
    return build_index(chunks, path=tmp_path_factory.mktemp("chroma"))
