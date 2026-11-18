"""Paths and settings shared by every pipeline stage. Settings are env-overridable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OVERRIDES_DIR = ROOT / "data" / "overrides"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_DIR = ROOT / "results"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
