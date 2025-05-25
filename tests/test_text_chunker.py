"""
Pytest suite to verify Chonky-based text chunking.

Replaces the original ad-hoc script with deterministic tests that:

1.  Create a temporary Markdown file.
2.  Pre-process it via ``MarkdownPreProcessor``.
3.  Ensure ``chunk_text`` returns non-empty chunks in *both* Python and
    JSON output formats.
4.  Validate that ``PreprocessorManager._process_text_file`` populates
    entities / relationships.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.ingest.chunking import chunk_text
from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor
from src.ingest.pre_processor.manager import PreprocessorManager


@pytest.fixture()
def sample_markdown(tmp_path: Path) -> Path:
    """Create a minimal Markdown file for testing and return its path."""
    md = tmp_path / "sample.md"
    md.write_text(
        """# Title\n\nThis is the *first* paragraph.\n\n## Section\n\nAnother paragraph.""",
        encoding="utf-8",
    )
    return md


def _preprocess(md_path: Path) -> Dict[str, Any]:
    pre = MarkdownPreProcessor()
    doc = pre.process_file(str(md_path))
    assert doc and doc.get("content"), "Markdown pre-processing failed"
    return doc


def test_chunk_text_python(sample_markdown: Path) -> None:
    """Ensure the default (Python) output format yields non-empty chunks."""
    doc = _preprocess(sample_markdown)
    chunks: List[Dict[str, Any]] = chunk_text(doc)
    assert chunks and isinstance(chunks, list), "chunk_text returned no chunks"
    # basic shape check
    assert {"id", "content", "type"}.issubset(chunks[0].keys())


def test_chunk_text_json(sample_markdown: Path) -> None:
    """Verify that JSON output is valid and round-trips back to list[dict]."""
    doc = _preprocess(sample_markdown)
    json_str = chunk_text(doc, output_format="json")
    parsed = json.loads(json_str)
    assert parsed and isinstance(parsed, list), "JSON output not parsable"


def test_preprocessor_manager_entities(sample_markdown: Path) -> None:
    """Validate entity / relationship extraction integration."""
    doc = _preprocess(sample_markdown)
    manager = PreprocessorManager()
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    manager._process_text_file(doc, entities, relationships)
    assert entities, "No entities created"
    assert relationships, "No relationships created"
