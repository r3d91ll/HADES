"""Common fixtures and setup for unit tests.

This module contains fixtures and setup code that are shared across
multiple test modules, providing consistent testing infrastructure.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any

# Import shared fixtures from common_fixtures
from tests.unit.common_fixtures import (
    sample_text_document,
    sample_code_document,
    create_sample_document,
    create_expected_chunks,
    SAMPLE_TEXT_CONTENT,
    SAMPLE_CODE_CONTENT
)


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    # Create the directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_documents() -> Dict[str, Dict[str, Any]]:
    """Return a set of sample documents of different types."""
    return {
        "text": create_sample_document(
            content=SAMPLE_TEXT_CONTENT,
            doc_id="text-doc-001",
            path="/path/to/document.md",
            doc_type="text"
        ),
        "python": create_sample_document(
            content=SAMPLE_CODE_CONTENT,
            doc_id="code-doc-001",
            path="/path/to/document.py",
            doc_type="python"
        ),
        "empty": create_sample_document(
            content="",
            doc_id="empty-doc",
            path="/path/to/empty.txt",
            doc_type="text"
        ),
    }


@pytest.fixture
def create_test_file(tmp_path):
    """Fixture to create test files for document processing tests."""
    
    def _create_file(file_name: str, content: str) -> Path:
        """Create a file with the given name and content.
        
        Args:
            file_name: Name of the file to create
            content: Content to write to the file
            
        Returns:
            Path to the created file
        """
        file_path = tmp_path / file_name
        file_path.write_text(content)
        return file_path
    
    return _create_file
