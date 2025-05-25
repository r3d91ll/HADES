"""Tests specifically for when the model engine is unavailable.

This module contains tests for scenarios where the Haystack model engine is unavailable.
These tests are isolated from other tests to avoid interference from fixtures and mocks.
"""

import unittest
import pytest
from unittest.mock import patch, MagicMock

from src.chunking.text_chunkers.chonky_chunker import chunk_text


@pytest.fixture
def isolated_environment():
    """Create an isolated environment for testing fallbacks.
    
    This fixture creates a clean environment to test the fallback behavior
    without interference from other test fixtures or mocks.
    """
    # Import the module inside the fixture to avoid affecting global state
    from src.chunking.text_chunkers import chonky_chunker
    
    # Store original values
    original_engine_available = chonky_chunker._ENGINE_AVAILABLE
    original_chonky_available = chonky_chunker._CHONKY_AVAILABLE
    
    # Set up test environment
    chonky_chunker._ENGINE_AVAILABLE = False
    chonky_chunker._CHONKY_AVAILABLE = True
    
    # Mock the model engine to return None
    with patch("src.chunking.text_chunkers.chonky_chunker.get_model_engine") as mock_get_engine:
        mock_get_engine.return_value = None
        
        yield
    
    # Restore original values
    chonky_chunker._ENGINE_AVAILABLE = original_engine_available
    chonky_chunker._CHONKY_AVAILABLE = original_chonky_available

def test_fallbacks_when_engine_unavailable(isolated_environment):
    """Test fallbacks when model engine is not available.
    
    This test verifies that when the engine is unavailable, the chunker falls back
    to basic paragraph splitting.
    """
    # Create a test document with clear paragraph breaks
    test_content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    test_doc_id = "test-doc-id"
    
    # Use the isolated environment to ensure no mocks interfere
    result = chunk_text(
        content=test_content,
        doc_id=test_doc_id,
        output_format="dict"
    )

    # Verify basic structure
    assert isinstance(result, dict)
    assert "chunks" in result
    assert len(result["chunks"]) == 3  # Should have 3 paragraphs
    
    # Verify the content of the chunks matches our expected paragraphs
    expected_contents = ["First paragraph.", "Second paragraph.", "Third paragraph."] 
    actual_contents = [chunk["content"] for chunk in result["chunks"]]
    
    # Each expected paragraph should be in the chunks
    for expected in expected_contents:
        assert expected in actual_contents, f"Expected to find '{expected}' in chunks"
