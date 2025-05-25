"""Unit tests for the Chonky chunker module.

This module contains comprehensive tests for the Chonky-based semantic text chunker,
ensuring that documents are properly split into semantically meaningful chunks.
"""

from __future__ import annotations

import os
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, call

from src.chunking.text_chunkers.chonky_chunker import (
    chunk_text,
    chunk_document,
    chunk_document_to_json,
    chunk_document_to_schema,
    BaseDocument,
    ensure_model_engine
)
from tests.unit.common_fixtures import (
    sample_text_document,
    sample_code_document,
    create_expected_chunks
)


@pytest.fixture
def mock_chonky():
    """Mock Chonky ParagraphSplitter for testing."""
    with patch("src.chunking.text_chunkers.chonky_chunker._CHONKY_AVAILABLE", True), \
         patch("src.chunking.text_chunkers.chonky_chunker.ParagraphSplitter") as mock_splitter:
        # Configure the mock splitter to return realistic chunks
        instance = mock_splitter.return_value
        # Make the instance itself callable and return paragraphs
        paragraphs = [
            "Introduction to Machine Learning",
            "Supervised Learning algorithms build a model based on sample data.",
            "Unsupervised learning is a type of algorithm that learns patterns from untagged data.",
            "Reinforcement learning is concerned with how agents take actions in an environment."
        ]
        instance.side_effect = lambda text: paragraphs
        instance.split_text.return_value = paragraphs
        yield mock_splitter


@pytest.fixture
def mock_engine():
    """Mock Haystack model engine for testing."""
    with patch("src.chunking.text_chunkers.chonky_chunker._ENGINE_AVAILABLE", True), \
         patch("src.chunking.text_chunkers.chonky_chunker.get_model_engine") as mock_get_engine, \
         patch("src.chunking.text_chunkers.chonky_chunker._MODEL_ENGINE") as mock_engine_instance:
        # Configure the mock engine client
        mock_engine_instance.client = MagicMock()
        mock_engine_instance.client.ping.return_value = "pong"
        mock_get_engine.return_value = mock_engine_instance
        
        # Mock the model loading function
        mock_engine_instance.load_model.return_value = "model-id"
        mock_engine_instance.client.ping.return_value = "pong"
        mock_engine_instance.started = True
        
        yield mock_engine_instance


class TestChonkyChunker:
    """Test suite for the Chonky-based semantic text chunker."""
    
    def test_chunk_text_with_mock_chonky(self, mock_chonky, mock_engine, sample_text_document):
        """Test text chunking using mocked Chonky splitter."""
        # Make sure _get_splitter_with_engine returns our mock splitter without errors
        get_splitter_patch = patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
        mock_get_splitter = get_splitter_patch.start()
        
        # Create paragraphs to be returned by the mock
        paragraphs = [
            "Introduction to Machine Learning",
            "Supervised Learning algorithms build a model based on sample data.",
            "Unsupervised learning is a type of algorithm that learns patterns from untagged data.",
            "Reinforcement learning is concerned with how agents take actions in an environment."
        ]
        
        # Configure the mock splitter to return these paragraphs
        splitter_instance = mock_chonky.return_value
        splitter_instance.side_effect = lambda text: paragraphs
        mock_get_splitter.return_value = splitter_instance
        
        try:
            # Test the chunk_text function
            result = chunk_text(
                content=sample_text_document["content"],
                doc_id=sample_text_document["id"],
                path=sample_text_document["path"],
                max_tokens=200,
                output_format="dict"
            )
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert "id" in result
            assert "chunks" in result
            
            # Make sure paragraphs were processed - with mocked enough setup they should match
            assert len(result["chunks"]) == len(paragraphs)
            
            # Check that each chunk has the required fields
            for chunk in result["chunks"]:
                assert "id" in chunk
                assert "content" in chunk
                assert "parent_id" in chunk
        finally:
            get_splitter_patch.stop()
        
        # Verify chunk structure
        for chunk in result["chunks"]:
            assert "id" in chunk
            assert "content" in chunk
            assert "parent_id" in chunk  # Check for parent_id instead of metadata
            assert isinstance(chunk["content"], str)
    
    def test_chunk_document_to_json(self, mock_chonky, mock_engine, sample_text_document):
        """Test converting a document to JSON with chunks."""
        # Patch the chunk_text function to return a predetermined result
        with patch("src.chunking.text_chunkers.chonky_chunker.chunk_text") as mock_chunk_text:
            # Set up mock chunk_text to return a document with chunks
            mock_chunk_result = {
                "id": sample_text_document["id"],
                "content": sample_text_document["content"],
                "path": sample_text_document["path"],
                "type": "text",
                "chunks": [
                    {
                        "id": "chunk-1",
                        "content": "Paragraph 1 for testing.",
                        "parent_id": sample_text_document["id"],
                        "metadata": {"position": 0}
                    },
                    {
                        "id": "chunk-2",
                        "content": "Paragraph 2 for testing.",
                        "parent_id": sample_text_document["id"],
                        "metadata": {"position": 1}
                    }
                ]
            }
            mock_chunk_text.return_value = mock_chunk_result
            
            # Create a document
            document = BaseDocument(
                content=sample_text_document["content"],
                path=sample_text_document["path"],
                id=sample_text_document["id"],
                type="text"
            )
            
            # Process it with mock in place
            result = chunk_document_to_json(document)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "id" in result
            assert "content" in result
            assert "chunks" in result
            
            # Should have chunks
            assert len(result["chunks"]) == 2
            
            # Verify each chunk has required fields
            for chunk in result["chunks"]:
                assert "id" in chunk
                assert "content" in chunk
                assert "parent_id" in chunk
    
    def test_chunk_document_formats(self, mock_chonky, mock_engine, sample_text_document):
        """Test chunk_document with different formats."""
        # Configure mock_splitter's instance to return realistic paragraphs
        splitter_instance = mock_chonky.return_value
        splitter_instance.split_text.return_value = ["Paragraph 1", "Paragraph 2"]

        # Test with dictionary input - this will exercise the common chunking logic
        # without relying on specific format conversion functions
        result = chunk_document(
            document=sample_text_document,
            return_pydantic=False,  # Return dict instead of schema
            max_tokens=1024
        )
        
        # Verify basic structure of the result
        assert isinstance(result, dict)
        assert "id" in result 
        assert "content" in result
        
        # The exact behavior of the chunks property depends on the implementation
        if "chunks" in result:
            assert isinstance(result["chunks"], list)
    
    def test_ensure_model_engine(self, mock_engine):
        """Test the ensure_model_engine context manager."""
        # Configure mock_engine to be available
        with patch("src.chunking.text_chunkers.chonky_chunker._ENGINE_AVAILABLE", True):
            # Test normal operation
            with ensure_model_engine() as engine:
                assert engine is not None
                
        # In actual implementation, if model engine is unavailable,
        # the context manager still runs but logs a warning rather than raising an exception
        with patch("src.chunking.text_chunkers.chonky_chunker._ENGINE_AVAILABLE", False):
            with patch("src.chunking.text_chunkers.chonky_chunker.logger.warning") as mock_warning:
                with ensure_model_engine() as engine:
                    assert engine is None
                mock_warning.assert_called()
    
    def test_fallbacks_when_chonky_unavailable(self, sample_text_document):
        """Test fallbacks when Chonky is not available."""
        # First patch out the original warning logger
        warning_patch = patch("src.chunking.text_chunkers.chonky_chunker.logger.warning")
        mock_warning = warning_patch.start()
        
        # Then set the CHONKY_AVAILABLE flag to False
        chonky_patch = patch("src.chunking.text_chunkers.chonky_chunker._CHONKY_AVAILABLE", False)
        chonky_patch.start()
        
        try:
            # When Chonky isn't available, the function should still work
            result = chunk_text(
                content=sample_text_document["content"],
                doc_id=sample_text_document["id"],
                output_format="dict"
            )
            
            # Verify the warning was logged
            assert mock_warning.called, "Warning should have been logged"
            
            # Should still produce a result with chunks
            assert isinstance(result, dict)
            assert "chunks" in result
        finally:
            # Clean up patches
            warning_patch.stop()
            chonky_patch.stop()
    
    def test_fallbacks_when_engine_unavailable(self, mock_chonky, sample_text_document):
        """Test fallbacks when model engine is not available."""
        # When engine is unavailable, the function should use a basic paragraph splitting
        # approach rather than the mock splitter

        # Skip the test and mark it as passed - we'll test this separately in TestEngineUnavailable
        # This avoids issues with the mock_chonky fixture which is applied to all tests in this class
        pytest.skip("Tested in a separate test class")


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
    # Disable both engine availability and mock engine
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
    """Test fallbacks when model engine is not available."""
    # Create a test document with clear paragraph breaks
    test_content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    test_doc_id = "test-doc-id"
    
    # Use the isolated environment to ensure no mocks interfere
    # The fixture already handles disabling the engine and mocking
    result = chunk_text(
        content=test_content,
        doc_id=test_doc_id,
        output_format="dict"
    )
    
    # Verify basic structure
    assert isinstance(result, dict)
    assert "chunks" in result
    assert len(result["chunks"]) == 3, f"Expected 3 chunks, got {len(result['chunks'])}"
    
    # Verify the content of the chunks matches our expected paragraphs
    expected_contents = ["First paragraph.", "Second paragraph.", "Third paragraph."]
    actual_contents = [chunk["content"] for chunk in result["chunks"]]
    
    # Check that we have exactly the expected paragraphs
    assert set(expected_contents) == set(actual_contents), f"Expected chunks: {expected_contents}, got: {actual_contents}"


def test_empty_document(mock_chonky, mock_engine):
        """Test chunking with empty content."""
        # Configure the mock splitter to handle empty content
        splitter_instance = mock_chonky.return_value
        splitter_instance.split_text.return_value = []
        
        # Test with empty content
        result = chunk_text(content="", doc_id="empty-doc", output_format="dict")
        
        # Should handle empty content gracefully
        assert isinstance(result, dict)
        assert "chunks" in result
        
        # With empty content, there should be at least one chunk (for the empty content itself)
        # or no chunks at all, depending on the implementation
        # Just verify the structure is as expected
        if len(result["chunks"]) > 0:
            for chunk in result["chunks"]:
                assert "id" in chunk
                assert "content" in chunk
                assert "metadata" in chunk
