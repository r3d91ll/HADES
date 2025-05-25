"""Minimal test module for chonky_batch.py coverage improvements.

This module focuses only on the core functionality with minimal mocking
to avoid memory issues while still improving coverage.
"""

import sys
import json
from unittest.mock import Mock, patch, MagicMock

# Mock problematic imports first
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()

# Import the target function
from src.chunking.text_chunkers.chonky_batch import process_document_to_dict


def test_process_document_with_dict_input():
    """Test process_document_to_dict with a dictionary input."""
    # Create a test document
    doc = {
        "id": "test-doc",
        "content": "This is test content",
        "path": "/path/to/test.txt",
        "type": "text"
    }
    
    # Mock chunk_text to return a simple result
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
        mock_chunk_text.return_value = [
            {"id": "chunk1", "content": "This is test content"}
        ]
        
        # Call the function
        result = process_document_to_dict(doc)
        
        # Verify the result
        assert isinstance(result, dict)
        assert result["id"] == "test-doc"
        assert "chunks" in result
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["id"] == "chunk1"


def test_process_document_with_empty_content():
    """Test process_document_to_dict with a document that has empty content."""
    # Create a document with empty content
    doc = {
        "id": "empty-doc",
        "content": "",
        "path": "/path/to/empty.txt",
        "type": "text"
    }
    
    # Mock chunk_text (it shouldn't be called for empty content)
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
        # Call the function
        result = process_document_to_dict(doc)
        
        # Should return the original document without calling chunk_text
        assert result["id"] == "empty-doc"
        assert not mock_chunk_text.called


def test_process_document_with_chunk_text_returning_dict():
    """Test process_document_to_dict when chunk_text returns a dictionary."""
    # Create a test document
    doc = {
        "id": "test-doc",
        "content": "This is test content",
        "path": "/path/to/test.txt",
        "type": "text"
    }
    
    # Mock chunk_text to return a dictionary with chunks
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
        mock_chunk_text.return_value = {
            "id": "test-doc",
            "content": "This is test content",
            "chunks": [{"id": "chunk1", "content": "This is test content"}],
            "extra_field": "extra value"
        }
        
        # Call the function
        result = process_document_to_dict(doc)
        
        # Verify the result
        assert "chunks" in result
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["id"] == "chunk1"
        assert "extra_field" in result
        assert result["extra_field"] == "extra value"


def test_process_document_with_error():
    """Test process_document_to_dict handles errors gracefully."""
    # Create a test document
    doc = {
        "id": "test-doc",
        "content": "This is test content",
        "path": "/path/to/test.txt",
        "type": "text"
    }
    
    # Mock chunk_text to raise an exception
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
        mock_chunk_text.side_effect = Exception("Test error")
        
        # Call the function
        result = process_document_to_dict(doc)
        
        # Should return a fallback document with generated ID and empty chunks
        assert isinstance(result, dict)
        assert "id" in result  # ID is generated, not preserved
        assert "chunks" in result
        assert isinstance(result["chunks"], list)
        # When an error occurs, chunks list is empty
        assert len(result["chunks"]) == 0
