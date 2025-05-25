"""Simple focused tests for improving chonky_batch.py coverage.

This module contains small, focused tests that target specific code paths
to improve coverage without excessive resource usage.
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock, call


# Import the module to test - mock problematic dependencies first
with patch.dict('sys.modules', {'torch': MagicMock(), 'transformers': MagicMock()}):
    from src.chunking.text_chunkers.chonky_batch import (
        process_document_to_dict,
        chunk_document_batch,
        process_documents_batch,
        chunk_documents,
        chunk_text_batch,
        DocumentSchema,
        TQDM_AVAILABLE
    )


def test_process_document_to_dict_error_handling():
    """Test error handling in process_document_to_dict."""
    # Create a test document
    doc = {"id": "test-doc", "content": "test content"}
    
    # Mock chunk_document to raise an exception
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
        mock_chunk.side_effect = Exception("Test chunking error")
        
        # Call the function
        result = process_document_to_dict(doc)
        
        # Should return a fallback document with the original fields
        assert isinstance(result, dict)
        assert result["id"] == "test-doc"
        assert result["content"] == "test content"
        assert "chunks" in result
        # Verify chunks is a list with at least one element (default chunk)
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0
        assert result["chunks"][0]["content"] == "test content"


def test_process_document_to_dict_with_model():
    """Test process_document_to_dict with a Pydantic model."""
    # Create a mock Pydantic model
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {
        "id": "model-doc",
        "content": "model content",
        "type": "text"
    }
    
    # Patch chunk_document to return a valid result
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
        mock_chunk.return_value = {
            "id": "model-doc",
            "content": "model content",
            "type": "text",
            "chunks": [{"id": "chunk1"}]
        }
        
        # Call the function
        result = process_document_to_dict(mock_model)
        
        # Check the result
        assert result["id"] == "model-doc"
        assert "chunks" in result
        assert len(result["chunks"]) == 1


def test_process_document_to_dict_with_unsupported_type():
    """Test process_document_to_dict with unsupported type."""
    # Try with an unsupported type (string)
    result = process_document_to_dict("not a valid document")
    
    # Should return a default document
    assert isinstance(result, dict)
    assert "id" in result
    assert "chunks" in result
    assert isinstance(result["chunks"], list)


def test_chunk_document_batch_with_empty_input():
    """Test chunk_document_batch with empty input list."""
    # Call with an empty list
    result = chunk_document_batch([])
    
    # Should return an empty list
    assert isinstance(result, list)
    assert len(result) == 0


def test_chunk_document_batch_with_save_to_disk_optimized():
    """Test chunk_document_batch with save_to_disk option using memory-optimized approach."""
    # Create a minimal test document to reduce memory usage
    doc = {"id": "tiny-doc", "content": "minimal content"}
    
    # Use explicit garbage collection and completely mock the file operations
    with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process, \
         patch('src.chunking.text_chunkers.chonky_batch.os.makedirs') as mock_makedirs, \
         patch('builtins.open') as mock_open, \
         patch('json.dump') as mock_json_dump:
        
        # Configure mocks with minimal objects
        mock_process.return_value = {"id": "tiny-doc", "content": "minimal content", "chunks": []}
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Call with save_to_disk=True but only process a single small document
        result = chunk_document_batch(
            [doc], 
            save_to_disk=True, 
            output_dir="/tmp/test_output",
            return_pydantic=False  # Avoid additional conversion overhead
        )
        
        # Basic verification without storing large objects
        assert mock_makedirs.called
        assert mock_open.called
        assert mock_json_dump.called
        assert len(result) == 1
        
        # Clear references to help garbage collection
        result = None
        mock_process = None
        mock_file = None
        
        # Force garbage collection
        import gc
        gc.collect()


def test_chunk_text_batch_output_formats():
    """Test chunk_text_batch with different output formats."""
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1", "path": "/path/1", "type": "text"},
        {"id": "doc2", "content": "content 2", "path": "/path/2", "type": "text"}
    ]
    
    # Mock chunk_text for different formats
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
        # Set up for Python format
        mock_chunk.return_value = [{"id": "chunk1", "content": "chunk content"}]
        
        # Test Python format
        result = chunk_text_batch(docs, output_format="python", parallel=False)
        assert isinstance(result, list)
        assert len(result) == 2  # One list of chunks per document
        assert isinstance(result[0], list)  # Each element is a list of chunks
        
        # Reset and set up for JSON format
        mock_chunk.reset_mock()
        mock_chunk.return_value = json.dumps([{"id": "chunk1", "content": "chunk content"}])
        
        # Test JSON format
        result = chunk_text_batch(docs, output_format="json", parallel=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], str)  # Each element is a JSON string


def test_chunk_documents_with_progress():
    """Test chunk_documents with progress reporting."""
    # Create test documents
    docs = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(5)]
    
    # Mock dependencies
    with patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process, \
         patch('src.chunking.text_chunkers.chonky_batch.tqdm') as mock_tqdm:
        
        # Configure mocks
        mock_process.return_value = docs  # Return unmodified
        mock_tqdm.return_value = docs  # Simple pass-through
        
        # Call with progress=True
        result = chunk_documents(docs, progress=True, batch_size=2)
        
        # Should have used tqdm
        mock_tqdm.assert_called_once()
        
        # Should have processed in batches
        assert mock_process.call_count == 3  # 5 docs with batch_size=2 means 3 calls


def test_process_documents_batch_parallel_processing():
    """Test parallel processing in process_documents_batch."""
    # Create test documents
    docs = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(3)]
    
    # Mock ThreadPool
    with patch('src.chunking.text_chunkers.chonky_batch.ThreadPool') as mock_pool, \
         patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
        
        # Configure pool mock
        mock_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_instance
        mock_instance.map.return_value = [doc.copy() for doc in docs]  # Return copies
        
        # Call with parallel=True
        result = process_documents_batch(docs, parallel=True, num_workers=2)
        
        # Verify ThreadPool was created with correct num_workers
        mock_pool.assert_called_once_with(processes=2)
        
        # Verify map was called with process_single function
        mock_instance.map.assert_called_once()


def test_chunk_text_batch_with_parallel_processing():
    """Test parallel processing in chunk_text_batch."""
    # Create test documents
    docs = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(3)]
    
    # Mock ThreadPool
    with patch('src.chunking.text_chunkers.chonky_batch.ThreadPool') as mock_pool, \
         patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
        
        # Configure pool mock
        mock_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_instance
        # Return a list of chunks for each document
        mock_instance.map.return_value = [[{"id": f"chunk-{i}"}] for i in range(3)]
        
        # Call with parallel=True
        result = chunk_text_batch(docs, parallel=True, num_workers=2)
        
        # Verify ThreadPool was created with correct num_workers
        mock_pool.assert_called_once_with(processes=2)
        
        # Verify result format
        assert isinstance(result, list)
        assert len(result) == 3
