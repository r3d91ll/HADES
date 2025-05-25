"""Unit tests for chonky_batch.py with optimized memory usage.

This module focuses on testing the chunk_document_batch function in a way
that avoids excessive memory usage while improving code coverage.
"""

import sys
import json
import os
import tempfile
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, mock_open

# Mock problematic imports first
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()

import pytest
from pydantic import BaseModel

# Import the module being tested
from src.chunking.text_chunkers.chonky_batch import (
    chunk_document_batch,
    process_document_to_dict,
    DocumentSchema
)


class TestChonkyBatchOptimized:
    """Test suite for Chonky batch processing with optimized memory usage."""

    def test_chunk_document_batch_empty_list(self):
        """Test handling of empty document list."""
        result = chunk_document_batch([])
        assert result == []

    def test_process_document_to_dict_basic(self):
        """Test basic functionality of process_document_to_dict."""
        # Create a test document
        doc = {
            "id": "test-doc",
            "content": "This is test content",
            "path": "/path/to/test.txt",
            "type": "text"
        }
        
        # Mock chunk_text
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
            # Return a simple list of chunks
            mock_chunk_text.return_value = [
                {"id": "chunk1", "content": "This is test content"}
            ]
            
            # Call the function
            result = process_document_to_dict(doc)
            
            # Verify result
            assert isinstance(result, dict)
            assert result["id"] == "test-doc"
            assert "chunks" in result
            assert len(result["chunks"]) == 1

    def test_chunk_document_batch_single_document_no_pydantic(self):
        """Test processing a single document without converting to Pydantic."""
        # Create a test document
        doc = {
            "id": "test-doc",
            "content": "This is test content",
            "path": "/path/to/test.txt",
            "type": "text"
        }
        
        # Mock process_document_to_dict
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process:
            mock_process.return_value = {
                "id": "test-doc",
                "content": "This is test content",
                "chunks": [{"id": "chunk1", "content": "chunk content"}]
            }
            
            # Call with return_pydantic=False
            result = chunk_document_batch([doc], return_pydantic=False)
            
            # Verify result
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert result[0]["id"] == "test-doc"
            assert "chunks" in result[0]
            
            # Verify process_document_to_dict was called
            mock_process.assert_called_once()

    def test_chunk_document_batch_with_pydantic(self):
        """Test processing a document with conversion to Pydantic."""
        # Create a test document
        doc = {
            "id": "test-doc",
            "content": "This is test content",
            "path": "/path/to/test.txt",
            "type": "text"
        }
        
        # Create a mock DocumentSchema
        mock_schema = MagicMock()
        
        # Mock both process_document_to_dict and model_validate
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process, \
             patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate', return_value=mock_schema):
            
            mock_process.return_value = {
                "id": "test-doc",
                "content": "This is test content",
                "chunks": [{"id": "chunk1", "content": "chunk content"}]
            }
            
            # Call with return_pydantic=True (default)
            result = chunk_document_batch([doc])
            
            # Verify result
            assert len(result) == 1
            assert result[0] == mock_schema

    def test_chunk_document_batch_parallel_processing(self):
        """Test parallel processing of multiple documents."""
        # Create test documents
        docs = [
            {"id": f"test-doc-{i}", "content": f"Content {i}"} 
            for i in range(3)
        ]
        
        # Mock process_document_to_dict to return different values for different docs
        def side_effect(doc, **kwargs):
            doc_id = doc.get("id", "unknown")
            return {
                "id": doc_id,
                "content": doc.get("content", ""),
                "chunks": [{"id": f"chunk-{doc_id}", "content": "chunk content"}]
            }
        
        # Mock process_document_to_dict
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict', side_effect=side_effect):
            # Process documents in parallel
            results = chunk_document_batch(docs, return_pydantic=False)
            
            # Verify results
            assert len(results) == 3
            # Check that all documents were processed
            result_ids = [doc["id"] for doc in results]
            assert "test-doc-0" in result_ids
            assert "test-doc-1" in result_ids
            assert "test-doc-2" in result_ids

    def test_chunk_document_batch_save_to_disk_simple(self):
        """Test saving processed documents to disk with minimal document size."""
        # Create a small test document to prevent memory issues
        doc = {
            "id": "test-doc-small",
            "content": "Small content",
            "type": "text"
        }
        
        # Setup temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock file operations
            mock_file = MagicMock()
            
            # Mock process_document_to_dict and open
            with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process, \
                 patch('builtins.open', mock_open()) as mock_file_open:
                
                # Return a simple processed document
                mock_process.return_value = {
                    "id": "test-doc-small",
                    "content": "Small content",
                    "chunks": [{"id": "chunk1", "content": "Small chunk"}]
                }
                
                # Call function with save_to_disk=True
                result = chunk_document_batch(
                    [doc], 
                    return_pydantic=False,
                    save_to_disk=True,
                    output_dir=temp_dir
                )
                
                # Verify mock_open was called
                assert mock_file_open.called
                # Verify json.dump was called on the file
                file_handle = mock_file_open.return_value.__enter__.return_value
                assert file_handle.write.called

    def test_chunk_document_batch_error_handling(self):
        """Test error handling in chunk_document_batch."""
        # Create test documents
        docs = [
            {"id": "good-doc", "content": "Good content"},
            {"id": "error-doc", "content": "Will cause error"}
        ]
        
        # Mock process_document_to_dict to raise an exception for the second document
        def side_effect(doc, **kwargs):
            if doc["id"] == "error-doc":
                raise ValueError("Test error")
            return {
                "id": doc["id"],
                "content": doc["content"],
                "chunks": [{"id": "chunk1", "content": "chunk content"}]
            }
        
        # Mock process_document_to_dict
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict', side_effect=side_effect):
            # Process documents
            results = chunk_document_batch(docs, return_pydantic=False)
            
            # Should only get the successful document
            assert len(results) == 1
            assert results[0]["id"] == "good-doc"

    def test_document_conversion_error_handling(self):
        """Test handling of Pydantic conversion errors."""
        # Create a test document
        doc = {"id": "test-doc", "content": "Test content"}
        
        # Mock process_document_to_dict and model_validate
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process, \
             patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate:
            
            # Return a processed document
            mock_process.return_value = {
                "id": "test-doc",
                "content": "Test content",
                "chunks": [{"id": "chunk1", "content": "chunk content"}]
            }
            
            # Make validation raise an exception
            mock_validate.side_effect = ValueError("Invalid document")
            
            # Call function
            result = chunk_document_batch([doc])
            
            # Should return the dict version when Pydantic conversion fails
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert result[0]["id"] == "test-doc"
