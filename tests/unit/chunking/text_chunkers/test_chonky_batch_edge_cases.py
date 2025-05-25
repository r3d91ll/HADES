"""Tests for edge cases and less-covered paths in the chonky_batch module.

This module focuses on testing edge cases and paths that aren't well covered
by the existing test suite, aiming to improve overall test coverage.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock, ANY
from typing import Dict, Any, List, Optional, Union

from src.chunking.text_chunkers.chonky_batch import (
    chunk_text_batch,
    process_document_to_dict,
    chunk_document_batch,
    process_documents_batch,
    chunk_documents,
    TQDM_AVAILABLE,
    BaseDocument,
    DocumentSchema
)


class TestChonkyBatchEdgeCases:
    """Test suite for edge cases in chonky_batch."""
    
    def test_process_document_to_dict_with_invalid_input(self):
        """Test processing document with invalid input types."""
        # Test with an unsupported type
        with pytest.raises(ValueError) as excinfo:
            process_document_to_dict("not a valid document type")
        
        assert "Unsupported document type" in str(excinfo.value)
    
    def test_process_document_to_dict_error_handling(self):
        """Test error handling in process_document_to_dict."""
        # Mock chunk_document to raise an exception
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
            mock_chunk.side_effect = Exception("Test exception")
            
            # Test with dictionary input
            doc = {"id": "test-doc", "content": "test content"}
            result = process_document_to_dict(doc)
            
            # Should return a fallback document with the original fields
            assert isinstance(result, dict)
            assert result["id"] == "test-doc"
            assert result["content"] == "test content"
            assert "chunks" in result
            assert result["chunks"] == []
    
    def test_chunk_document_batch_with_empty_list(self):
        """Test chunk_document_batch with an empty list."""
        result = chunk_document_batch([])
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_chunk_document_batch_with_output_dir_creation(self):
        """Test chunk_document_batch creates output directory if it doesn't exist."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tempdir:
            # Path that doesn't exist yet
            nonexistent_dir = os.path.join(tempdir, "new_output_dir")
            
            # Create test document
            doc = {"id": "test-doc", "content": "test content"}
            
            # Mock chunk_document to avoid actual chunking
            with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk, \
                 patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate:
                 
                mock_chunk.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
                mock_validate.return_value = MagicMock()
                
                # Call with save_to_disk=True and a non-existent output directory
                chunk_document_batch([doc], save_to_disk=True, output_dir=nonexistent_dir)
                
                # Directory should have been created
                assert os.path.exists(nonexistent_dir)
    
    def test_chunk_document_batch_with_failed_save(self):
        """Test chunk_document_batch handles errors during save_to_disk."""
        # Create a test document
        doc = {"id": "test-doc", "content": "test content"}
        
        # Mock dependencies
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk, \
             patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate, \
             patch('builtins.open') as mock_open:
            
            # Set up mocks
            mock_chunk.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
            mock_validate.return_value = MagicMock()
            mock_open.side_effect = IOError("Failed to open file")
            
            # Call with save_to_disk=True
            with tempfile.TemporaryDirectory() as tempdir:
                result = chunk_document_batch([doc], save_to_disk=True, output_dir=tempdir)
                
                # Should still return processed document even if save fails
                assert len(result) == 1
                assert isinstance(result[0], MagicMock)  # DocumentSchema mock
    
    def test_process_documents_batch_with_single_document(self):
        """Test process_documents_batch with a single document."""
        # Create a test document
        doc = {"id": "test-doc", "content": "test content"}
        
        # Mock chunk_document to avoid actual chunking
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
            mock_chunk.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
            
            # Process a single document
            result = process_documents_batch([doc])
            
            # Should process it without using parallelism
            assert len(result) == 1
            assert result[0]["id"] == "test-doc"
            assert "chunks" in result[0]
    
    def test_process_documents_batch_chunking_errors(self):
        """Test process_documents_batch handles chunking errors."""
        # Create test documents
        docs = [
            {"id": "doc1", "content": "content 1"},
            {"id": "doc2", "content": "content 2"},
            {"id": "doc3", "content": "content 3"}
        ]
        
        # Mock chunk_document to raise an exception for the second document
        def mock_chunk_doc(doc, **kwargs):
            if doc["id"] == "doc2":
                raise Exception("Test chunking error")
            return {"id": doc["id"], "content": doc["content"], "chunks": []}
        
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk:
            mock_chunk.side_effect = mock_chunk_doc
            
            # Process documents
            result = process_documents_batch(docs)
            
            # Should still return all 3 documents
            assert len(result) == 3
            
            # First and third should have been processed normally
            assert result[0]["id"] == "doc1"
            assert result[2]["id"] == "doc3"
            
            # Second should have a fallback with empty chunks
            assert result[1]["id"] == "doc2"
            assert result[1]["chunks"] == []
    
    def test_chunk_documents_with_custom_batch_size(self):
        """Test chunk_documents with different batch sizes."""
        # Create 15 test documents
        docs = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(15)]
        
        # Create a counter to verify batch processing
        processed_batches = []
        
        # Mock process_documents_batch to track batch sizes
        def mock_process_batch(batch_docs, **kwargs):
            processed_batches.append(len(batch_docs))
            return batch_docs  # Return unmodified for simplicity
        
        with patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process:
            mock_process.side_effect = mock_process_batch
            
            # Process with batch_size=5
            result = chunk_documents(docs, batch_size=5, progress=False)
            
            # Should have processed 3 batches of size 5
            assert processed_batches == [5, 5, 5]
            assert len(result) == 15
            
            # Reset for next test
            processed_batches.clear()
            
            # Process with batch_size=7
            result = chunk_documents(docs, batch_size=7, progress=False)
            
            # Should have processed 2 batches of size 7 and 1 batch of size 1
            assert processed_batches == [7, 7, 1]
            assert len(result) == 15
    
    def test_chunk_text_batch_with_different_output_format(self):
        """Test chunk_text_batch handles both output formats correctly."""
        docs = [
            {"id": "doc1", "content": "content 1", "path": "/path/1", "type": "text"},
            {"id": "doc2", "content": "content 2", "path": "/path/2", "type": "text"}
        ]
        
        # Mock chunk_text for both formats
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
            # For Python format
            mock_chunk.return_value = [{"id": "chunk1", "content": "chunk content"}]
            
            # Test Python format
            result = chunk_text_batch(docs, output_format="python")
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], list)
            
            # Test JSON format
            mock_chunk.return_value = json.dumps([{"id": "chunk1", "content": "chunk content"}])
            result = chunk_text_batch(docs, output_format="json")
            assert isinstance(result, list)
            assert isinstance(result[0], str)
    
    def test_chunk_text_batch_error_handling_variations(self):
        """Test different error cases in chunk_text_batch."""
        docs = [
            {"id": "doc1", "content": "content 1"},
            {"id": "doc2", "content": "content 2"},
        ]
        
        # Test handling when chunk_text returns unusual types
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
            # Case 1: Returns None
            mock_chunk.return_value = None
            result = chunk_text_batch(docs)
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(len(r) == 0 for r in result)  # Empty lists
            
            # Case 2: Returns dict without 'chunks'
            mock_chunk.return_value = {"something": "else"}
            result = chunk_text_batch(docs)
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(len(r) == 0 for r in result)  # Empty lists
            
            # Case 3: For JSON format, returns invalid type
            mock_chunk.return_value = {"not": "a string"}
            result = chunk_text_batch(docs, output_format="json")
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(r == "[]" for r in result)  # Empty JSON arrays
    
    def test_chunk_documents_with_progress_reporting(self):
        """Test chunk_documents with progress reporting enabled."""
        # Create test documents
        docs = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(5)]
        
        # Mock tqdm to verify it's called when progress=True
        mock_tqdm = MagicMock()
        mock_tqdm.return_value.__iter__.return_value = range(1)
        
        with patch('src.chunking.text_chunkers.chonky_batch.tqdm', mock_tqdm), \
             patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process:
            
            mock_process.return_value = docs  # Return unmodified for simplicity
            
            # Call with progress=True
            result = chunk_documents(docs, progress=True)
            
            # Should have used tqdm
            mock_tqdm.assert_called_once()
            assert len(result) == 5
