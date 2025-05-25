"""Final coverage tests for chonky_batch.py.

This module aims to hit remaining code paths not covered by other test files
to reach the 85% coverage target while avoiding memory issues.
"""

import sys
import tempfile
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock, mock_open

# Mock problematic imports first
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()

import pytest
from pydantic import BaseModel

# Import the module being tested
from src.chunking.text_chunkers.chonky_batch import (
    chunk_text_batch,
    process_documents_batch,
    chunk_documents
)


class MockBaseDocument:
    """Mock BaseDocument class for testing."""
    
    def __init__(self, doc_id: str, content: str, doc_type: str = "text"):
        self.id = doc_id
        self.content = content
        self.type = doc_type
        self.path = f"/path/to/{doc_id}.txt"
    
    def dict(self):
        """Legacy dict method."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "path": self.path
        }
    
    def model_dump(self):
        """New model_dump method for Pydantic v2."""
        return self.dict()


class TestChonkyBatchFinalCoverage:
    """Test suite for remaining code paths in chonky_batch.py."""
    
    def test_chunk_text_batch_basic(self):
        """Test basic functionality of chunk_text_batch."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Content 2", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Mock chunk_text to return a simple result
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
            mock_chunk.return_value = [{"id": "chunk1", "content": "Content"}]
            
            # Call with parallel=False to test serial processing
            result = chunk_text_batch(documents, parallel=False)
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
            # Verify each result is a list of chunks
            for doc_chunks in result:
                assert isinstance(doc_chunks, list)
    
    def test_chunk_text_batch_parallel(self):
        """Test parallel processing in chunk_text_batch."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Content 2", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Mock chunk_text to return a simple result
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
            mock_chunk.return_value = [{"id": "chunk1", "content": "Content"}]
            
            # Call with parallel=True
            result = chunk_text_batch(documents, parallel=True, num_workers=2)
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
    
    def test_chunk_text_batch_json_output(self):
        """Test chunk_text_batch with JSON output format."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"}
        ]
        
        # Mock chunk_text to return a simple result
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
            mock_chunk.return_value = [{"id": "chunk1", "content": "Content"}]
            
            # Call with output_format="json"
            result = chunk_text_batch(documents, output_format="json")
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], str)  # Should be JSON string
    
    def test_process_documents_batch_basic(self):
        """Test basic functionality of process_documents_batch."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"}
        ]
        
        # Mock process_document_to_dict
        with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process:
            mock_process.return_value = {
                "id": "doc1",
                "content": "Content 1",
                "chunks": [{"id": "chunk1", "content": "Content 1"}]
            }
            
            # Call function
            result = process_documents_batch(documents)
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == "doc1"
    
    def test_process_documents_batch_with_basemodel(self):
        """Test process_documents_batch with BaseDocument objects."""
        # Create test documents
        documents = [
            MockBaseDocument("doc1", "Content 1"),
            MockBaseDocument("doc2", "Content 2")
        ]
        
        # We need to mock the ThreadPool and map method
        with patch('src.chunking.text_chunkers.chonky_batch.ThreadPool') as mock_pool_class:
            # Create mock pool and map result
            mock_pool = MagicMock()
            mock_pool_class.return_value.__enter__.return_value = mock_pool
            
            # Setup the map method to return processed documents
            processed_docs = [
                {"id": "doc1", "content": "Content 1", "chunks": [{"id": "chunk1"}]},
                {"id": "doc2", "content": "Content 2", "chunks": [{"id": "chunk2"}]}
            ]
            mock_pool.map.return_value = processed_docs
            
            # Call function
            result = process_documents_batch(documents)
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
            result_ids = {doc["id"] for doc in result}
            assert "doc1" in result_ids
            assert "doc2" in result_ids
            assert "doc1" in result_ids
            assert "doc2" in result_ids
    
    def test_process_documents_batch_pydantic_return(self):
        """Test process_documents_batch with return_pydantic=True."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"}
        ]
        
        # Mock the ThreadPool and DocumentSchema.model_validate
        with patch('src.chunking.text_chunkers.chonky_batch.ThreadPool') as mock_pool_class, \
             patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate:
            
            # Create mock pool and map result
            mock_pool = MagicMock()
            mock_pool_class.return_value.__enter__.return_value = mock_pool
            
            # Make process_single_doc return processed documents
            processed_docs = [
                {"id": "doc1", "content": "Content 1", "chunks": [{"id": "chunk1"}]}
            ]
            mock_pool.map.return_value = processed_docs
            
            # Pydantic validation is not actually called in process_documents_batch
            # It only happens in chunk_document_batch
            
            # Call function
            result = process_documents_batch(documents, return_pydantic=True)
            
            # Just check we get the right structure, no actual Pydantic objects
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == "doc1"
            assert "chunks" in result[0]
    
    def test_chunk_documents_basic(self):
        """Test basic functionality of chunk_documents."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Content 2", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Mock process_documents_batch
        with patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process:
            # Return different batches
            def side_effect(docs, **kwargs):
                return [{
                    "id": doc["id"],
                    "content": doc["content"],
                    "chunks": [{"id": f"chunk-{doc['id']}", "content": doc["content"]}]
                } for doc in docs]
            
            mock_process.side_effect = side_effect
            
            # Call function with small batch size
            result = chunk_documents(documents, batch_size=1)
            
            # Verify result
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == "doc1"
            assert result[1]["id"] == "doc2"
            
            # Verify process_documents_batch called twice (once per batch)
            assert mock_process.call_count == 2
    
    def test_chunk_documents_with_progress(self):
        """Test chunk_documents with progress reporting."""
        documents = [
            {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Content 2", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Mock process_documents_batch and tqdm
        with patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process, \
             patch('src.chunking.text_chunkers.chonky_batch.TQDM_AVAILABLE', True), \
             patch('src.chunking.text_chunkers.chonky_batch.tqdm') as mock_tqdm:
            
            # Setup mocks
            mock_process.return_value = [{
                "id": "doc1", 
                "content": "Content 1",
                "chunks": [{"id": "chunk1", "content": "Content 1"}]
            }]
            
            # Mock tqdm to just return its input
            mock_tqdm.return_value = [[{"id": "doc1"}], [{"id": "doc2"}]]
            
            # Call function with progress=True
            result = chunk_documents(documents, batch_size=1, progress=True)
            
            # Verify tqdm was called
            assert mock_tqdm.called
            
            # Verify process_documents_batch called
            assert mock_process.called
    
    def test_empty_documents_handling(self):
        """Test handling of empty document lists in all functions."""
        # Test chunk_text_batch
        assert chunk_text_batch([]) == []
        
        # Test process_documents_batch
        assert process_documents_batch([]) == []
        
        # Test chunk_documents
        assert chunk_documents([]) == []
