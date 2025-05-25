"""Memory-safe tests for chonky_batch.py.

This module contains tests that avoid the problematic save_to_disk functionality
that causes extreme memory usage and OOM errors.
"""

import sys
from unittest.mock import Mock, patch, MagicMock

# Mock problematic imports first
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()

# Import the functions we want to test
from src.chunking.text_chunkers.chonky_batch import (
    process_document_to_dict,
    chunk_document_batch,
    process_documents_batch,
    chunk_documents,
    chunk_text_batch
)


def test_process_document_to_dict_basic():
    """Test basic functionality of process_document_to_dict."""
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


def test_process_document_to_dict_error_handling():
    """Test error handling in process_document_to_dict."""
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
        
        # Should return a document with default ID and empty chunks
        assert isinstance(result, dict)
        assert "id" in result
        assert "chunks" in result
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) == 0


def test_chunk_document_batch_no_disk():
    """Test chunk_document_batch without save_to_disk option."""
    # Create test document
    doc = {
        "id": "test-doc",
        "content": "Test content",
        "path": "/path/to/test.txt",
        "type": "text"
    }
    
    # Mock process_document_to_dict
    with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process:
        mock_process.return_value = {
            "id": "test-doc",
            "content": "Test content",
            "chunks": [{"id": "chunk1", "content": "Test content"}]
        }
        
        # Call with save_to_disk=False
        result = chunk_document_batch([doc], save_to_disk=False)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1
        assert mock_process.call_count == 1


def test_chunk_document_batch_parallel():
    """Test parallel processing in chunk_document_batch."""
    # Create test documents
    docs = [
        {"id": f"doc-{i}", "content": f"Content {i}"} for i in range(3)
    ]
    
    # Mock process_document_to_dict
    with patch('src.chunking.text_chunkers.chonky_batch.process_document_to_dict') as mock_process:
        # Return different values for different documents
        def side_effect(doc, **kwargs):
            doc_id = doc.get("id", "unknown")
            return {
                "id": doc_id,
                "content": doc.get("content", ""),
                "chunks": [{"id": f"chunk-{doc_id}", "content": "chunk content"}]
            }
        
        mock_process.side_effect = side_effect
        
        # Call with parallel=True
        result = chunk_document_batch(docs, parallel=True, save_to_disk=False)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 3
        assert mock_process.call_count == 3


def test_chunk_text_batch_basic():
    """Test basic functionality of chunk_text_batch."""
    docs = [
        {"id": "doc1", "content": "Content 1", "path": "/path/to/doc1.txt", "type": "text"},
        {"id": "doc2", "content": "Content 2", "path": "/path/to/doc2.txt", "type": "text"}
    ]
    
    # Mock chunk_text
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk:
        mock_chunk.return_value = [{"id": "chunk1", "content": "Content"}]
        
        # Call function
        result = chunk_text_batch(docs)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        # Each document's chunks should be a list
        for doc_chunks in result:
            assert isinstance(doc_chunks, list)


def test_process_documents_batch_basic():
    """Test basic functionality of process_documents_batch."""
    docs = [
        {"id": "doc1", "content": "Content 1"},
        {"id": "doc2", "content": "Content 2"}
    ]
    
    # Mock ThreadPool.map to return processed documents
    with patch('src.chunking.text_chunkers.chonky_batch.ThreadPool') as mock_pool_class:
        # Create mock pool and map result
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        
        # Setup processed documents
        processed_docs = [
            {"id": "doc1", "content": "Content 1", "chunks": [{"id": "chunk1"}]},
            {"id": "doc2", "content": "Content 2", "chunks": [{"id": "chunk2"}]}
        ]
        mock_pool.map.return_value = processed_docs
        
        # Call function
        result = process_documents_batch(docs)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[1]["id"] == "doc2"


def test_chunk_documents_basic():
    """Test basic functionality of chunk_documents."""
    docs = [
        {"id": "doc1", "content": "Content 1"},
        {"id": "doc2", "content": "Content 2"}
    ]
    
    # Mock process_documents_batch
    with patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch') as mock_process:
        # Return processed documents for each batch
        def side_effect(batch, **kwargs):
            return [{
                "id": doc["id"],
                "content": doc["content"],
                "chunks": [{"id": f"chunk-{doc['id']}", "content": doc["content"]}]
            } for doc in batch]
        
        mock_process.side_effect = side_effect
        
        # Call function with small batch size
        result = chunk_documents(docs, batch_size=1)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[1]["id"] == "doc2"
        
        # Should be called twice (once per batch)
        assert mock_process.call_count == 2


def test_empty_document_handling():
    """Test handling of empty document lists in all functions."""
    assert chunk_document_batch([]) == []
    assert process_documents_batch([]) == []
    assert chunk_documents([]) == []
    assert chunk_text_batch([]) == []
