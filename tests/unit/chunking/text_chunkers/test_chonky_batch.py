"""Unit tests for the Chonky batch processing module.

This module contains comprehensive tests for the Chonky batch processing functionality,
ensuring that multiple documents can be efficiently and correctly processed in batches.
"""

from __future__ import annotations

import os
import pytest
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, ANY

from src.chunking.text_chunkers.chonky_batch import (
    chunk_text_batch,
    chunk_document_batch,
    process_document_to_dict,
    process_documents_batch,
    chunk_documents
)
from src.chunking.text_chunkers.chonky_chunker import BaseDocument
from tests.unit.common_fixtures import (
    sample_text_document,
    sample_code_document,
    create_expected_chunks
)


@pytest.fixture
def mock_chunk_text():
    """Mock the chunk_text function for testing."""
    with patch("src.chunking.text_chunkers.chonky_batch.chunk_text") as mock_func:
        # Configure mock to return realistic chunked documents
        def side_effect(content, doc_id=None, **kwargs):
            doc_id = doc_id or f"doc-{uuid.uuid4().hex}"
            return {
                "id": doc_id,
                "content": content[:100] + "...",  # Truncate for readability
                "chunks": [
                    {
                        "id": f"{doc_id}-chunk-1",
                        "content": content[:50] + "...",
                        "metadata": {"source_id": doc_id, "chunk_index": 0}
                    },
                    {
                        "id": f"{doc_id}-chunk-2",
                        "content": content[50:100] + "...",
                        "metadata": {"source_id": doc_id, "chunk_index": 1}
                    }
                ]
            }
        
        mock_func.side_effect = side_effect
        yield mock_func


class TestChonkyBatch:
    """Test suite for the Chonky batch processing functionality."""
    
    def test_process_document_to_dict_with_dict(self, mock_chunk_text, sample_text_document):
        """Test processing a document that's already a dictionary."""
        result = process_document_to_dict(sample_text_document)
        
        # Should return the input if it's already a dict
        assert result == sample_text_document
    
    def test_process_document_to_dict_with_base_document(self, mock_chunk_text):
        """Test processing a BaseDocument object."""
        # Create a BaseDocument
        doc = BaseDocument(
            content="Test content",
            path="/path/to/doc.txt",
            id="test-doc",
            type="text"
        )
        
        result = process_document_to_dict(doc)
        
        # Should convert to dict
        assert isinstance(result, dict)
        assert result["id"] == "test-doc"
        assert result["content"] == "Test content"
        assert result["path"] == "/path/to/doc.txt"
        assert result["type"] == "text"
    
    def test_process_document_to_dict_with_pydantic_model(self, mock_chunk_text):
        """Test processing a Pydantic model."""
        # Mock a Pydantic-like model with dict method
        class MockPydanticDoc:
            def dict(self):
                return {
                    "id": "pydantic-doc",
                    "content": "Pydantic content",
                    "path": "/path/to/pydantic.txt",
                    "type": "text"
                }
        
        doc = MockPydanticDoc()
        result = process_document_to_dict(doc)
        
        # Should use the dict method
        assert isinstance(result, dict)
        assert result["id"] == "pydantic-doc"
        assert result["content"] == "Pydantic content"
    
    def test_process_document_to_dict_fallback(self, mock_chunk_text):
        """Test fallback behavior for other object types."""
        # Test with a minimal object
        class MinimalDoc:
            pass
        
        doc = MinimalDoc()
        result = process_document_to_dict(doc)
        
        # Should create a fallback document
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "type" in result
        assert result["type"] == "text"
        assert result["content"] == ""
    
    def test_process_document_to_dict_error_handling(self, mock_chunk_text):
        """Test error handling in document processing."""
        # Mock an object that raises an exception when dict is called
        class ProblemDoc:
            def dict(self):
                raise ValueError("Simulated error")
        
        doc = ProblemDoc()
        result = process_document_to_dict(doc)
        
        # Should handle the error and return a fallback document
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert result["content"] == ""
        
    def test_process_documents_batch_empty(self, mock_chunk_text):
        """Test batch processing with empty input."""
        result = process_documents_batch([])
        
        # Should return an empty list
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_process_documents_batch(self, mock_chunk_text, sample_text_document, sample_code_document):
        """Test batch processing of multiple documents."""
        # Create a batch of documents
        docs = [sample_text_document, sample_code_document]
        
        # Process the batch
        results = process_documents_batch(docs)
        
        # Should have two processed documents
        assert len(results) == 2
        
        # Verify each document has chunks
        for result in results:
            assert "chunks" in result
            assert len(result["chunks"]) > 0
            
        # Check that chunk_text was called twice
        assert mock_chunk_text.call_count == 2
    
    def test_chunk_documents(self, mock_chunk_text, sample_text_document, sample_code_document):
        """Test the main chunk_documents function."""
        # Create a list of documents
        docs = [sample_text_document, sample_code_document]
        
        # Test with default parameters
        results = chunk_documents(docs)
        
        # Should return all processed documents
        assert len(results) == 2
        
        # Test with batch_size parameter
        with patch("src.chunking.text_chunkers.chonky_batch.process_documents_batch") as mock_batch:
            mock_batch.return_value = []
            _ = chunk_documents(docs, batch_size=1)
            
            # Should be called twice with batch_size=1
            assert mock_batch.call_count == 2
    
    def test_chunk_documents_with_progress(self, mock_chunk_text, sample_text_document):
        """Test chunking documents with progress reporting."""
        # Create multiple documents
        docs = [sample_text_document.copy() for _ in range(5)]
        
        # Test with progress=True
        with patch("src.chunking.text_chunkers.chonky_batch.tqdm") as mock_tqdm:
            mock_tqdm.return_value.__iter__.return_value = iter([docs])
            results = chunk_documents(docs, progress=True)
            
            # Should use tqdm for progress
            mock_tqdm.assert_called_once()
        
        # Should have processed all documents
        assert len(results) == 5
    
    def test_error_handling_in_batch(self, mock_chunk_text, sample_text_document):
        """Test error handling during batch processing."""
        # Create a document that will cause an error
        problem_doc = {"id": "problem-doc", "content": None}  # content is None, will cause error
        
        # Mix with a good document
        docs = [sample_text_document, problem_doc]
        
        # Configure mock to raise an exception for the problem doc
        def mock_side_effect(content, doc_id=None, **kwargs):
            if content is None:
                raise ValueError("Cannot process None content")
            return {"id": doc_id, "chunks": []}
            
        mock_chunk_text.side_effect = mock_side_effect
        
        # Create a mock for chunk_document that returns the original doc on error
        with patch("src.chunking.text_chunkers.chonky_batch.chunk_document") as mock_chunk_doc:
            # When chunk_document is called with a problematic document, return the document itself
            def chunk_doc_side_effect(doc, **kwargs):
                if doc.get("content") is None:
                    # Return the original document
                    return doc
                # For good docs, return with empty chunks list
                result = dict(doc)
                result["chunks"] = []
                return result
                
            mock_chunk_doc.side_effect = chunk_doc_side_effect
            
            # Process batch - should not crash
            results = process_documents_batch(docs)
            
            # Should still return two results
            assert len(results) == 2
            
            # The second result should still be the problem doc
            assert results[1]["id"] == "problem-doc"
            # The problem document might not have chunks, just check it exists
            assert results[1] is not None
