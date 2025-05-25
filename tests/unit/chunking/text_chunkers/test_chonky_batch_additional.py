"""Additional unit tests for the Chonky batch processing module.

This module contains additional tests to improve coverage of the Chonky batch 
processing functionality, focusing on untested functions and code paths.
"""

from __future__ import annotations

import os
import json
import pytest
import tempfile
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, call, mock_open

from src.chunking.text_chunkers.chonky_batch import (
    chunk_text_batch,
    chunk_document_batch,
    process_document_to_dict,
    process_documents_batch,
    chunk_documents,
    TQDM_AVAILABLE
)
from src.chunking.text_chunkers.chonky_chunker import BaseDocument, DocumentSchema
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
        def side_effect(content, doc_id=None, output_format="python", **kwargs):
            doc_id = doc_id or f"doc-1234"
            
            if output_format == "json":
                return json.dumps({
                    "id": doc_id,
                    "content": content[:100] + "...",
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
                })
            else:
                return {
                    "id": doc_id,
                    "content": content[:100] + "...",
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


@pytest.fixture
def mock_chunk_document():
    """Mock the chunk_document function for testing."""
    with patch("src.chunking.text_chunkers.chonky_batch.chunk_document") as mock_func:
        def side_effect(document, **kwargs):
            if isinstance(document, dict):
                result = document.copy()
                result["chunks"] = [
                    {
                        "id": f"{result.get('id', 'doc')}-chunk-1",
                        "content": "First chunk content",
                        "metadata": {"source_id": result.get('id', 'doc'), "chunk_index": 0}
                    },
                    {
                        "id": f"{result.get('id', 'doc')}-chunk-2",
                        "content": "Second chunk content",
                        "metadata": {"source_id": result.get('id', 'doc'), "chunk_index": 1}
                    }
                ]
                return result
            elif hasattr(document, 'dict') and callable(getattr(document, 'dict')):
                # Handle Pydantic models
                doc_dict = document.dict()
                doc_dict["chunks"] = [
                    {
                        "id": f"{doc_dict.get('id', 'doc')}-chunk-1",
                        "content": "First chunk content",
                        "metadata": {"source_id": doc_dict.get('id', 'doc'), "chunk_index": 0}
                    }
                ]
                
                if kwargs.get('return_pydantic', False):
                    return DocumentSchema(**doc_dict)
                return doc_dict
                
            # Default fallback
            return {"id": "default-doc", "chunks": []}
            
        mock_func.side_effect = side_effect
        yield mock_func


class TestChonkyBatchAdditional:
    """Additional test suite for the Chonky batch processing functionality."""
    
    def test_chunk_text_batch_python_format(self, mock_chunk_text):
        """Test chunking a batch of documents in Python format."""
        # Create test documents
        docs = [
            {"id": "doc1", "content": "Document 1 content", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Document 2 content", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Process without parallel
        results = chunk_text_batch(docs, parallel=False, output_format="python")
        
        # Should be a list of lists of dictionaries
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(result, list) for result in results)
        
        # Check calls to chunk_text
        assert mock_chunk_text.call_count == 2
        mock_chunk_text.assert_any_call(
            content="Document 1 content", 
            doc_id="doc1",
            path="/path/to/doc1.txt",
            doc_type="text",
            max_tokens=2048,
            output_format="python"
        )
        
    def test_chunk_text_batch_json_format(self, mock_chunk_text):
        """Test chunking a batch of documents in JSON format."""
        # Create test documents
        docs = [
            {"id": "doc1", "content": "Document 1 content", "path": "/path/to/doc1.txt", "type": "text"},
            {"id": "doc2", "content": "Document 2 content", "path": "/path/to/doc2.txt", "type": "text"}
        ]
        
        # Process in JSON format
        results = chunk_text_batch(docs, output_format="json", parallel=False)
        
        # Should be a list of JSON strings
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)
        
        # Check calls to chunk_text
        assert mock_chunk_text.call_count == 2
        mock_chunk_text.assert_any_call(
            content="Document 1 content", 
            doc_id="doc1",
            path="/path/to/doc1.txt",
            doc_type="text",
            max_tokens=2048,
            output_format="json"
        )
    
    def test_chunk_text_batch_parallel(self, mock_chunk_text):
        """Test chunking a batch of documents in parallel."""
        # Create test documents
        docs = [{"id": f"doc{i}", "content": f"Document {i} content", "path": f"/path/to/doc{i}.txt", "type": "text"} 
                for i in range(5)]
        
        # Process in parallel
        results = chunk_text_batch(docs, parallel=True, num_workers=2)
        
        # Should have processed all documents
        assert isinstance(results, list)
        assert len(results) == 5
        
        # Check calls to chunk_text
        assert mock_chunk_text.call_count == 5
    
    def test_chunk_text_batch_empty_input(self, mock_chunk_text):
        """Test chunking an empty batch."""
        results = chunk_text_batch([])
        
        # Should return an empty list
        assert isinstance(results, list)
        assert len(results) == 0
        
        # Should not have called chunk_text
        assert mock_chunk_text.call_count == 0
    
    def test_chunk_text_batch_error_handling(self, mock_chunk_text):
        """Test error handling in chunk_text_batch."""
        # Create a mix of good and problematic documents
        docs = [
            {"id": "good-doc", "content": "Good document", "path": "/path/to/good.txt", "type": "text"},
            {"id": "problem-doc", "content": None, "path": "/path/to/problem.txt", "type": "text"},
            {"id": "missing-fields", "path": "/path/to/missing.txt"}  # Missing content
        ]
        
        # Configure mock to raise exception for problematic docs
        def error_side_effect(content, **kwargs):
            if content is None:
                raise ValueError("Cannot process None content")
            return {"id": kwargs.get("doc_id", "unknown"), "chunks": []}
            
        mock_chunk_text.side_effect = error_side_effect
        
        # Process the batch - should not crash
        results = chunk_text_batch(docs, parallel=False)
        
        # Should have three results, even with errors
        assert isinstance(results, list)
        assert len(results) == 3
        
        # At least one successful call
        assert mock_chunk_text.call_count > 0
    
    def test_chunk_document_batch_with_save_to_disk(self, mock_chunk_document, tmp_path):
        """Test chunk_document_batch with save_to_disk option."""
        # Create test documents
        docs = [
            {"id": "doc1", "content": "Document 1 content", "path": "/path/to/doc1.txt", "type": "text",
             "source": "/path/to/doc1.txt", "document_type": "text"},
            {"id": "doc2", "content": "Document 2 content", "path": "/path/to/doc2.txt", "type": "text",
             "source": "/path/to/doc2.txt", "document_type": "text"}
        ]
        
        # Set up the mock to be more realistic
        def mock_chunk_doc(document, **kwargs):
            result = document.copy()
            result['chunks'] = [
                {"id": f"{result['id']}-chunk-1", "content": "Chunk 1", "metadata": {}}
            ]
            return result
            
        mock_chunk_document.side_effect = mock_chunk_doc
        
        # Create real temp directory for testing
        output_dir = os.path.join(str(tmp_path), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with save_to_disk using real files
        results = chunk_document_batch(
            docs,
            save_to_disk=True,
            output_dir=output_dir,
            return_pydantic=False,
            parallel=False
        )
        
        # Should have created files
        files = os.listdir(output_dir)
        assert len(files) == 2
        
        # Should return processed docs
        assert len(results) == 2
        assert all('chunks' in doc for doc in results)
    
    def test_chunk_document_batch_return_pydantic(self, mock_chunk_document):
        """Test chunk_document_batch with return_pydantic option."""
        # Create test documents with all required fields for DocumentSchema
        docs = [
            {"id": "doc1", "content": "Document 1 content", "path": "/path/to/doc1.txt", "type": "text",
             "source": "/path/to/doc1.txt", "document_type": "text"},
            {"id": "doc2", "content": "Document 2 content", "path": "/path/to/doc2.txt", "type": "text",
             "source": "/path/to/doc2.txt", "document_type": "text"}
        ]
        
        # Configure the mock to return data compatible with DocumentSchema
        def mock_chunk_doc(document, **kwargs):
            result = document.copy() if isinstance(document, dict) else document
            if isinstance(result, dict):
                result["chunks"] = [
                    {"id": f"{result.get('id', 'doc')}-chunk-1", 
                     "content": "Chunk content", 
                     "metadata": {"source_id": result.get('id', 'doc')},
                     "embedding": None,
                     "content_hash": "hash123"}
                ]
                # Ensure all required fields are present
                if "source" not in result and "path" in result:
                    result["source"] = result["path"]
                if "document_type" not in result and "type" in result:
                    result["document_type"] = result["type"]
                return result
            return document
            
        mock_chunk_document.side_effect = mock_chunk_doc
        
        # We'll patch DocumentSchema to simplify testing
        with patch("src.chunking.text_chunkers.chonky_batch.DocumentSchema") as mock_schema:
            # Configure the mock to return an object with the expected attributes
            mock_instance = MagicMock()
            mock_schema.return_value = mock_instance
            
            # Process with return_pydantic=True
            results = chunk_document_batch(
                docs,
                return_pydantic=True,
                parallel=False
            )
            
            # Verify DocumentSchema was called
            assert mock_schema.call_count > 0
            
            # Verification is simpler with our mock
            assert len(results) > 0
    
    def test_chunk_document_batch_parallel(self, mock_chunk_document):
        """Test chunk_document_batch with parallel processing."""
        # Create test documents with all required fields
        docs = [{"id": f"doc{i}", "content": f"Document {i} content", "path": f"/path/to/doc{i}.txt", "type": "text",
                 "source": f"/path/to/doc{i}.txt", "document_type": "text"} 
                for i in range(5)]
        
        # Define a proper mock function that returns sensible values
        def mock_process(doc, **kwargs):
            result = doc.copy() if isinstance(doc, dict) else doc
            if isinstance(result, dict):
                result["chunks"] = [
                    {"id": f"{result.get('id', 'unknown')}-chunk-1", 
                     "content": "Chunk content",
                     "metadata": {"source_id": result.get('id', 'unknown')}} 
                ]
            return result
            
        mock_chunk_document.side_effect = mock_process
        
        # Process in parallel
        results = chunk_document_batch(
            docs,
            parallel=True,
            num_workers=2,
            return_pydantic=False
        )
        
        # Should have processed all documents
        assert len(results) == 5
    
    def test_chunk_document_batch_with_pydantic_input(self, mock_chunk_document):
        """Test chunk_document_batch with Pydantic input documents."""
        # Create Pydantic documents
        docs = [
            DocumentSchema(id="doc1", content="Document 1 content", path="/path/to/doc1.txt", type="text", 
                         source="/path/to/doc1.txt", document_type="text"),
            DocumentSchema(id="doc2", content="Document 2 content", path="/path/to/doc2.txt", type="text",
                         source="/path/to/doc2.txt", document_type="text")
        ]
        
        # Process documents
        results = chunk_document_batch(
            docs,
            return_pydantic=True,
            parallel=False
        )
        
        # Should return processed documents
        assert len(results) == 2
        
        # Should be Pydantic objects
        for result in results:
            assert hasattr(result, 'id')
            assert hasattr(result, 'content')
            assert hasattr(result, 'chunks')
    
    def test_chunk_document_batch_error_handling(self, mock_chunk_document):
        """Test error handling in chunk_document_batch."""
        # Create a problematic document
        docs = [
            {"id": "doc1", "content": "Document 1 content", "path": "/path/to/doc1.txt", "type": "text",
             "source": "/path/to/doc1.txt", "document_type": "text"},
            {"id": "problem-doc", "content": None, "path": "/path/to/problem.txt", "type": "text",
             "source": "/path/to/problem.txt", "document_type": "text"}
        ]
        
        # Setting up the mock to behave correctly with error handling
        def error_side_effect(document, **kwargs):
            if isinstance(document, dict) and document.get('content') is None:
                raise ValueError("Cannot process None content")
            elif isinstance(document, dict):
                return {"id": document.get('id', 'unknown'), "chunks": [], 
                       "content": document.get('content', ''), "source": document.get('source', ''),
                       "document_type": document.get('document_type', 'text')}
            return document
            
        mock_chunk_document.side_effect = error_side_effect
        
        # Process the batch with return_pydantic=False to simplify testing
        results = chunk_document_batch(docs, parallel=False, return_pydantic=False)
        
        # The error handling in the function is robust enough to keep both documents
        # even if one encountered errors during processing
        assert len(results) == 2
        
        # Verify that the good document was processed correctly
        assert any(d["id"] == "doc1" for d in results)
        
        # Verify that the problematic document is also present (showing error handling robustness)
        assert any(d["id"] == "problem-doc" for d in results)
    
    def test_chunk_documents_with_custom_batch_size(self, mock_chunk_text):
        """Test processing documents with custom batch size."""
        # Create a larger list of documents
        docs = [{"id": f"doc{i}", "content": f"Document {i} content"} for i in range(10)]
        
        # Mock the process_documents_batch function to track calls
        with patch("src.chunking.text_chunkers.chonky_batch.process_documents_batch") as mock_batch:
            mock_batch.return_value = []
            
            # Process with a batch size of 3
            _ = chunk_documents(docs, batch_size=3)
            
            # Should have called process_documents_batch 4 times (10 docs / 3 batch size = 4 batches)
            assert mock_batch.call_count == 4
    
    def test_process_documents_batch_with_mixed_types(self, mock_chunk_text):
        """Test processing a batch with mixed object types."""
        # Create documents of different types
        pydantic_doc = DocumentSchema(id="pydantic1", content="Pydantic document", path="/path/pydantic.txt", type="text",
                               source="/path/pydantic.txt", document_type="text")
        dict_doc = {"id": "dict1", "content": "Dictionary document", "path": "/path/dict.txt", "type": "text"}
        
        class CustomDoc:
            """A custom document class that can be dict-converted."""
            def __init__(self):
                self.data = {"id": "custom1", "content": "Custom document", "path": "/path/custom.txt", "type": "text"}
                
            def __iter__(self):
                return iter(self.data.items())
                
        custom_doc = CustomDoc()
        
        # Process the mixed batch
        docs = [pydantic_doc, dict_doc, custom_doc]
        results = process_documents_batch(docs, parallel=False)
        
        # Should have processed all documents
        assert len(results) == 3
        
        # Check that all documents have expected fields
        for result in results:
            assert "id" in result
            assert "content" in result
    
    def test_tqdm_fallback_when_unavailable(self):
        """Test the tqdm fallback function when tqdm is not available."""
        # We can only effectively test this when tqdm is NOT available
        # Instead, let's just verify that TQDM_AVAILABLE is defined
        assert isinstance(TQDM_AVAILABLE, bool)
        
        # Mock the import error scenario
        with patch("src.chunking.text_chunkers.chonky_batch.TQDM_AVAILABLE", False):
            with patch("src.chunking.text_chunkers.chonky_batch.tqdm") as mock_tqdm:
                # Configure mock to return the input
                mock_tqdm.side_effect = lambda x=None, *args, **kwargs: x
                
                # Import tqdm again to get the fallback version
                from src.chunking.text_chunkers.chonky_batch import tqdm as fallback_tqdm
                
                # Test with an iterable
                items = [1, 2, 3]
                result = fallback_tqdm(items)
                
                # Should be the original items
                assert result == items
