"""Targeted test module for chonky_batch.py coverage improvements.

This module takes a minimal approach with focused tests that isolate
specific logic paths to improve coverage without excessive resource usage.
"""

import sys
import os
import json
from typing import List, Dict, Any, Union
import pytest
from unittest.mock import patch, MagicMock, Mock, call

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


# Very simple targeted tests for the process_document_to_dict function
@patch('src.chunking.text_chunkers.chonky_batch.chunk_text')
def test_process_document_empty_content(mock_chunk_text):
    """Test handling of documents with empty content."""
    # Create document with empty content
    doc = {"id": "doc1", "content": ""}
    
    # Call the function
    result = process_document_to_dict(doc)
    
    # Should return the document without calling chunk_text
    assert result["id"] == "doc1"
    assert not mock_chunk_text.called


@patch('src.chunking.text_chunkers.chonky_batch.chunk_text')
def test_process_document_chunk_text_returns_dict(mock_chunk_text):
    """Test when chunk_text returns a dictionary with chunks."""
    # Set up the mock
    mock_chunk_text.return_value = {
        "id": "doc1",
        "content": "test content",
        "chunks": [{"id": "chunk1", "content": "chunk content"}],
        "extra_field": "extra value"
    }
    
    # Create a test document
    doc = {"id": "doc1", "content": "test content"}
    
    # Call the function
    result = process_document_to_dict(doc)
    
    # Should have the chunks and extra fields
    assert "chunks" in result
    assert len(result["chunks"]) == 1
    assert result["extra_field"] == "extra value"


@patch('src.chunking.text_chunkers.chonky_batch.chunk_text')
def test_process_document_chunk_text_returns_list(mock_chunk_text):
    """Test when chunk_text returns a list of chunks."""
    # Set up the mock
    mock_chunk_text.return_value = [
        {"id": "chunk1", "content": "chunk content"}
    ]
    
    # Create a test document
    doc = {"id": "doc1", "content": "test content"}
    
    # Call the function
    result = process_document_to_dict(doc)
    
    # Should have the chunks
    assert "chunks" in result
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["id"] == "chunk1"


# Test the dict conversion logic
def test_process_document_dict_like_input():
    """Test processing a dict-like object."""
    # Create a dict-like object with a __dict__ method
    class DictLike:
        def __init__(self):
            self.id = "dict-like"
            self.content = "dict-like content"
            
        def __iter__(self):
            yield "id", self.id
            yield "content", self.content
    
    # Create an instance
    dict_like = DictLike()
    
    # Process it with patched chunk_text
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_text') as mock_chunk_text:
        mock_chunk_text.return_value = [{"id": "chunk1"}]
        
        # Call the function
        result = process_document_to_dict(dict_like)
        
        # Should have processed it correctly
        assert result["id"] == "dict-like"
        assert "chunks" in result


# Tests for chunk_document_batch function
@patch('src.chunking.text_chunkers.chonky_batch.chunk_document')
@patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate')
def test_chunk_document_batch_empty_input(mock_validate, mock_chunk_document):
    """Test chunk_document_batch with empty input."""
    # Call with empty list
    result = chunk_document_batch([])
    
    # Should return empty list
    assert isinstance(result, list)
    assert len(result) == 0
    # Shouldn't call any of the mocked functions
    mock_chunk_document.assert_not_called()
    mock_validate.assert_not_called()


@patch('src.chunking.text_chunkers.chonky_batch.chunk_document')
@patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate')
@patch('src.chunking.text_chunkers.chonky_batch.os.makedirs')
@patch('builtins.open')
def test_chunk_document_batch_save_to_disk(mock_open, mock_makedirs, mock_validate, mock_chunk_document):
    """Test chunk_document_batch with save_to_disk=True."""
    # Set up mocks
    mock_chunk_document.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
    mock_validate.return_value = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Create a test document
    doc = {"id": "test-doc", "content": "test content"}
    
    # Call with save_to_disk=True
    result = chunk_document_batch([doc], save_to_disk=True, output_dir="/test/output")
    
    # Verify directory was created
    mock_makedirs.assert_called_once_with("/test/output", exist_ok=True)
    
    # Verify file was opened for writing
    mock_open.assert_called_once()
    assert "test-doc" in str(mock_open.call_args[0][0])
    
    # Verify content was written
    mock_file.write.assert_called_once()


# Tests for process_documents_batch function
@patch('src.chunking.text_chunkers.chonky_batch.chunk_document')
def test_process_documents_batch_serial(mock_chunk_document):
    """Test process_documents_batch with serial processing."""
    # Set up mock
    mock_chunk_document.side_effect = lambda doc, **kwargs: {
        "id": doc["id"],
        "content": doc["content"],
        "chunks": [{"id": f"chunk-{doc['id']}"}]
    }
    
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1"},
        {"id": "doc2", "content": "content 2"}
    ]
    
    # Call with parallel=False
    result = process_documents_batch(docs, parallel=False)
    
    # Verify results
    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[1]["id"] == "doc2"
    assert "chunks" in result[0]
    assert "chunks" in result[1]
    
    # Verify chunk_document was called for each document
    assert mock_chunk_document.call_count == 2


@patch('src.chunking.text_chunkers.chonky_batch.ThreadPool')
@patch('src.chunking.text_chunkers.chonky_batch.chunk_document')
def test_process_documents_batch_parallel(mock_chunk_document, mock_pool):
    """Test process_documents_batch with parallel processing."""
    # Set up mocks
    pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = pool_instance
    
    # Configure pool to return processed documents
    processed_docs = [
        {"id": "doc1", "content": "content 1", "chunks": [{"id": "chunk1"}]},
        {"id": "doc2", "content": "content 2", "chunks": [{"id": "chunk2"}]}
    ]
    pool_instance.map.return_value = processed_docs
    
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1"},
        {"id": "doc2", "content": "content 2"}
    ]
    
    # Call with parallel=True
    result = process_documents_batch(docs, parallel=True, num_workers=2)
    
    # Verify ThreadPool was used with correct num_workers
    mock_pool.assert_called_once_with(processes=2)
    
    # Verify results
    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[1]["id"] == "doc2"
    assert "chunks" in result[0]
    assert "chunks" in result[1]


# Tests for chunk_documents function
@patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch')
def test_chunk_documents_batching(mock_process_documents_batch):
    """Test chunk_documents batches documents correctly."""
    # Set up mock to return the input documents
    mock_process_documents_batch.side_effect = lambda docs, **kwargs: docs
    
    # Create test documents
    docs = [{"id": f"doc{i}"} for i in range(5)]
    
    # Call with batch_size=2
    result = chunk_documents(docs, batch_size=2, progress=False)
    
    # Verify process_documents_batch was called for each batch
    assert mock_process_documents_batch.call_count == 3  # 5 docs with batch_size=2 -> 3 batches
    
    # Verify batch sizes
    calls = mock_process_documents_batch.call_args_list
    assert len(calls[0][0][0]) == 2  # First batch: 2 docs
    assert len(calls[1][0][0]) == 2  # Second batch: 2 docs
    assert len(calls[2][0][0]) == 1  # Third batch: 1 doc
    
    # Verify all documents were returned
    assert len(result) == 5


@patch('src.chunking.text_chunkers.chonky_batch.process_documents_batch')
@patch('src.chunking.text_chunkers.chonky_batch.tqdm')
def test_chunk_documents_with_progress(mock_tqdm, mock_process_documents_batch):
    """Test chunk_documents with progress reporting."""
    # Set up mocks
    mock_process_documents_batch.side_effect = lambda docs, **kwargs: docs
    mock_tqdm.return_value = range(3)  # Simulate tqdm returning a range
    
    # Create test documents
    docs = [{"id": f"doc{i}"} for i in range(5)]
    
    # Call with progress=True
    result = chunk_documents(docs, batch_size=2, progress=True)
    
    # Verify tqdm was used
    mock_tqdm.assert_called_once()
    
    # Verify all documents were returned
    assert len(result) == 5


# Tests for chunk_text_batch function
@patch('src.chunking.text_chunkers.chonky_batch.chunk_text')
def test_chunk_text_batch_serial_python_format(mock_chunk_text):
    """Test chunk_text_batch with serial processing and Python output format."""
    # Set up mock
    mock_chunk_text.return_value = [{"id": "chunk1", "content": "chunk content"}]
    
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1", "path": "/path/1", "type": "text"},
        {"id": "doc2", "content": "content 2", "path": "/path/2", "type": "text"}
    ]
    
    # Call with parallel=False and output_format="python"
    result = chunk_text_batch(docs, parallel=False, output_format="python")
    
    # Verify results
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
    assert result[0][0]["id"] == "chunk1"
    
    # Verify chunk_text was called for each document
    assert mock_chunk_text.call_count == 2


@patch('src.chunking.text_chunkers.chonky_batch.chunk_text')
def test_chunk_text_batch_serial_json_format(mock_chunk_text):
    """Test chunk_text_batch with serial processing and JSON output format."""
    # Set up mock
    mock_chunk_text.return_value = json.dumps([{"id": "chunk1", "content": "chunk content"}])
    
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1", "path": "/path/1", "type": "text"},
        {"id": "doc2", "content": "content 2", "path": "/path/2", "type": "text"}
    ]
    
    # Call with parallel=False and output_format="json"
    result = chunk_text_batch(docs, parallel=False, output_format="json")
    
    # Verify results
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert json.loads(result[0])[0]["id"] == "chunk1"
    
    # Verify chunk_text was called for each document
    assert mock_chunk_text.call_count == 2


@patch('src.chunking.text_chunkers.chonky_batch.ThreadPool')
def test_chunk_text_batch_parallel(mock_pool):
    """Test chunk_text_batch with parallel processing."""
    # Set up mocks
    pool_instance = MagicMock()
    mock_pool.return_value.__enter__.return_value = pool_instance
    
    # Configure pool to return processed chunks
    processed_chunks = [
        [{"id": "chunk1-1"}, {"id": "chunk1-2"}],
        [{"id": "chunk2-1"}]
    ]
    pool_instance.map.return_value = processed_chunks
    
    # Create test documents
    docs = [
        {"id": "doc1", "content": "content 1", "path": "/path/1", "type": "text"},
        {"id": "doc2", "content": "content 2", "path": "/path/2", "type": "text"}
    ]
    
    # Call with parallel=True
    result = chunk_text_batch(docs, parallel=True, num_workers=2)
    
    # Verify ThreadPool was used with correct num_workers
    mock_pool.assert_called_once_with(processes=2)
    
    # Verify results
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 1
    assert result[0][0]["id"] == "chunk1-1"
    assert result[1][0]["id"] == "chunk2-1"
