"""Unit tests for the CPU-optimized chunker module.

This module tests the functionality of the CPU-optimized chunker, including:
- Parallel processing for large documents
- Multi-threaded segment handling
- API compatibility with the standard chunker
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.chunking.text_chunkers.cpu_chunker import (
    chunk_text_cpu,
    chunk_document_cpu,
    process_content_with_cpu,
    _process_segment
)
from src.chunking.text_chunkers.chonky_chunker import (
    ParagraphSplitter
)
from src.schema.document_schema import DocumentSchema, DocumentType


class TestCPUChunker(unittest.TestCase):
    """Test cases for the CPU-optimized chunker module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample document
        self.sample_document = {
            "content": "This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph should be chunked separately.",
            "path": "/path/to/test/document.txt",
            "type": "text",
            "id": "test_doc_001"
        }
        
        # Create a larger document for testing segment processing
        self.large_document = {
            "content": "\n\n".join([f"Paragraph {i}. " + "This is test content. " * 100 for i in range(20)]),
            "path": "/path/to/test/large_document.txt",
            "type": "text",
            "id": "test_doc_002"
        }
        
        # Create a sample document as a Pydantic model
        self.sample_document_model = DocumentSchema(
            content="This is a test document as a Pydantic model.\n\nIt has multiple paragraphs.\n\nEach paragraph should be chunked separately.",
            source="/path/to/test/document_model.txt",
            document_type=DocumentType.TEXT,
            id="test_doc_003"
        )

    @patch('src.chunking.text_chunkers.cpu_chunker.ParagraphSplitter')
    def test_chunk_text_cpu_basic(self, mock_splitter_class):
        """Test basic CPU chunking functionality."""
        # Set up mock paragraph splitter
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            "This is a test document.",
            "It has multiple paragraphs.",
            "Each paragraph should be chunked separately."
        ]
        mock_splitter_class.return_value = mock_splitter
        
        # Call chunk_text_cpu
        result = chunk_text_cpu(
            content=self.sample_document["content"],
            doc_id=self.sample_document["id"],
            path=self.sample_document["path"],
            doc_type=self.sample_document["type"]
        )
        
        # Verify the result
        self.assertEqual(result["id"], self.sample_document["id"])
        self.assertEqual(result["content"], self.sample_document["content"])
        self.assertEqual(result["path"], self.sample_document["path"])
        self.assertEqual(result["type"], self.sample_document["type"])
        self.assertEqual(len(result["chunks"]), 3)
        
        # Check the chunks
        self.assertEqual(result["chunks"][0]["content"], "This is a test document.")
        self.assertEqual(result["chunks"][1]["content"], "It has multiple paragraphs.")
        self.assertEqual(result["chunks"][2]["content"], "Each paragraph should be chunked separately.")
        
        # Verify splitter was called correctly
        mock_splitter_class.assert_called_once_with(
            model_id="mirth/chonky_modernbert_large_1",
            device="cpu",
            use_model_engine=False
        )
        mock_splitter.split_text.assert_called_once_with(self.sample_document["content"])
    
    @patch('src.chunking.text_chunkers.cpu_chunker.ParagraphSplitter')
    def test_chunk_text_cpu_empty_content(self, mock_splitter_class):
        """Test CPU chunking with empty content."""
        # Call chunk_text_cpu with empty content
        result = chunk_text_cpu(
            content="",
            doc_id="empty_doc",
            path="/path/to/empty.txt",
            doc_type="text"
        )
        
        # Verify the result
        self.assertEqual(result["id"], "empty_doc")
        self.assertEqual(result["content"], "")
        self.assertEqual(len(result["chunks"]), 0)
        
        # Verify splitter was not called
        mock_splitter_class.assert_not_called()
    
    @patch('src.chunking.text_chunkers.cpu_chunker.ThreadPool')
    @patch('src.chunking.text_chunkers.cpu_chunker.ParagraphSplitter')
    def test_chunk_text_cpu_large_document(self, mock_splitter_class, mock_thread_pool):
        """Test CPU chunking with a large document that triggers parallel processing."""
        # Set up mock paragraph splitter
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["Mock paragraph"]
        mock_splitter_class.return_value = mock_splitter
        
        # Set up mock thread pool
        mock_pool = MagicMock()
        mock_pool.map.return_value = [["Segment 1 paragraph"], ["Segment 2 paragraph"]]
        mock_thread_pool.return_value.__enter__.return_value = mock_pool
        
        # Create a large document content (>10k chars)
        large_content = "Test content. " * 1000  # ~12k chars
        
        # Call chunk_text_cpu
        result = chunk_text_cpu(
            content=large_content,
            doc_id="large_doc",
            path="/path/to/large.txt",
            doc_type="text",
            num_workers=2
        )
        
        # Verify the result
        self.assertEqual(result["id"], "large_doc")
        self.assertEqual(result["content"], large_content)
        self.assertEqual(len(result["chunks"]), 2)  # One from each segment
        
        # Verify thread pool was called
        mock_thread_pool.assert_called_once()
        self.assertTrue(mock_pool.map.called)
    
    @patch('src.chunking.text_chunkers.cpu_chunker.ParagraphSplitter')
    def test_process_segment(self, mock_splitter_class):
        """Test processing a single text segment."""
        # Set up mock paragraph splitter
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["Paragraph 1", "Paragraph 2"]
        
        # Create a segment
        segment_data = (1, 2, "This is segment 1 content.\n\nWith multiple paragraphs.")
        
        # Process segment
        result = _process_segment(segment_data, mock_splitter)
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Paragraph 1")
        self.assertEqual(result[1], "Paragraph 2")
        
        # Verify splitter was called correctly
        mock_splitter.split_text.assert_called_once_with(segment_data[2])
    
    @patch('src.chunking.text_chunkers.cpu_chunker.chunk_text_cpu')
    def test_chunk_document_cpu_with_dict(self, mock_chunk_text):
        """Test chunking a document with CPU optimization using a dict input."""
        # Set up mock chunk_text_cpu
        mock_chunk_text.return_value = {
            "id": self.sample_document["id"],
            "content": self.sample_document["content"],
            "path": self.sample_document["path"],
            "type": self.sample_document["type"],
            "chunks": [{"id": "chunk1", "content": "Chunk 1 content"}]
        }
        
        # Call chunk_document_cpu
        result = chunk_document_cpu(
            document=self.sample_document,
            return_pydantic=False
        )
        
        # Verify the result
        self.assertEqual(result["id"], self.sample_document["id"])
        self.assertEqual(len(result["chunks"]), 1)
        
        # Verify chunk_text_cpu was called correctly
        mock_chunk_text.assert_called_once_with(
            content=self.sample_document["content"],
            doc_id=self.sample_document["id"],
            path=self.sample_document["path"],
            doc_type=self.sample_document["type"],
            max_tokens=2048,
            output_format="dict",
            model_id="mirth/chonky_modernbert_large_1",
            num_workers=4
        )
    
    @patch('src.chunking.text_chunkers.cpu_chunker.chunk_text_cpu')
    def test_chunk_document_cpu_with_pydantic(self, mock_chunk_text):
        """Test chunking a document with CPU optimization using a Pydantic model input."""
        # Set up mock chunk_text_cpu
        mock_chunk_text.return_value = {
            "id": self.sample_document_model.id,
            "content": self.sample_document_model.content,
            "path": self.sample_document_model.source,
            "type": self.sample_document_model.document_type,
            "chunks": [{"id": "chunk1", "content": "Chunk 1 content"}]
        }
        
        # Call chunk_document_cpu
        result = chunk_document_cpu(
            document=self.sample_document_model,
            return_pydantic=True
        )
        
        # Verify the result
        self.assertEqual(result["id"], self.sample_document_model.id)
        self.assertEqual(len(result["chunks"]), 1)
        
        # Verify result is a dictionary (since we're using mock that returns a dict)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
