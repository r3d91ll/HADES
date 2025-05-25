"""
Integration test for Pydantic model integration in the document processing pipeline.

This test validates that the document processing pipeline correctly uses Pydantic models
for in-memory operations and avoids unnecessary disk operations.
"""

import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import the updated modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.docproc.schemas.base import BaseDocument
from src.schema.document_schema import DocumentSchema, ChunkMetadata
from src.schema.validation import validate_document, ValidationStage

# Import the updated document processing and chunking modules
from src.docproc.core_updated import process_document
from src.chunking.text_chunkers.chonky_chunker_updated import chunk_text, chunk_document


class TestPydanticIntegration(unittest.TestCase):
    """Test Pydantic model integration in the document processing pipeline."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        # Define paths
        self.data_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/data")
        self.test_markdown_file = self.data_dir / "sample_text.txt"
        
        # Skip tests if test files don't exist
        if not self.test_markdown_file.exists():
            self.skipTest(f"Test file not found: {self.test_markdown_file}")
    
    def test_process_document_returns_pydantic(self) -> None:
        """Test that process_document returns a Pydantic model instance."""
        # Process the document
        result = process_document(self.test_markdown_file)
        
        # Check that the result is a Pydantic model instance
        self.assertIsInstance(result, BaseDocument, 
                             "process_document should return a Pydantic model instance")
        
        # Check that we can access model attributes
        self.assertIsNotNone(result.id, "Document should have an ID")
        self.assertIsNotNone(result.content, "Document should have content")
    
    def test_chunk_text_with_pydantic_input(self) -> None:
        """Test that chunk_text can accept a Pydantic model instance as input."""
        # Process the document to get a Pydantic model
        doc_model = process_document(self.test_markdown_file)
        
        # Use the model directly with chunk_text
        chunks = chunk_text(doc_model, max_tokens=1024, output_format="python")
        
        # Check that we got chunks
        self.assertIsInstance(chunks, list, "chunk_text should return a list of chunks")
        self.assertGreater(len(chunks), 0, "Should have generated at least one chunk")
        
        # Check chunk structure
        self.assertIn("content", chunks[0], "Chunk should have content")
        self.assertIn("symbol_type", chunks[0], "Chunk should have a symbol type")
    
    def test_chunk_text_returns_pydantic_chunks(self) -> None:
        """Test that chunk_text can return Pydantic ChunkMetadata instances."""
        # Process the document to get a Pydantic model
        doc_model = process_document(self.test_markdown_file)
        
        # Request Pydantic output format
        chunks = chunk_text(doc_model, max_tokens=1024, output_format="pydantic")
        
        # Check that we got Pydantic ChunkMetadata instances
        self.assertIsInstance(chunks, list, "chunk_text should return a list")
        self.assertGreater(len(chunks), 0, "Should have generated at least one chunk")
        self.assertIsInstance(chunks[0], ChunkMetadata, 
                             "chunk_text should return ChunkMetadata instances with output_format='pydantic'")
        
        # Check that we can access ChunkMetadata attributes
        self.assertIsNotNone(chunks[0].chunk_index, "Chunk should have an index")
        self.assertIsNotNone(chunks[0].metadata.content, "Chunk should have content")
    
    def test_chunk_document_end_to_end(self) -> None:
        """Test the end-to-end document chunking process with Pydantic models."""
        # Process the document to get a Pydantic model
        doc_model = process_document(self.test_markdown_file)
        
        # Chunk the document and get back a Pydantic model with chunks
        chunked_doc = chunk_document(
            doc_model, 
            max_tokens=1024, 
            return_pydantic=True,
            save_to_disk=False  # Important: we're testing in-memory processing
        )
        
        # Check that we got a Pydantic DocumentSchema instance
        self.assertIsInstance(chunked_doc, DocumentSchema, 
                             "chunk_document should return a DocumentSchema instance")
        
        # Check that the document has chunks
        self.assertIsNotNone(chunked_doc.chunks, "Document should have chunks")
        self.assertGreater(len(chunked_doc.chunks), 0, "Document should have at least one chunk")
        
        # Check that the chunks are ChunkMetadata instances
        self.assertIsInstance(chunked_doc.chunks[0], ChunkMetadata, 
                             "Chunks should be ChunkMetadata instances")
        
        # Check chunk content
        self.assertIsNotNone(chunked_doc.chunks[0].metadata.content, 
                            "Chunk should have content")
        self.assertGreater(len(chunked_doc.chunks[0].metadata.content), 0, 
                          "Chunk content should not be empty")


if __name__ == "__main__":
    unittest.main()
