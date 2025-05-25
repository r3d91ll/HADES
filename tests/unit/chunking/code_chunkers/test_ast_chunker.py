"""Unit tests for the AST-based code chunker.

This module contains comprehensive tests for the AST-based code chunker,
ensuring that code files are properly parsed and chunked according to their
logical structure (functions, classes, etc.).
"""

from __future__ import annotations

import os
import ast
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

from src.chunking.code_chunkers.ast_chunker import (
    chunk_python_code,
    extract_chunk_content,
    estimate_tokens,
    create_chunk_id
)
from tests.unit.common_fixtures import (
    sample_code_document,
    create_expected_chunks,
    SAMPLE_CODE_CONTENT
)


class TestASTChunker:
    """Test suite for the AST-based code chunker."""
    
    def test_estimate_tokens(self):
        """Test the token estimation function."""
        # Test with empty string
        assert estimate_tokens("") == 0
        
        # Test with short code snippet
        code = "def test_func():\n    return 42"
        estimated = estimate_tokens(code)
        assert estimated > 0
        assert isinstance(estimated, int)
    
    def test_create_chunk_id(self):
        """Test creation of stable chunk IDs."""
        # Test with different inputs
        id1 = create_chunk_id("file.py", "function", "test_func", 1, 10)
        id2 = create_chunk_id("file.py", "function", "test_func", 1, 10)
        id3 = create_chunk_id("file.py", "class", "TestClass", 1, 10)
        
        # Same inputs should produce same IDs
        assert id1 == id2
        # Different inputs should produce different IDs
        assert id1 != id3
        # IDs should be strings
        assert isinstance(id1, str)
    
    def test_extract_chunk_content(self):
        """Test extracting content between line numbers."""
        # Multi-line source with different indentation
        source = "line1\nline2\n  line3\nline4"
        
        # Extract different ranges
        content1 = extract_chunk_content(source, 1, 2)  # First two lines
        content2 = extract_chunk_content(source, 3, 3)  # Just line 3
        content3 = extract_chunk_content(source, 1, 4)  # All lines
        
        # Check content is extracted correctly
        assert content1 == "line1\nline2"
        assert content2 == "  line3"
        assert content3 == source
    
    def test_chunk_python_code(self):
        """Test the chunk_python_code function."""
        # Create a test Python document with symbol table
        document = {
            "id": "test_doc",
            "content": "def test_func():\n    return 42\n\nclass TestClass:\n    def method(self):\n        pass",
            "path": "test/file.py",
            "type": "python",
            "language": "python"
        }
        
        # Test with Python output format
        result = chunk_python_code(document, max_tokens=1024, output_format="python")
        
        # Check the result
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each chunk should have the expected structure
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "id" in chunk
            assert "content" in chunk
            # Check that essential fields are present
            assert "line_start" in chunk
            assert "line_end" in chunk
            assert isinstance(chunk["line_start"], int)
            assert isinstance(chunk["line_end"], int)
        
        # Test with JSON output format
        json_result = chunk_python_code(document, max_tokens=1024, output_format="json")
        assert isinstance(json_result, str)
    
    def test_chunk_python_code_with_ast_errors(self):
        """Test chunking a Python document with AST errors."""
        # Create a test document with invalid Python code
        document = {
            "id": "test_doc",
            "content": "def broken_function(:",  # Invalid syntax
            "path": "test/file.py",
            "type": "python",
            "language": "python"
        }
        
        # Should handle the error gracefully
        result = chunk_python_code(document, max_tokens=1024)
        
        # Should still return a list
        assert isinstance(result, list)
        
        # Should have at least one chunk containing the broken code
        assert len(result) > 0
        
        # The chunk should have the basic required fields
        assert "id" in result[0]
        assert "content" in result[0]
        
        # Check that the content contains the invalid code
        assert "def broken_function(" in result[0]["content"]
    
    def test_chunk_python_code_with_max_tokens(self):
        """Test chunking a Python document with different max_tokens."""
        # Create a document with lengthy Python code
        long_code = "def long_function():\n" + "    x = 1\n" * 500
        document = {
            "id": "test_doc",
            "content": long_code,
            "path": "test/file.py",
            "type": "python",
            "language": "python"
        }
        
        # Test with different max_tokens values
        chunks_small = chunk_python_code(document, max_tokens=100)
        chunks_large = chunk_python_code(document, max_tokens=1000)
        
        # Smaller max_tokens should result in more chunks
        assert len(chunks_small) >= len(chunks_large)
        
        # Verify chunk structure for both sets of chunks
        for chunk_set in [chunks_small, chunks_large]:
            for chunk in chunk_set:
                assert "id" in chunk
                assert "content" in chunk
                
                # Each chunk should have line info
                assert "line_start" in chunk
                assert "line_end" in chunk
                assert isinstance(chunk["line_start"], int)
                assert isinstance(chunk["line_end"], int)
    
    def test_empty_document_handling(self):
        """Test handling of empty documents in Python chunker."""
        # Create a document with empty content
        empty_doc = {
            "id": "empty-doc",
            "content": "",
            "path": "/path/to/empty.py",
            "type": "python",
            "language": "python"
        }
        
        # Should handle the empty document
        result = chunk_python_code(empty_doc)
        
        # For empty documents, the implementation returns an empty list
        assert isinstance(result, list)
        # The implementation returns an empty list for empty documents
        # as specified in the implementation at line 117-119
    
    def test_json_output_format(self):
        """Test JSON output format for Python chunker."""
        # Create a simple document
        doc = {
            "id": "json-test",
            "content": "def test_func():\n    return 'Hello, world!'",
            "path": "/path/to/test.py",
            "type": "python",
            "language": "python"
        }
        
        # Get output in JSON format
        result = chunk_python_code(doc, output_format="json")
        
        # Should be a JSON string
        assert isinstance(result, str)
        
        # Should be parseable as JSON
        import json
        parsed = json.loads(result)
        assert isinstance(parsed, list)
