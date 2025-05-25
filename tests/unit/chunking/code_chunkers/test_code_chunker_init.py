"""Unit tests for the code chunker module.

This module contains tests for the code chunker initialization and dispatching functionality.
"""

from __future__ import annotations

import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import the module under test
import src.chunking.code_chunkers
from src.chunking.code_chunkers import chunk_code, _LANG_DISPATCH


class TestCodeChunkerInit:
    """Test suite for the code chunker initialization and dispatching functionality."""
    
    def test_python_code_dispatch(self):
        """Test that Python code documents are properly dispatched."""
        # Create a Python document
        document = {
            "id": "test-python",
            "type": "python",
            "content": "def test():\n    pass",
            "path": "/path/to/test.py"
        }
        
        # Need to patch _LANG_DISPATCH directly since that's what chunk_code uses
        original_dispatch = src.chunking.code_chunkers._LANG_DISPATCH.copy()
        try:
            # Create a mock for the Python chunker
            mock_chunker = MagicMock()
            mock_chunker.return_value = [{"id": "mock-chunk"}]
            
            # Replace the real chunker with our mock
            src.chunking.code_chunkers._LANG_DISPATCH = {"python": mock_chunker}
            
            # Call chunk_code which should dispatch to our mock
            result = chunk_code(document)
            
            # Verify it called our mock chunker
            mock_chunker.assert_called_once()
            assert result == [{"id": "mock-chunk"}]
            # Verify the right parameters were passed
            mock_chunker.assert_called_once_with(document, max_tokens=2048, output_format="python")
        finally:
            # Restore the original dispatcher
            src.chunking.code_chunkers._LANG_DISPATCH = original_dispatch
    
    def test_language_type_variations(self):
        """Test that different representations of the language type are handled."""
        # Need to patch _LANG_DISPATCH directly
        original_dispatch = src.chunking.code_chunkers._LANG_DISPATCH.copy()
        try:
            # Create a mock for the Python chunker
            mock_chunker = MagicMock()
            mock_chunker.return_value = [{"id": "mock-chunk"}]
            
            # Replace the real chunker with our mock
            src.chunking.code_chunkers._LANG_DISPATCH = {"python": mock_chunker}
            
            # Test with 'language' field instead of 'type'
            document = {
                "id": "test-python-alt",
                "language": "python",
                "content": "def test():\n    pass",
                "path": "/path/to/test.py"
            }
            
            result = chunk_code(document)
            assert mock_chunker.call_count == 1
            mock_chunker.reset_mock()
            
            # Test with uppercase language type
            document = {
                "id": "test-python-case",
                "type": "PYTHON",
                "content": "def test():\n    pass",
                "path": "/path/to/test.py"
            }
            
            result = chunk_code(document)
            assert mock_chunker.call_count == 1
        finally:
            # Restore the original dispatcher
            src.chunking.code_chunkers._LANG_DISPATCH = original_dispatch
    
    def test_unsupported_language(self):
        """Test error handling for unsupported languages."""
        # Create document with unsupported language
        document = {
            "id": "test-unsupported",
            "type": "unsupported_lang",
            "content": "function test() { }",
            "path": "/path/to/test.unsupported"
        }
        
        # Should raise ValueError for unsupported language
        with pytest.raises(ValueError) as excinfo:
            result = chunk_code(document)
        
        # Check error message
        assert "No chunker registered for language" in str(excinfo.value)
    
    def test_unknown_language(self):
        """Test handling of documents with unknown language."""
        # Create document with no language specified
        document = {
            "id": "test-unknown",
            "content": "Some content",
            "path": "/path/to/test.txt"
        }
        
        # Should raise ValueError for unknown language
        with pytest.raises(ValueError) as excinfo:
            result = chunk_code(document)
        
        # Check error message
        assert "No chunker registered for language: unknown" in str(excinfo.value)
    
    def test_output_format_passing(self):
        """Test that output_format is correctly passed to the chunker."""
        document = {
            "id": "test-output-format",
            "type": "python",
            "content": "def test():\n    pass",
            "path": "/path/to/test.py"
        }
        
        # Need to patch _LANG_DISPATCH directly
        original_dispatch = src.chunking.code_chunkers._LANG_DISPATCH.copy()
        try:
            # Create a mock for the Python chunker
            mock_chunker = MagicMock()
            mock_chunker.return_value = "JSON_STRING"
            
            # Replace the real chunker with our mock
            src.chunking.code_chunkers._LANG_DISPATCH = {"python": mock_chunker}
            
            # Call with json output format
            result = chunk_code(document, output_format="json")
            
            # Verify output_format was passed correctly
            mock_chunker.assert_called_once_with(document, max_tokens=2048, output_format="json")
            assert result == "JSON_STRING"
        finally:
            # Restore the original dispatcher
            src.chunking.code_chunkers._LANG_DISPATCH = original_dispatch
    
    def test_max_tokens_passing(self):
        """Test that max_tokens is correctly passed to the chunker."""
        document = {
            "id": "test-max-tokens",
            "type": "python",
            "content": "def test():\n    pass",
            "path": "/path/to/test.py"
        }
        
        # Need to patch _LANG_DISPATCH directly
        original_dispatch = src.chunking.code_chunkers._LANG_DISPATCH.copy()
        try:
            # Create a mock for the Python chunker
            mock_chunker = MagicMock()
            mock_chunker.return_value = [{"id": "mock-chunk"}]
            
            # Replace the real chunker with our mock
            src.chunking.code_chunkers._LANG_DISPATCH = {"python": mock_chunker}
            
            # Call with custom max_tokens
            result = chunk_code(document, max_tokens=1024)
            
            # Verify max_tokens was passed correctly
            mock_chunker.assert_called_once_with(document, max_tokens=1024, output_format="python")
        finally:
            # Restore the original dispatcher
            src.chunking.code_chunkers._LANG_DISPATCH = original_dispatch
