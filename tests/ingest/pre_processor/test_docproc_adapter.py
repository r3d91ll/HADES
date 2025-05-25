"""
Tests for the DocProcAdapter class.

This module provides test coverage for the adapter that connects
the pre_processor module with the new docproc module.
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path

from src.ingest.pre_processor.base_pre_processor import DocProcAdapter


class TestDocProcAdapter:
    """Tests for the DocProcAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create a DocProcAdapter instance."""
        return DocProcAdapter()
    
    @pytest.fixture
    def format_adapter(self):
        """Create a DocProcAdapter with a specific format override."""
        return DocProcAdapter(format_override='python')
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(b'def test_function():\n    """Test docstring."""\n    return "Hello, World!"')
            path = f.name
        yield path
        os.unlink(path)
    
    def test_initialization(self, adapter, format_adapter):
        """Test that the DocProcAdapter initializes correctly."""
        assert adapter.format_override is None
        assert format_adapter.format_override == 'python'
        assert hasattr(adapter, 'process_file')
        assert callable(adapter.process_file)
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file_calls_docproc(self, mock_process_document, adapter, temp_file):
        """Test that process_file calls the docproc process_document function."""
        # Set up mock return value
        mock_result = {
            'id': 'test-id',
            'source': temp_file,
            'content': 'Test content',
            'format': 'python',
            'metadata': {'test': 'metadata'},
            'entities': [
                {'type': 'function', 'name': 'test_function'},
                {'type': 'class', 'name': 'TestClass'}
            ]
        }
        mock_process_document.return_value = mock_result
        
        # Call the adapter
        result = adapter.process_file(temp_file)
        
        # Verify the docproc function was called
        mock_process_document.assert_called_once_with(temp_file)
        
        # Verify the result was converted to the legacy format
        assert result['id'] == 'test-id'
        assert result['path'] == temp_file
        assert result['content'] == 'Test content'
        assert result['type'] == 'python'
        assert result['metadata'] == {'test': 'metadata'}
        assert 'functions' in result
        assert result['functions'][0]['name'] == 'test_function'
        assert 'classes' in result
        assert result['classes'][0]['name'] == 'TestClass'
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file_with_symbols(self, mock_process_document, adapter, temp_file):
        """Test processing a file with symbols."""
        # Set up mock return value with symbols instead of entities
        mock_result = {
            'id': 'test-id',
            'source': temp_file,
            'content': 'Test content',
            'format': 'python',
            'metadata': {'test': 'metadata'},
            'symbols': [
                {'type': 'function', 'name': 'test_function'},
                {'type': 'class', 'name': 'TestClass'},
                {'type': 'import', 'name': 'os'}
            ]
        }
        mock_process_document.return_value = mock_result
        
        # Call the adapter
        result = adapter.process_file(temp_file)
        
        # Verify the symbols were mapped to the legacy format
        assert 'functions' in result
        assert result['functions'][0]['name'] == 'test_function'
        assert 'classes' in result
        assert result['classes'][0]['name'] == 'TestClass'
        assert 'imports' in result
        assert result['imports'][0]['name'] == 'os'
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file_with_error(self, mock_process_document, adapter, temp_file):
        """Test that errors are handled properly."""
        # Set up mock to raise an exception
        mock_process_document.side_effect = ValueError("Test error")
        
        # Call the adapter
        result = adapter.process_file(temp_file)
        
        # Verify error handling
        assert result is None
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_batch_processing(self, mock_process_document, adapter, temp_file):
        """Test batch processing through the adapter."""
        # Set up mock return value
        mock_result = {
            'id': 'test-id',
            'source': temp_file,
            'content': 'Test content',
            'format': 'python',
            'metadata': {'test': 'metadata'}
        }
        mock_process_document.return_value = mock_result
        
        # Process a batch with one file
        results = adapter.process_batch([temp_file])
        
        # Verify results
        assert len(results) == 1
        assert results[0]['path'] == temp_file
        assert mock_process_document.call_count == 1
