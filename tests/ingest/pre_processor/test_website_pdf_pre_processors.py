"""
Tests for the WebsitePreProcessor and PDFPreProcessor classes.

This module provides test coverage for the website and PDF pre-processors
that use the new docproc module adapters.
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path

from src.ingest.pre_processor.website_pdf_pre_processors import WebsitePreProcessor, PDFPreProcessor


class TestWebsitePreProcessor:
    """Tests for the WebsitePreProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a WebsitePreProcessor instance."""
        return WebsitePreProcessor()
    
    @pytest.fixture
    def temp_html_file(self):
        """Create a temporary HTML file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            f.write(b'<html><head><title>Test Page</title></head><body><h1>Hello, World!</h1></body></html>')
            path = f.name
        yield path
        os.unlink(path)
    
    def test_initialization(self, processor):
        """Test that the WebsitePreProcessor initializes correctly."""
        assert processor.format_override == 'html'
        assert hasattr(processor, 'process_file')
        assert callable(processor.process_file)
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file(self, mock_process_document, processor, temp_html_file):
        """Test that process_file calls the docproc process_document function and sets type to html."""
        # Set up mock return value
        mock_result = {
            'id': 'test-id',
            'source': temp_html_file,
            'content': '<html>...</html>',
            'format': 'html',
            'metadata': {'title': 'Test Page'}
        }
        mock_process_document.return_value = mock_result
        
        # Call the processor
        result = processor.process_file(temp_html_file)
        
        # Verify the docproc function was called
        mock_process_document.assert_called_once_with(str(temp_html_file))
        
        # Verify the type is set correctly
        assert result['type'] == 'html'
        assert result['path'] == temp_html_file
        assert result['metadata'] == {'title': 'Test Page'}
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file_error(self, mock_process_document, processor, temp_html_file):
        """Test error handling in process_file."""
        # Set up mock to return None (indicating an error in the adapter)
        mock_process_document.return_value = None
        
        # Call the processor
        result = processor.process_file(temp_html_file)
        
        # Verify result is None
        assert result is None


class TestPDFPreProcessor:
    """Tests for the PDFPreProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a PDFPreProcessor instance."""
        return PDFPreProcessor()
    
    @pytest.fixture
    def temp_pdf_path(self):
        """Return a path for a mock PDF file."""
        # Note: We don't actually create a PDF file as we mock the process_document call
        return "/mock/path/test.pdf"
    
    def test_initialization(self, processor):
        """Test that the PDFPreProcessor initializes correctly."""
        assert processor.format_override == 'pdf'
        assert hasattr(processor, 'process_file')
        assert callable(processor.process_file)
    
    @patch('src.ingest.pre_processor.base_pre_processor.process_document')
    def test_process_file(self, mock_process_document, processor, temp_pdf_path):
        """Test that process_file calls the docproc process_document function and sets type to pdf."""
        # Set up mock return value
        mock_result = {
            'id': 'test-id',
            'source': temp_pdf_path,
            'content': 'PDF content...',
            'format': 'pdf',
            'metadata': {'title': 'Test PDF'}
        }
        mock_process_document.return_value = mock_result
        
        # Call the processor
        result = processor.process_file(temp_pdf_path)
        
        # Verify the docproc function was called
        mock_process_document.assert_called_once_with(str(temp_pdf_path))
        
        # Verify the type is set correctly
        assert result['type'] == 'pdf'
        assert result['path'] == temp_pdf_path
        assert result['metadata'] == {'title': 'Test PDF'}
