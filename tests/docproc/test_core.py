"""
Tests for document processing core module.

Tests for the functionality in src.docproc.core.
"""

import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, Any, Optional

import src.docproc.core as docproc_core
from src.docproc.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the mock adapter."""
        self.options = options or {}
        self.process_called = False
        self.process_text_called = False
        self.extract_entities_called = False
        self.extract_metadata_called = False
        # Tracking for removed conversion methods removed
        
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation of process method."""
        self.process_called = True
        options = options or {}
        return {
            'content': f"Processed content from {file_path}",
            'metadata': {
                'format': 'mock',
                'file_path': str(file_path),
                'options': options
            }
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation of process_text method."""
        self.process_text_called = True
        options = options or {}
        return {
            'content': f"Processed text: {text[:20]}...",
            'metadata': {
                'format': 'mock',
                'options': options
            }
        }
    
    def extract_entities(self, content: Any) -> list:
        """Mock implementation of extract_entities method."""
        self.extract_entities_called = True
        return [{'type': 'mock_entity', 'content': str(content)[:30]}]
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Mock implementation of extract_metadata method."""
        self.extract_metadata_called = True
        return {'format': 'mock', 'length': len(str(content))}


class TestCore(unittest.TestCase):
    """Test cases for the document processing core module."""

    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create test files
        self.test_text_file = self.temp_dir_path / "test.txt"
        with open(self.test_text_file, 'w') as f:
            f.write("This is a test text file.")
        
        self.test_pdf_file = self.temp_dir_path / "test.pdf"
        with open(self.test_pdf_file, 'w') as f:
            f.write("This is a simulated PDF file.")
        
        self.test_json_file = self.temp_dir_path / "test.json"
        with open(self.test_json_file, 'w') as f:
            f.write('{"name": "Test JSON", "value": 42}')
    
    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_string_path(self, mock_get_adapter):
        """Test processing a document with a string path."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with string path
        result = docproc_core.process_document(str(self.test_text_file))
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['format'], 'mock')
        self.assertEqual(result['metadata']['file_path'], str(self.test_text_file))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_path_object(self, mock_get_adapter):
        """Test processing a document with a Path object."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with Path object
        result = docproc_core.process_document(self.test_pdf_file)
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['format'], 'mock')
        self.assertEqual(result['metadata']['file_path'], str(self.test_pdf_file))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_with_options(self, mock_get_adapter):
        """Test processing a document with options."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with options
        options = {'clean': True, 'extract_metadata': True}
        result = docproc_core.process_document(self.test_json_file, options)
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['options'], options)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_format_detection(self, mock_get_adapter):
        """Test that format is correctly detected and passed to get_adapter_for_format."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test processing different file types
        docproc_core.process_document(self.test_text_file)
        docproc_core.process_document(self.test_pdf_file)
        docproc_core.process_document(self.test_json_file)
        
        # Verify format detection worked for each call
        self.assertEqual(mock_get_adapter.call_count, 3)
        formats_used = [call_args[0][0] for call_args in mock_get_adapter.call_args_list]
        self.assertIn('text', formats_used)
        self.assertIn('pdf', formats_used)
        self.assertIn('json', formats_used)
    
    def test_process_document_file_not_found(self):
        """Test processing a non-existent file."""
        non_existent_file = self.temp_dir_path / "does_not_exist.txt"
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            docproc_core.process_document(non_existent_file)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_adapter_exception(self, mock_get_adapter):
        """Test handling adapter exceptions."""
        # Setup mock to raise exception
        mock_adapter = MagicMock()
        mock_adapter.process.side_effect = Exception("Adapter processing error")
        mock_get_adapter.return_value = mock_adapter
        
        # Should propagate the exception by default
        with self.assertRaises(Exception) as context:
            docproc_core.process_document(self.test_text_file)
        
        self.assertIn("Adapter processing error", str(context.exception))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text(self, mock_get_adapter):
        """Test processing text content directly."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test processing text
        text = "This is sample text content to process."
        result = docproc_core.process_text(text, 'text')
        
        # Verify
        self.assertTrue(mock_adapter.process_text_called)
        self.assertEqual(result['metadata']['format'], 'mock')
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text_with_options(self, mock_get_adapter):
        """Test processing text with options."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with options
        text = "Sample text with options."
        options = {'clean': True, 'extract_entities': True}
        result = docproc_core.process_text(text, 'text', options)
        
        # Verify
        self.assertTrue(mock_adapter.process_text_called)
        self.assertEqual(result['metadata']['options'], options)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text_empty(self, mock_get_adapter):
        """Test processing empty text."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with empty text
        result = docproc_core.process_text("", 'text')
        
        # Verify processing occurred
        self.assertTrue(mock_adapter.process_text_called)
    
    def test_process_text_unsupported_format(self):
        """Test processing text with an unsupported format."""
        # Should raise ValueError for unsupported format
        with self.assertRaises(ValueError):
            docproc_core.process_text("Sample text", 'unsupported_format')
    
    def test_get_format_for_document(self):
        """Test getting format for a document."""
        # Test different file types
        text_format = docproc_core.get_format_for_document(self.test_text_file)
        pdf_format = docproc_core.get_format_for_document(self.test_pdf_file)
        json_format = docproc_core.get_format_for_document(self.test_json_file)
        
        # Verify correct formats were detected
        self.assertEqual(text_format, 'text')
        self.assertEqual(pdf_format, 'pdf')
        self.assertEqual(json_format, 'json')
    
    def test_get_format_for_document_non_existent(self):
        """Test getting format for a non-existent document."""
        non_existent_file = self.temp_dir_path / "does_not_exist.txt"
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            docproc_core.get_format_for_document(non_existent_file)


if __name__ == '__main__':
    unittest.main()
