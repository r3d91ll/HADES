"""
Isolated tests for document processing core module.

This test module uses mocks to avoid external dependencies.
"""

import unittest
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Apply mocks before importing docproc modules
sys.path.insert(0, str(Path(__file__).parent))
from tests.docproc.docproc_test_utils import patch_modules
patch_modules()

# Now we can safely import our modules
from src.docproc.core import process_document, process_text, get_format_for_document, detect_format
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
    
    # convert_to_markdown and convert_to_text methods have been removed
    # as they are no longer part of the BaseAdapter interface


class TestCore(unittest.TestCase):
    """Test cases for the document processing core module."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create sample test files
        self.test_text_file = self.temp_dir_path / "test.txt"
        with open(self.test_text_file, "w") as f:
            f.write("This is a sample text file.")
        
        self.test_pdf_file = self.temp_dir_path / "test.pdf"
        with open(self.test_pdf_file, "w") as f:
            f.write("This is a sample PDF file (actually text).")
        
        self.test_json_file = self.temp_dir_path / "test.json"
        with open(self.test_json_file, "w") as f:
            f.write('{"name": "Test", "value": 123}')
    
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
        result = process_document(str(self.test_text_file))
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['file_path'], str(self.test_text_file))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_path_object(self, mock_get_adapter):
        """Test processing a document with a Path object."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with Path object
        result = process_document(self.test_text_file)
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['file_path'], str(self.test_text_file))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_with_options(self, mock_get_adapter):
        """Test processing a document with options."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with options
        options = {'clean': True, 'extract_entities': True}
        result = process_document(self.test_text_file, options)
        
        # Verify
        self.assertTrue(mock_adapter.process_called)
        self.assertEqual(result['metadata']['options'], options)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_format_detection(self, mock_get_adapter):
        """Test that format is correctly detected and passed to get_adapter_for_format."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test with text file
        process_document(self.test_text_file)
        
        # Verify format detection
        mock_get_adapter.assert_called_with('text')
        
        # Test with PDF file
        process_document(self.test_pdf_file)
        
        # Verify format detection
        mock_get_adapter.assert_called_with('pdf')
        
        # Test with JSON file
        process_document(self.test_json_file)
        
        # Verify format detection
        mock_get_adapter.assert_called_with('json')
    
    def test_process_document_file_not_found(self):
        """Test processing a non-existent file."""
        non_existent_file = self.temp_dir_path / "does_not_exist.txt"
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            process_document(non_existent_file)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_adapter_exception(self, mock_get_adapter):
        """Test handling adapter exceptions."""
        # Setup mock to raise exception
        mock_adapter = MagicMock()
        mock_adapter.process.side_effect = Exception("Adapter processing error")
        mock_get_adapter.return_value = mock_adapter
        
        # Should propagate the exception by default
        with self.assertRaises(Exception) as context:
            process_document(self.test_text_file)
        
        self.assertIn("Adapter processing error", str(context.exception))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text(self, mock_get_adapter):
        """Test processing text content directly."""
        # Setup mock
        mock_adapter = MockAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Test processing text
        text = "This is sample text content to process."
        result = process_text(text, 'text')
        
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
        result = process_text(text, 'text', options)
        
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
        result = process_text("", 'text')
        
        # Verify
        self.assertTrue(mock_adapter.process_text_called)
        self.assertIn('content', result)
    
    def test_process_text_unsupported_format(self):
        """Test processing text with an unsupported format."""
        # Should raise ValueError for unsupported format
        with self.assertRaises(ValueError):
            process_text("Sample text", 'unsupported_format')
    
    def test_get_format_for_document(self):
        """Test getting format for a document."""
        # Test different file types
        text_format = get_format_for_document(self.test_text_file)
        pdf_format = get_format_for_document(self.test_pdf_file)
        json_format = get_format_for_document(self.test_json_file)
        
        # Verify correct formats were detected
        self.assertEqual(text_format, 'text')
        self.assertEqual(pdf_format, 'pdf')
        self.assertEqual(json_format, 'json')
    
    def test_get_format_for_document_non_existent(self):
        """Test getting format for a non-existent document."""
        non_existent_file = self.temp_dir_path / "does_not_exist.txt"
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            get_format_for_document(non_existent_file)
    
    def test_detect_format(self):
        """Test the detect_format function."""
        # Test with various file paths
        self.assertEqual(detect_format(self.test_text_file), 'text')
        self.assertEqual(detect_format(self.test_pdf_file), 'pdf')
        self.assertEqual(detect_format(self.test_json_file), 'json')
        
        # Test with string paths
        self.assertEqual(detect_format(str(self.test_text_file)), 'text')
        self.assertEqual(detect_format(str(self.test_pdf_file)), 'pdf')
        self.assertEqual(detect_format(str(self.test_json_file)), 'json')


if __name__ == '__main__':
    unittest.main()
