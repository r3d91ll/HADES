"""
Unit tests for document processing adapters.

This module tests the functionality of the Python and Docling adapters for processing different file formats.
"""

import os
import tempfile
from pathlib import Path
import unittest
from typing import Dict, Any

from src.docproc.adapters import (
    get_adapter_for_format,
    get_supported_formats,
    PythonAdapter,
    DoclingAdapter
)
from src.docproc.core import process_document, process_text, detect_format


class TestAdapters(unittest.TestCase):
    """Test suite for document processing adapters."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test files
        self.test_files = {
            'python': self._create_python_file(),
            'text': self._create_text_file(),
            'pdf': self._create_text_file(),  # Mock PDF file for DoclingAdapter
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_python_file(self) -> Path:
        """Create a sample Python file."""
        file_path = self.temp_path / "sample.py"
        with open(file_path, "w") as f:
            f.write('''#!/usr/bin/env python3
"""
Sample Python module.
"""

def hello_world():
    """Print a greeting."""
    print("Hello, World!")

class TestClass:
    """A test class."""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        """Return a greeting."""
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    hello_world()
    test = TestClass("World")
    print(test.greet())
''')
        return file_path
    
    def _create_text_file(self) -> Path:
        """Create a sample text file."""
        file_path = self.temp_path / "sample.txt"
        with open(file_path, "w") as f:
            f.write("This is a sample text file.\nIt has multiple lines.\n")
        return file_path
    
    def test_get_supported_formats(self):
        """Test that get_supported_formats returns a list of supported formats."""
        formats = get_supported_formats()
        self.assertIsInstance(formats, list)
        self.assertGreater(len(formats), 0)
        
        # Check that our core formats are supported
        self.assertIn('python', formats)
        self.assertIn('text', formats)
        self.assertIn('pdf', formats)
    
    def test_get_adapter_for_format(self):
        """Test that get_adapter_for_format returns the correct adapter for each format."""
        # Test Python adapter
        python_adapter = get_adapter_for_format('python')
        self.assertIsInstance(python_adapter, PythonAdapter)
        
        # Test Docling adapter for text
        text_adapter = get_adapter_for_format('text')
        self.assertIsInstance(text_adapter, DoclingAdapter)
        
        # Test Docling adapter for PDF
        pdf_adapter = get_adapter_for_format('pdf')
        self.assertIsInstance(pdf_adapter, DoclingAdapter)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            get_adapter_for_format('invalid_format')
    
    def test_detect_format(self):
        """Test that detect_format correctly identifies file formats."""
        # Test Python file
        python_format = detect_format(self.test_files['python'])
        self.assertEqual(python_format, 'python')
        
        # Test text file
        text_format = detect_format(self.test_files['text'])
        self.assertEqual(text_format, 'text')
        
        # Test PDF file (actually a text file with .pdf extension)
        pdf_path = self.temp_path / "sample.pdf"
        with open(pdf_path, "w") as f:
            f.write("Mock PDF content")
        pdf_format = detect_format(pdf_path)
        self.assertEqual(pdf_format, 'pdf')
    
    def test_process_document(self):
        """Test that process_document correctly processes documents."""
        # Test Python file
        python_result = process_document(self.test_files['python'])
        self._verify_result_structure(python_result, 'python')
        
        # Test text file
        text_result = process_document(self.test_files['text'])
        self._verify_result_structure(text_result, 'text')
    
    def test_process_text(self):
        """Test that process_text correctly processes text content."""
        # Test Python code
        python_code = "def test(): return 'Hello, World!'"
        python_result = process_text(python_code, 'python')
        self._verify_result_structure(python_result, 'python')
        
        # Test plain text
        text_content = "This is plain text."
        text_result = process_text(text_content, 'text')
        self._verify_result_structure(text_result, 'text')
    
    def _verify_result_structure(self, result: Dict[str, Any], format_name: str):
        """Verify that a processing result has the correct structure."""
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertIn('format', result['metadata'])
        self.assertEqual(result['metadata']['format'], format_name)


if __name__ == "__main__":
    unittest.main()
