"""
Tests for document format detection utilities.

This test module focuses only on the format detector functionality.
"""

import unittest
import tempfile
from pathlib import Path

from src.docproc.utils.format_detector import detect_format_from_path, detect_format_from_content


class TestFormatDetector(unittest.TestCase):
    """Test cases for document format detector utilities."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
    
    def test_detect_format_from_path(self):
        """Test detecting format from file path."""
        # Test common file extensions
        self.assertEqual(detect_format_from_path(Path('document.pdf')), 'pdf')
        self.assertEqual(detect_format_from_path(Path('page.html')), 'html')
        self.assertEqual(detect_format_from_path(Path('script.py')), 'python')
        self.assertEqual(detect_format_from_path(Path('data.json')), 'json')
        self.assertEqual(detect_format_from_path(Path('config.yaml')), 'yaml')
        self.assertEqual(detect_format_from_path(Path('config.yml')), 'yaml')
        self.assertEqual(detect_format_from_path(Path('data.xml')), 'xml')
        self.assertEqual(detect_format_from_path(Path('dataset.csv')), 'csv')
        
        # Test uppercase extensions
        self.assertEqual(detect_format_from_path(Path('DOCUMENT.PDF')), 'pdf')
        self.assertEqual(detect_format_from_path(Path('PAGE.HTML')), 'html')
        
        # Test mixed case extensions
        self.assertEqual(detect_format_from_path(Path('Document.Pdf')), 'pdf')
        self.assertEqual(detect_format_from_path(Path('Page.Html')), 'html')
        
        # Test default for unknown extensions
        self.assertEqual(detect_format_from_path(Path('unknown.xyz')), 'text')
        
        # Test pure Path objects
        self.assertEqual(detect_format_from_path(Path('/path/to/document.pdf')), 'pdf')
        
        # Test paths with multiple dots
        self.assertEqual(detect_format_from_path(Path('archive.tar.gz')), 'archive')
        
        # Test paths without extensions
        self.assertEqual(detect_format_from_path(Path('README')), 'text')
    
    def test_detect_format_from_content(self):
        """Test detecting format from content."""
        # Test HTML content
        html_content = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"
        self.assertEqual(detect_format_from_content(html_content), 'html')
        
        # Test JSON content
        json_content = '{"name": "John", "age": 30, "city": "New York"}'
        self.assertEqual(detect_format_from_content(json_content), 'json')
        
        # Test YAML content
        yaml_content = "name: John\nage: 30\ncity: New York"
        self.assertEqual(detect_format_from_content(yaml_content), 'yaml')
        
        # Test XML content
        xml_content = '<?xml version="1.0" encoding="UTF-8"?><root><name>John</name></root>'
        self.assertEqual(detect_format_from_content(xml_content), 'xml')
        
        # Test Python content
        python_content = 'def hello():\n    print("Hello world")\n\nif __name__ == "__main__":\n    hello()'
        self.assertEqual(detect_format_from_content(python_content), 'python')
        
        # Test CSV content
        csv_content = 'name,age,city\nJohn,30,"New York"\nJane,25,Boston'
        self.assertEqual(detect_format_from_content(csv_content), 'csv')
        
        # Test plain text content
        text_content = "This is a plain text document with no specific format."
        self.assertEqual(detect_format_from_content(text_content), 'text')


if __name__ == '__main__':
    unittest.main()
