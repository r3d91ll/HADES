"""
Isolated tests for format detector utilities.

This test module uses mocks to avoid external dependencies.
"""

import unittest
import sys
import tempfile
from pathlib import Path

# Apply mocks before importing docproc modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.docproc.docproc_test_utils import patch_modules
patch_modules()

# Now we can safely import our modules
from src.docproc.utils.format_detector import detect_format_from_path, detect_format_from_content


class TestFormatDetectorIsolated(unittest.TestCase):
    """Isolated test cases for document format detector utilities."""
    
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
        self.assertEqual(detect_format_from_path(Path('page.htm')), 'html')  # .htm variant
        self.assertEqual(detect_format_from_path(Path('notes.md')), 'markdown')  # markdown
        self.assertEqual(detect_format_from_path(Path('document.docx')), 'docx')  # docx
        self.assertEqual(detect_format_from_path(Path('document.doc')), 'docx')  # doc
        self.assertEqual(detect_format_from_path(Path('script.py')), 'python')
        self.assertEqual(detect_format_from_path(Path('code.js')), 'javascript')  # js code
        self.assertEqual(detect_format_from_path(Path('code.java')), 'java')  # java code
        self.assertEqual(detect_format_from_path(Path('code.cpp')), 'code')  # c++ code
        self.assertEqual(detect_format_from_path(Path('code.ts')), 'code')  # typescript code
        self.assertEqual(detect_format_from_path(Path('data.json')), 'json')
        self.assertEqual(detect_format_from_path(Path('config.yaml')), 'yaml')
        self.assertEqual(detect_format_from_path(Path('config.yml')), 'yaml')
        self.assertEqual(detect_format_from_path(Path('data.xml')), 'xml')
        self.assertEqual(detect_format_from_path(Path('dataset.csv')), 'csv')
        self.assertEqual(detect_format_from_path(Path('notes.txt')), 'text')  # .txt explicit text
        
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
        
        # Test mimetype fallbacks
        with tempfile.NamedTemporaryFile(suffix='.htm') as temp_file:
            self.assertEqual(detect_format_from_path(Path(temp_file.name)), 'html')
            
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            self.assertEqual(detect_format_from_path(Path(temp_file.name)), 'text')
    
    def test_detect_format_from_content(self):
        """Test detecting format from content."""
        # Test PDF content
        pdf_content = "%PDF-1.5\nSome binary content"
        self.assertEqual(detect_format_from_content(pdf_content), 'pdf')
        
        # Test HTML content
        html_content1 = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"
        self.assertEqual(detect_format_from_content(html_content1), 'html')
        
        # Test HTML with doctype
        html_content2 = "<!DOCTYPE html>\n<html><head><title>Test</title></head><body></body></html>"
        self.assertEqual(detect_format_from_content(html_content2), 'html')
        
        # Test JSON content
        json_content = '{"name": "John", "age": 30, "city": "New York"}'
        self.assertEqual(detect_format_from_content(json_content), 'json')
        
        # Test YAML content
        yaml_content = "name: John\nage: 30\ncity: New York"
        self.assertEqual(detect_format_from_content(yaml_content), 'yaml')
        
        # Test YAML with multiple key-value pairs
        complex_yaml = "\n".join([
            "# Server Configuration",
            "server:",
            "  port: 8080",
            "  host: localhost",
            "\n",
            "database:",
            "  url: jdbc:mysql://localhost:3306/db",
            "  username: admin",
            "  password: password123"
        ])
        self.assertEqual(detect_format_from_content(complex_yaml), 'yaml')
        
        # Test XML content
        xml_content = '<?xml version="1.0" encoding="UTF-8"?><root><name>John</name></root>'
        self.assertEqual(detect_format_from_content(xml_content), 'xml')
        
        # Test XML with simple tag
        simple_xml = '<config><server>localhost</server></config>'
        self.assertEqual(detect_format_from_content(simple_xml), 'xml')
        
        # Test Markdown content
        markdown_content = "# Title\n\nThis is some **bold** text with a [link](https://example.com)\n\n## Subtitle\n\n- List item 1\n- List item 2"
        self.assertEqual(detect_format_from_content(markdown_content), 'markdown')
        
        # Test Markdown with just headings and separator
        simple_markdown = "# Document Title\n---\nContent starts here"
        self.assertEqual(detect_format_from_content(simple_markdown), 'markdown')
        
        # Test Python content
        python_content = 'def hello():\n    print("Hello world")\n\nif __name__ == "__main__":\n    hello()'
        self.assertEqual(detect_format_from_content(python_content), 'python')
        
        # Test other code content
        js_content = 'function sayHello() {\n  console.log("Hello!");\n}\n\nclass User {\n  constructor(name) {\n    this.name = name;\n  }\n}'
        self.assertEqual(detect_format_from_content(js_content), 'code')
        
        # Test CSV content
        csv_content = 'name,age,city\nJohn,30,"New York"\nJane,25,Boston'
        self.assertEqual(detect_format_from_content(csv_content), 'csv')
        
        # Test plain text content
        text_content = "This is a plain text document with no specific format."
        self.assertEqual(detect_format_from_content(text_content), 'text')


if __name__ == '__main__':
    unittest.main()
