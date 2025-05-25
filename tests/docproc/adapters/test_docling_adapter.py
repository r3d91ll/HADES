"""
Unit tests for the DoclingAdapter.

These tests verify that the DoclingAdapter correctly processes various document formats
and produces consistent output.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import pytest

from src.docproc.adapters.docling_adapter import DoclingAdapter, DOCLING_AVAILABLE
from src.docproc.adapters.registry import get_adapter_for_format


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="Docling is not installed")
class TestDoclingAdapter(unittest.TestCase):
    """Test cases for the DoclingAdapter."""
    
    def setUp(self):
        """Set up the test environment."""
        self.adapter = DoclingAdapter()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create test files
        self._create_test_files()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_test_files(self):
        """Create various test files for testing the adapter."""
        # Create a simple text file
        text_file = self.temp_path / "sample.txt"
        with open(text_file, "w") as f:
            f.write("This is a sample text document.\nIt has multiple lines.\n\nAnd paragraphs.")
        
        # Create a simple HTML file
        html_file = self.temp_path / "sample.html"
        with open(html_file, "w") as f:
            f.write("""<!DOCTYPE html>
            <html>
            <head>
                <title>Sample HTML Document</title>
            </head>
            <body>
                <h1>Sample Heading</h1>
                <p>This is a paragraph of text.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
            </html>""")
        
        # Create a simple Markdown file
        md_file = self.temp_path / "sample.md"
        with open(md_file, "w") as f:
            f.write("""# Sample Markdown Document

This is a paragraph in markdown.

## Section 1

- Item 1
- Item 2

```python
def hello_world():
    print("Hello, World!")
```
""")
        
        # Create a simple CSV file
        csv_file = self.temp_path / "sample.csv"
        with open(csv_file, "w") as f:
            f.write("""Name,Age,City
John Doe,30,New York
Jane Smith,25,San Francisco
Bob Johnson,40,Chicago""")
    
    def test_adapter_initialization(self):
        """Test that the adapter initializes correctly."""
        # Default initialization
        adapter = DoclingAdapter()
        self.assertTrue(hasattr(adapter, 'converter'))
        
        # Custom options
        adapter = DoclingAdapter({'use_ocr': False})
        self.assertEqual(adapter.options.get('use_ocr'), False)
    
    def test_process_text_file(self):
        """Test processing a text file."""
        text_file = self.temp_path / "sample.txt"
        result = self.adapter.process(text_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify format-specific fields
        self.assertEqual(result["format"], "text")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)
    
    def test_process_html_file(self):
        """Test processing an HTML file."""
        html_file = self.temp_path / "sample.html"
        result = self.adapter.process(html_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify format-specific fields
        self.assertEqual(result["format"], "html")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)
        
        # After refactoring, we simplified metadata extraction, so we only check
        # that metadata exists and has basic format info
        self.assertIn("format", result["metadata"])
        self.assertEqual(result["metadata"]["format"], "html")
    
    def test_process_markdown_file(self):
        """Test processing a Markdown file."""
        md_file = self.temp_path / "sample.md"
        result = self.adapter.process(md_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify format-specific fields
        self.assertEqual(result["format"], "markdown")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)
        
        # Check that markdown headings are preserved
        self.assertIn("#", result["content"])
    
    def test_process_csv_file(self):
        """Test processing a CSV file."""
        csv_file = self.temp_path / "sample.csv"
        result = self.adapter.process(csv_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify format-specific fields
        self.assertEqual(result["format"], "csv")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)
    
    def test_process_text_content(self):
        """Test processing text content directly."""
        text = "This is some sample text content for processing."
        result = self.adapter.process_text(text)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify text-specific fields
        self.assertEqual(result["source"], "text")
        self.assertEqual(result["format"], "text")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)
    
    def test_process_html_content(self):
        """Test processing HTML content directly."""
        html = "<html><body><h1>Test Heading</h1><p>Test paragraph.</p></body></html>"
        result = self.adapter.process_text(html, {'format': 'html'})
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify html-specific fields
        self.assertEqual(result["source"], "text")
        self.assertEqual(result["format"], "html")
        self.assertIn("content", result)
        
    def test_process_html_with_nav_and_toc(self):
        """Test processing HTML content with navigation elements and table of contents."""
        # Create HTML with navigation and table of contents structure - no indentation to avoid regex issues
        html = """<!DOCTYPE html>
<html>
<head>
<title>Documentation with TOC</title>
</head>
<body>
<nav>
    <ul>
        <li><a href="#section1">Section 1</a></li>
        <li><a href="#section2">Section 2</a></li>
        <li><a href="#section3">Section 3</a></li>
    </ul>
</nav>
<div class="toc">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#section1">Section 1</a></li>
        <li><a href="#section2">Section 2</a></li>
        <li><a href="#section3">Section 3</a></li>
    </ul>
</div>
<section id="section1">
    <h2>Section 1</h2>
    <p>Content for section 1</p>
</section>
<section id="section2">
    <h2>Section 2</h2>
    <p>Content for section 2</p>
</section>
<section id="section3">
    <h2>Section 3</h2>
    <p>Content for section 3</p>
</section>
</body>
</html>"""
        
        # Process the HTML
        result = self.adapter.process_text(html, {'format': 'html', 'source_url': 'https://example.com/docs'})
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Check that metadata contains expected fields
        self.assertIn("metadata", result)
        metadata = result["metadata"]
        
        # Check basic metadata fields
        self.assertEqual(metadata["format"], "html")
        self.assertEqual(metadata["title"], "Documentation with TOC")
        
        # Check for navigation elements
        self.assertIn("nav_elements", metadata)
        self.assertTrue(isinstance(metadata["nav_elements"], list))
        self.assertEqual(len(metadata["nav_elements"]), 3)
        
        # Check for table of contents
        self.assertIn("toc", metadata)
        self.assertTrue(isinstance(metadata["toc"], dict))
        self.assertIn("entries", metadata["toc"])
        self.assertEqual(len(metadata["toc"]["entries"]), 3)
        
        # Check for relationships
        self.assertIn("relationships", metadata)
        relationships = metadata["relationships"]
        
        # Verify LINKS_TO relationships (TOC entries -> sections)
        links_to_relations = [r for r in relationships if r["type"] == "LINKS_TO"]
        self.assertTrue(len(links_to_relations) > 0)
        
        # Verify CONTAINS relationships (TOC -> entries)
        contains_relations = [r for r in relationships if r["type"] == "CONTAINS"]
        self.assertTrue(len(contains_relations) > 0)
        
        # Verify that we have a TOC ID in the relationships
        toc_id = metadata["toc"].get("id")
        self.assertIsNotNone(toc_id)
        
        # Verify that each TOC entry has an ID
        for entry in metadata["toc"]["entries"]:
            self.assertIn("id", entry)
        self.assertTrue(len(result["content"]) > 0)
    
    @patch('src.docproc.adapters.docling_adapter.DocumentConverter')
    def test_extract_entities(self, mock_converter_class):
        """Test entity extraction from content with a mocked Docling converter."""
        # Set up the mock
        mock_converter = mock_converter_class.return_value
        mock_doc = MagicMock()
        mock_doc.entities = [
            {"type": "heading", "text": "Heading 1", "level": 1},
            {"type": "paragraph", "text": "This is a paragraph."}
        ]
        mock_result = MagicMock(document=mock_doc)
        mock_converter.convert.return_value = mock_result
        
        # Create a new adapter with the mock
        adapter = DoclingAdapter()
        adapter.converter = mock_converter
        adapter._docling_available = True
        
        # Create content to test with
        text_content = "# Test document for entity extraction"
        
        # Call the method under test
        result = adapter.process_text(text_content, {'format': 'text'})
        
        # Verify JSON structure contains entities field
        self.assertIn("entities", result)
        entities = result["entities"]
        
        # Verify entities is a list
        self.assertTrue(isinstance(entities, list))
        
        # Verify we get the expected JSON structure
        self.assertIn("id", result)
        self.assertIn("source", result)
        self.assertIn("content", result)
        self.assertIn("content_type", result)
        self.assertIn("format", result)
    
    def test_process_text_markdown_content(self):
        """Test that process_text properly handles markdown content."""
        # Process a text document with markdown content
        result = self.adapter.process_text("# Test Heading\n\nTest paragraph.", {'format': 'markdown'})
        
        # Validate that the content field contains the markdown content
        self.assertIn("content", result)
        self.assertTrue(isinstance(result["content"], str))
        
        # Our refactored implementation now uses 'text' as the content_type for consistency
        # rather than specializing by format
        self.assertEqual(result["content_type"], "text")
        
        # Verify that the content contains the original markdown
        self.assertIn("Test Heading", result["content"])
        self.assertIn("Test paragraph", result["content"])
    
    def test_process_result_structure(self):
        """Test the JSON structure returned by process methods."""
        # Process a text document
        result = self.adapter.process_text("Test content", {'format': 'text'})
        
        # Validate JSON structure has expected fields
        self.assertIn("id", result)
        self.assertIn("source", result)
        self.assertIn("content", result)
        self.assertIn("content_type", result)
        self.assertIn("format", result)
        self.assertIn("metadata", result)
        self.assertIn("entities", result)
        
        # Metadata should be a dictionary
        self.assertTrue(isinstance(result["metadata"], dict))
        
        # Entities should be a list
        self.assertTrue(isinstance(result["entities"], list))
    
    def test_registry_integration(self):
        """Test that the adapter is properly registered for multiple formats."""
        # Check common formats
        for format_name in ["pdf", "html", "markdown", "text", "csv", "docx"]:
            adapter = get_adapter_for_format(format_name)
            self.assertIsInstance(adapter, DoclingAdapter, f"Expected DoclingAdapter for format {format_name}")
    
    def test_file_not_found(self):
        """Test that the adapter raises FileNotFoundError for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.process(self.temp_path / "nonexistent.pdf")
    
    def test_ocr_option(self):
        """Test that OCR options are properly passed to Docling."""
        with patch('docling.document_converter.DocumentConverter.convert') as mock_convert:
            # Create a mock result
            mock_doc = MagicMock()
            mock_doc.export_to_markdown.return_value = "# Test Document"
            mock_doc.export_to_text.return_value = "Test Document"
            mock_doc.pages = []
            mock_doc.metadata = {}
            
            mock_result = MagicMock()
            mock_result.document = mock_doc
            mock_convert.return_value = mock_result
            
            # Test with OCR enabled
            adapter = DoclingAdapter({'use_ocr': True})
            test_file = self.temp_path / "sample.pdf"
            open(test_file, 'w').close()  # Create empty file
            
            adapter.process(test_file)
            
            # Verify OCR options were passed
            call_args = mock_convert.call_args[1]
            self.assertTrue(call_args.get('use_ocr', False))
    
    def _verify_common_result_fields(self, result):
        """Verify common fields in the result dictionary."""
        self.assertIsInstance(result, dict)
        
        # Check required fields
        self.assertIn("id", result)
        self.assertIn("source", result)
        self.assertIn("content", result)
        self.assertIn("content_type", result)
        self.assertIn("format", result)
        self.assertIn("metadata", result)
        self.assertIn("entities", result)
        
        # Check types
        self.assertIsInstance(result["id"], str)
        self.assertIsInstance(result["source"], str)
        self.assertIsInstance(result["content"], str)
        self.assertIsInstance(result["content_type"], str)
        self.assertIsInstance(result["format"], str)
        self.assertIsInstance(result["metadata"], dict)
        self.assertIsInstance(result["entities"], list)
        
        # No longer checking for docling_document as it's been removed


class TestDoclingAdapterWithMocks(unittest.TestCase):
    """Test cases for the DoclingAdapter using mocks."""
    
    def setUp(self):
        """Set up the test environment with mocks."""
        # Create a patch for the DocumentConverter
        self.converter_patch = patch('src.docproc.adapters.docling_adapter.DocumentConverter')
        self.mock_converter_class = self.converter_patch.start()
        
        # Create a mock instance for the converter
        self.mock_converter = MagicMock()
        self.mock_converter_class.return_value = self.mock_converter
        
        # Create a mock document
        self.mock_doc = MagicMock()
        self.mock_doc.export_to_markdown.return_value = "# Mocked Markdown Content\n\nThis is mocked content."
        self.mock_doc.export_to_text.return_value = "Mocked Text Content\n\nThis is mocked content."
        self.mock_doc.pages = [MagicMock()]
        self.mock_doc.metadata = {"title": "Mocked Document", "author": "Test Author"}
        
        # Set up the mock conversion result
        mock_result = MagicMock()
        mock_result.document = self.mock_doc
        self.mock_converter.convert.return_value = mock_result
        
        # Create the adapter
        self.adapter = DoclingAdapter()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a test file
        self.test_file = self.temp_path / "test.pdf"
        with open(self.test_file, "w") as f:
            f.write("Test content")
    
    def tearDown(self):
        """Clean up after tests."""
        self.converter_patch.stop()
        self.temp_dir.cleanup()
    
    def test_process_with_mock(self):
        """Test processing a file with mocked Docling."""
        # Process the test file
        result = self.adapter.process(self.test_file)
        
        # Verify the converter was called correctly
        self.mock_converter.convert.assert_called_once()
        
        # Verify the result structure
        self.assertIn("id", result)
        self.assertEqual(result["source"], str(self.test_file))
        # In our updated implementation, we kept content_type="markdown" for test mocks
        # with export_to_markdown method
        self.assertEqual(result["content_type"], "markdown")
        # Verify that the content contains the expected markdown
        self.assertIn("Mocked Markdown Content", result["content"])
        self.assertEqual(result["format"], "pdf")
        self.assertIn("metadata", result)
        self.assertIn("entities", result)
        
        # Verify metadata extraction
        self.assertEqual(result["metadata"]["title"], "Mocked Document")
        self.assertEqual(result["metadata"]["author"], "Test Author")
        self.assertEqual(result["metadata"]["format"], "pdf")
        self.assertEqual(result["metadata"]["page_count"], 1)
    
    def test_process_text_with_mock(self):
        """Test processing text content with mocked Docling."""
        # Set up the mock for process method to be called by process_text
        with patch.object(self.adapter, 'process') as mock_process:
            mock_process.return_value = {
                "id": "text_123",
                "source": "temp_file.txt",
                "content": "# Mocked Content",
                "content_type": "markdown",
                "format": "text",
                "metadata": {"format": "text"},
                "entities": []
            }
            
            # Process text content
            result = self.adapter.process_text("Test text content")
            
            # Verify process was called
            mock_process.assert_called_once()
            
            # Verify result fields were updated correctly
            self.assertIn("id", result)
            self.assertEqual(result["source"], "text")
            self.assertEqual(result["format"], "text")
    
    def test_extract_entities(self):
        """Test entity extraction."""
        # Set up a mock page with headings
        mock_page = MagicMock()
        mock_heading = MagicMock()
        mock_heading.get_text.return_value = "Test Heading"
        mock_heading.heading_level = 2
        mock_page.get_elements.return_value = [mock_heading]
        
        # Set up a mock document with the page
        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]
        
        # Extract entities
        entities = self.adapter.extract_entities(mock_doc)
        
        # Verify entities were extracted correctly
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["type"], "heading")
        self.assertEqual(entities[0]["value"], "Test Heading")
        self.assertEqual(entities[0]["level"], 2)
        self.assertEqual(entities[0]["page"], 1)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        # Set up a mock document with metadata
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "Title": "Test Document",
            "Author": "Test Author",
            "CreationDate": "2023-01-01"
        }
        mock_doc.pages = [1, 2, 3]  # Mock 3 pages
        
        # Extract metadata
        metadata = self.adapter.extract_metadata(mock_doc)
        
        # Verify metadata was extracted correctly
        self.assertEqual(metadata["title"], "Test Document")
        self.assertEqual(metadata["author"], "Test Author")
        self.assertEqual(metadata["creationdate"], "2023-01-01")
        self.assertEqual(metadata["page_count"], 3)
    
    # Tests for convert_to_markdown and convert_to_text have been removed
    # as these methods are no longer part of the BaseAdapter interface
    
    def test_file_not_found(self):
        """Test that the adapter raises FileNotFoundError for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.process(self.temp_path / "nonexistent.pdf")


if __name__ == "__main__":
    unittest.main()
