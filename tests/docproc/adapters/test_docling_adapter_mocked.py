"""
Unit tests for the DoclingAdapter with mocks.

These tests verify that the DoclingAdapter correctly processes various document formats
and produces consistent output, using mocks to avoid dependencies on the actual Docling library.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import pytest

# Mock the Docling imports before importing DoclingAdapter
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.datamodel'] = MagicMock()
sys.modules['docling.datamodel.base_models'] = MagicMock()
sys.modules['docling.datamodel.document'] = MagicMock()

# Now import the DoclingAdapter
from src.docproc.adapters.docling_adapter import DoclingAdapter


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
        self.assertEqual(result["content"], "# Mocked Markdown Content\n\nThis is mocked content.")
        self.assertEqual(result["content_type"], "markdown")
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
    
    def test_file_not_found(self):
        """Test that the adapter raises FileNotFoundError for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.process(self.temp_path / "nonexistent.pdf")


if __name__ == "__main__":
    unittest.main()
