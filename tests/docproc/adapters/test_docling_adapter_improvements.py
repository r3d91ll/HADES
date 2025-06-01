"""
Tests for the docling_adapter module.

This module contains tests for the DoclingAdapter class, focusing on the
previously untested methods to improve test coverage.
"""

import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

# Mock dependencies to avoid import errors
import sys
sys.modules['src.utils'] = mock.MagicMock()
sys.modules['src.utils.file_utils'] = mock.MagicMock()
sys.modules['src.types'] = mock.MagicMock()
sys.modules['src.types.docproc'] = mock.MagicMock()
sys.modules['src.types.docproc.adapter'] = mock.MagicMock()
sys.modules['src.types.docproc.metadata'] = mock.MagicMock()
sys.modules['src.docproc.adapters.registry'] = mock.MagicMock()
sys.modules['docling'] = mock.MagicMock()

# Define minimal required classes for testing
class ProcessedDocument(dict):
    pass

class EntityDict(dict):
    pass

class MetadataDict(dict):
    pass

class AdapterOptions(dict):
    pass

# Add these to the mocked modules
sys.modules['src.types.docproc.adapter'].ProcessedDocument = ProcessedDocument
sys.modules['src.types.docproc.adapter'].AdapterOptions = AdapterOptions
sys.modules['src.types.docproc.metadata'].EntityDict = EntityDict
sys.modules['src.types.docproc.metadata'].MetadataDict = MetadataDict

# Import the DoclingAdapter class
from src.docproc.adapters.docling_adapter import DoclingAdapter

class TestDoclingAdapter(unittest.TestCase):
    """Test cases for DoclingAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock docling document for testing
        self.mock_docling_doc = mock.MagicMock()
        self.mock_docling_doc.text = "This is a test document."
        self.mock_docling_doc.tokens = [
            mock.MagicMock(text="This", start_char=0, end_char=4, pos_="DET"),
            mock.MagicMock(text="is", start_char=5, end_char=7, pos_="AUX"),
            mock.MagicMock(text="a", start_char=8, end_char=9, pos_="DET"),
            mock.MagicMock(text="test", start_char=10, end_char=14, pos_="NOUN"),
            mock.MagicMock(text="document", start_char=15, end_char=23, pos_="NOUN"),
            mock.MagicMock(text=".", start_char=23, end_char=24, pos_="PUNCT"),
        ]
        self.mock_docling_doc.ents = [
            mock.MagicMock(text="test document", start_char=10, end_char=23, label_="TEST_ENTITY")
        ]
        self.mock_docling_doc.noun_chunks = [
            mock.MagicMock(text="This", start_char=0, end_char=4),
            mock.MagicMock(text="a test document", start_char=8, end_char=23),
        ]
        
        # Create the adapter with a mocked docling instance
        self.mock_docling = mock.MagicMock()
        self.mock_docling.load.return_value = self.mock_docling_doc
        with mock.patch('docling.load', self.mock_docling.load):
            self.adapter = DoclingAdapter()
    
    def test_init(self):
        """Test initialization of DoclingAdapter."""
        # Test with default options
        adapter = DoclingAdapter()
        self.assertEqual({}, adapter.options)
        
        # Test with custom options
        custom_options = {"model": "en_core_web_sm", "extract_entities": True}
        adapter = DoclingAdapter(custom_options)
        self.assertEqual(custom_options, adapter.options)
    
    def test_extract_metadata_text(self):
        """Test extract_metadata with text input."""
        with mock.patch('docling.load', return_value=self.mock_docling_doc):
            metadata = self.adapter.extract_metadata("This is a test document.")
            
            # Basic checks
            self.assertIsInstance(metadata, dict)
            self.assertEqual("text", metadata["document_type"])
            self.assertEqual("en", metadata["language"])
            
            # Check token counts
            self.assertEqual("6", metadata["token_count"])
            self.assertEqual("5", metadata["word_count"])  # Excluding punctuation
            self.assertEqual("1", metadata["sentence_count"])
            self.assertEqual("1", metadata["entity_count"])
    
    def test_extract_metadata_dict(self):
        """Test extract_metadata with dict input."""
        # Create a processed document
        processed_doc = {
            "content": "This is a test document.",
            "metadata": {
                "existing_metadata": "value"
            }
        }
        
        with mock.patch('docling.load', return_value=self.mock_docling_doc):
            metadata = self.adapter.extract_metadata(processed_doc)
            
            # Check that we get a combined metadata dict
            self.assertIsInstance(metadata, dict)
            self.assertEqual("text", metadata["document_type"])
            self.assertEqual("value", metadata["existing_metadata"])
    
    def test_extract_entities_text(self):
        """Test _extract_entities with text input."""
        with mock.patch('docling.load', return_value=self.mock_docling_doc):
            # Access the private method for testing
            entities = self.adapter._extract_entities("This is a test document.")
            
            # Check basic entity structure
            self.assertIsInstance(entities, list)
            self.assertGreaterEqual(len(entities), 1)
            
            # Check entity contents
            entity_types = [e["type"] for e in entities]
            self.assertIn("TEST_ENTITY", entity_types)
            
            # Check noun chunks
            noun_chunk_entities = [e for e in entities if e["type"] == "NOUN_CHUNK"]
            self.assertGreaterEqual(len(noun_chunk_entities), 1)
    
    def test_extract_metadata_private(self):
        """Test _extract_metadata private method."""
        with mock.patch('docling.load', return_value=self.mock_docling_doc):
            # Access the private method for testing
            metadata = self.adapter._extract_metadata("This is a test document.")
            
            # Check metadata structure
            self.assertIsInstance(metadata, dict)
            self.assertEqual("text", metadata["document_type"])
            self.assertEqual("en", metadata["language"])
            
            # Check POS counts
            self.assertIn("noun_count", metadata)
            self.assertIn("verb_count", metadata)
            self.assertIn("adj_count", metadata)
            
            # POS tag counts based on our mock tokens
            self.assertEqual("2", metadata["noun_count"])  # "test" and "document"
            self.assertEqual("0", metadata["verb_count"])  # No verbs in our example
            
            # Check entity counts
            self.assertEqual("1", metadata["entity_count"])

if __name__ == "__main__":
    unittest.main()
