"""
Tests for the metadata extraction utilities.
"""

import unittest
from pathlib import Path
from src.docproc.utils.metadata_extractor import (
    extract_metadata,
    extract_academic_pdf_metadata
)


class TestMetadataExtractor(unittest.TestCase):
    """Test cases for the metadata extraction utilities."""

    def test_extract_metadata_pdf(self):
        """Test extracting metadata from PDF content."""
        content = """
        ## Path-Augmented Retrieval
        
        Author1, Author2, Author3
        University of Research
        
        Abstract
        This paper introduces PathRAG, a novel approach to RAG systems.
        
        Published in 2023
        """
        source_path = "test_document.pdf"
        format_type = "pdf"
        
        metadata = extract_metadata(content, source_path, format_type)
        
        # Check that PDF extraction was used
        self.assertEqual(metadata["doc_type"], "academic_pdf")
        self.assertEqual(metadata["title"], "Path-Augmented Retrieval")
        self.assertIn("Author1, Author2, Author3", metadata["authors"])
        self.assertEqual(metadata["date_published"], "2023")
        self.assertEqual(metadata["source"], source_path)
    
    def test_extract_metadata_non_pdf(self):
        """Test extracting metadata from non-PDF content."""
        content = "# Some markdown content"
        source_path = "test_document.md"
        format_type = "markdown"
        
        metadata = extract_metadata(content, source_path, format_type)
        
        # Check default metadata for non-PDF
        self.assertEqual(metadata["doc_type"], "markdown")
        self.assertEqual(metadata["title"], "test_document")
        self.assertEqual(metadata["authors"], [])
        self.assertEqual(metadata["date_published"], "UNK")
        self.assertEqual(metadata["source"], source_path)
    
    def test_extract_academic_pdf_metadata(self):
        """Test academic PDF metadata extraction in detail."""
        content = """
        ## Comprehensive Study of AI
        
        John Smith, Jane Doe
        AI Research Institute
        
        Abstract
        This is the abstract of this paper.
        
        1. Introduction
        
        Published in 2022 by IEEE
        """
        source_path = "academic_paper.pdf"
        
        metadata = extract_academic_pdf_metadata(content, source_path)
        
        self.assertEqual(metadata["doc_type"], "academic_pdf")
        self.assertEqual(metadata["title"], "Comprehensive Study of AI")
        self.assertIn("John Smith, Jane Doe", metadata["authors"])
        self.assertEqual(metadata["date_published"], "2022")
        self.assertEqual(metadata["publisher"], "IEEE")
        self.assertEqual(metadata["source"], source_path)
    
    def test_extract_academic_pdf_minimal_content(self):
        """Test PDF metadata extraction with minimal content."""
        content = "Some content with no clear metadata"
        source_path = "minimal.pdf"
        
        metadata = extract_academic_pdf_metadata(content, source_path)
        
        # Should use fallbacks
        self.assertEqual(metadata["doc_type"], "academic_pdf")
        self.assertEqual(metadata["title"], "Some content with no clear metadata")
        self.assertEqual(metadata["authors"], ["UNK"])
        self.assertEqual(metadata["date_published"], "UNK")
        self.assertEqual(metadata["publisher"], "UNK")
        self.assertEqual(metadata["source"], source_path)


if __name__ == "__main__":
    unittest.main()
