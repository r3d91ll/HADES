"""
Tests for the metadata extraction functionality.
"""

import unittest
from src.docproc.utils.metadata_extractor import (
    extract_metadata,
    extract_academic_pdf_metadata,
    extract_website_doc_metadata
)


class TestMetadataExtractor(unittest.TestCase):
    """Test cases for metadata extraction functions."""

    def test_extract_academic_pdf_metadata(self):
        """Test metadata extraction from academic PDF content."""
        # Sample academic paper content
        content = """## PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths

## Boyu Chen 1, Zirui Guo 1,2, Zidan Yang 1,3, Yuluo Chen 1, Junze Chen 1, Zhenghao Liu 3, Chuan Shi 1, Cheng Yang 1

1 Beijing University of Posts and Telecommunications

2 University of Hong Kong 3 Northeastern University chenbys4@bupt.edu.cn,yangcheng@bupt.edu.cn

## Abstract

Retrieval-augmented generation (RAG) improves the response quality of large language models...
"""
        source_path = "/path/to/test.pdf"
        
        metadata = extract_academic_pdf_metadata(content, source_path)
        
        self.assertEqual(metadata["doc_type"], "academic_pdf")
        self.assertEqual(metadata["title"], "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths")
        self.assertTrue(len(metadata["authors"]) > 0)
        self.assertIn("Boyu Chen", " ".join(metadata["authors"]))

    def test_extract_website_doc_metadata(self):
        """Test metadata extraction from website documentation."""
        # Sample HTML content
        content = """<!DOCTYPE html>
<html>
<head>
<title>LangChain Documentation: Getting Started</title>
<meta name="description" content="Get started with LangChain">
<meta name="keywords" content="langchain, ai, documentation">
<meta name="author" content="LangChain Team">
</head>
<body>
<h1 id="getting-started">Getting Started with LangChain</h1>
<p>Last updated: 2023-12-15</p>
<nav>
  <ul class="toc">
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quickstart">Quick Start</a></li>
    <li><a href="#concepts">Key Concepts</a></li>
  </ul>
</nav>
<div>This guide helps you start using LangChain...</div>
<h2 id="installation">Installation</h2>
<p>Install LangChain with pip:</p>
<pre><code>pip install langchain</code></pre>
<h2 id="quickstart">Quick Start</h2>
<p>Here's a simple example to get started:</p>
<h2 id="concepts">Key Concepts</h2>
<p>LangChain is built around several key concepts:</p>
<h3 id="chains">Chains</h3>
<p>Chains combine multiple components together.</p>
<h3 id="agents">Agents</h3>
<p>Agents use LLMs to determine actions to take.</p>
</body>
</html>
"""
        source_url = "https://python.langchain.com/docs/getting_started.html"
        
        metadata = extract_website_doc_metadata(content, source_url)
        
        # Basic metadata tests
        self.assertEqual(metadata["doc_type"], "website_documentation")
        self.assertEqual(metadata["title"], "LangChain Documentation: Getting Started")
        self.assertEqual(metadata["publisher"], "python.langchain.com")
        self.assertEqual(metadata["date_published"], "2023-12-15")
        self.assertTrue(metadata["is_documentation"])
        
        # Test author from meta tag
        self.assertEqual(metadata["authors"], ["LangChain Team"])
        
        # Test meta tags
        self.assertIn("meta_tags", metadata)
        self.assertEqual(metadata["meta_tags"]["description"], "Get started with LangChain")
        self.assertEqual(metadata["meta_tags"]["keywords"], "langchain, ai, documentation")
        
        # Test navigation structure
        self.assertIn("navigation", metadata)
        self.assertIn("toc", metadata["navigation"])
        self.assertIn("sections", metadata["navigation"])
        
        # Test table of contents
        toc_entries = metadata["navigation"]["toc"]
        self.assertTrue(len(toc_entries) >= 3)
        toc_titles = [entry["title"] for entry in toc_entries]
        self.assertIn("Installation", toc_titles)
        self.assertIn("Quick Start", toc_titles)
        self.assertIn("Key Concepts", toc_titles)
        
        # Test sections
        sections = metadata["navigation"]["sections"]
        self.assertTrue(len(sections) >= 5)  # h1, h2, h2, h2, h3, h3
        
        # Verify section hierarchy
        h1_sections = [s for s in sections if s["level"] == 1]
        h2_sections = [s for s in sections if s["level"] == 2]
        h3_sections = [s for s in sections if s["level"] == 3]
        
        self.assertEqual(len(h1_sections), 1)
        self.assertTrue(len(h2_sections) >= 3)
        self.assertTrue(len(h3_sections) >= 2)
        
        # Test relationships
        self.assertIn("relationships", metadata)
        relationships = metadata["relationships"]
        self.assertTrue(len(relationships) > 0)
        
        # Test that we have CONTAINS relationships (parent-child)
        contains_relationships = [r for r in relationships if r["type"] == "CONTAINS"]
        self.assertTrue(len(contains_relationships) > 0)
        
        # Test that we have FOLLOWS relationships (sequential sections)
        follows_relationships = [r for r in relationships if r["type"] == "FOLLOWS"]
        self.assertTrue(len(follows_relationships) > 0)
        
        # Test that we have LINKS_TO relationships from TOC to sections
        links_to_relationships = [r for r in relationships if r["type"] == "LINKS_TO"]
        self.assertTrue(len(links_to_relationships) > 0)

    def test_extract_metadata_pdf(self):
        """Test the main metadata extraction function for PDF."""
        content = "## Test Paper\n\nBy Author X\n\n## Abstract\nThis is a test."
        source_path = "test.pdf"
        format_type = "pdf"
        
        metadata = extract_metadata(content, source_path, format_type)
        
        self.assertEqual(metadata["doc_type"], "academic_pdf")
        self.assertEqual(metadata["title"], "Test Paper")

    def test_extract_metadata_html(self):
        """Test the main metadata extraction function for HTML."""
        content = "<title>Test Page</title><h1>Test</h1>"
        source_path = "https://example.com/test.html"
        format_type = "html"
        
        metadata = extract_metadata(content, source_path, format_type)
        
        self.assertEqual(metadata["doc_type"], "website_documentation")
        self.assertEqual(metadata["title"], "Test Page")

    def test_extract_metadata_unknown_format(self):
        """Test the main metadata extraction function for unknown format."""
        content = "Some content"
        source_path = "test.unknown"
        format_type = "unknown"
        
        metadata = extract_metadata(content, source_path, format_type)
        
        self.assertEqual(metadata["doc_type"], "unknown")
        self.assertEqual(metadata["title"], "UNK")
        self.assertEqual(metadata["authors"], ["UNK"])


if __name__ == "__main__":
    unittest.main()
