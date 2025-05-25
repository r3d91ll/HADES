"""
Unit tests for the markdown adapter.
"""

import unittest
from pathlib import Path
import tempfile
import os

from src.docproc.adapters.markdown_adapter import MarkdownAdapter, create_adapter


class TestMarkdownAdapter(unittest.TestCase):
    """Test the MarkdownAdapter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = MarkdownAdapter()
        
        # Create a temporary test file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = Path(self.temp_dir.name) / "test_document.md"
        
        # Sample markdown content
        self.test_content = """# Test Document Title

Author: John Doe, Jane Smith

Published: 2023-05-15

## Introduction

This is a test markdown document.

### Subsection

This is a subsection with some content.

```python
def hello_world():
    print("Hello, world!")
```

## Data Table

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

Check out [this link](https://example.com) for more info.
"""
        
        # Write content to file
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write(self.test_content)
    
    def tearDown(self):
        """Clean up resources."""
        self.temp_dir.cleanup()
    
    def test_process_markdown(self):
        """Test basic markdown processing."""
        result = self.adapter.process(self.test_file_path)
        
        # Check basic structure
        self.assertEqual(result["format"], "markdown")
        self.assertEqual(result["source"], str(self.test_file_path))
        self.assertEqual(result["id"], "test_document")
        self.assertIn("content", result)
        self.assertIn("entities", result)
        self.assertIn("metadata", result)
    
    def test_metadata_extraction(self):
        """Test metadata extraction from markdown."""
        result = self.adapter.process(self.test_file_path)
        metadata = result["metadata"]
        
        # Check extracted metadata
        self.assertEqual(metadata["title"], "Test Document Title")
        self.assertIn("John Doe", metadata["authors"])
        self.assertIn("Jane Smith", metadata["authors"])
        self.assertEqual(metadata["date_published"], "2023-05-15")
        self.assertEqual(metadata["doc_type"], "markdown")
    
    def test_entity_extraction(self):
        """Test entity extraction from markdown."""
        result = self.adapter.process(self.test_file_path)
        entities = result["entities"]
        
        # Check entities types
        entity_types = [entity["type"] for entity in entities]
        self.assertIn("h1", entity_types)
        self.assertIn("h2", entity_types)
        self.assertIn("h3", entity_types)
        self.assertIn("code_block", entity_types)
        self.assertIn("table", entity_types)
        self.assertIn("link", entity_types)
        
        # Verify heading hierarchy
        h1_entities = [e for e in entities if e["type"] == "h1"]
        h2_entities = [e for e in entities if e["type"] == "h2"]
        h3_entities = [e for e in entities if e["type"] == "h3"]
        
        self.assertEqual(len(h1_entities), 1)
        self.assertEqual(h1_entities[0]["name"], "Test Document Title")
        
        self.assertEqual(len(h2_entities), 2)
        self.assertEqual(h2_entities[0]["name"], "Introduction")
        
        self.assertEqual(len(h3_entities), 1)
        self.assertEqual(h3_entities[0]["name"], "Subsection")
        
        # Check code block
        code_blocks = [e for e in entities if e["type"] == "code_block"]
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]["language"], "python")
        self.assertIn("hello_world", code_blocks[0]["content"])
        
        # Check link
        links = [e for e in entities if e["type"] == "link"]
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["url"], "https://example.com")
        self.assertEqual(links[0]["name"], "this link")
    
    def test_minimal_content(self):
        """Test processing with minimal content."""
        minimal_path = Path(self.temp_dir.name) / "minimal.md"
        with open(minimal_path, 'w', encoding='utf-8') as f:
            f.write("Simple content without metadata or structure")
        
        result = self.adapter.process(minimal_path)
        
        # Check fallbacks
        self.assertEqual(result["metadata"]["title"], "minimal")
        self.assertEqual(result["metadata"]["authors"], [])
        self.assertEqual(result["metadata"]["date_published"], "UNK")
        self.assertEqual(len(result["entities"]), 0)
    
    def test_create_adapter_factory(self):
        """Test adapter factory function."""
        adapter = create_adapter()
        self.assertIsInstance(adapter, MarkdownAdapter)


if __name__ == "__main__":
    unittest.main()
