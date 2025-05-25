"""
Tests for the markdown entity extractor.
"""

import unittest
from pathlib import Path
import tempfile
import os

from src.docproc.utils.markdown_entity_extractor import (
    extract_markdown_entities,
    extract_markdown_metadata
)


class TestMarkdownEntityExtractor(unittest.TestCase):
    """Test the markdown entity extractor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample markdown content for testing
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
    
    def test_extract_markdown_entities(self):
        """Test entity extraction from markdown content."""
        entities = extract_markdown_entities(self.test_content)
        
        # Check that we found various entity types
        entity_types = [entity["type"] for entity in entities]
        
        # Check headings with different levels
        self.assertIn("heading_1", entity_types)
        self.assertIn("heading_2", entity_types)
        self.assertIn("heading_3", entity_types)
        
        # Check code blocks
        self.assertIn("code_block", entity_types)
        
        # Check links
        self.assertIn("link", entity_types)
        
        # Check tables
        self.assertIn("table", entity_types)
        
        # Verify specific entities
        headings = [e for e in entities if e["type"].startswith("heading_")]
        code_blocks = [e for e in entities if e["type"] == "code_block"]
        links = [e for e in entities if e["type"] == "link"]
        tables = [e for e in entities if e["type"] == "table"]
        
        # Check heading content
        self.assertEqual(len(headings), 4)
        h1 = next((h for h in headings if h["level"] == 1), None)
        self.assertEqual(h1["value"], "Test Document Title")
        
        # Check code block
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]["language"], "python")
        self.assertIn("hello_world", code_blocks[0]["value"])
        
        # Check link
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["url"], "https://example.com")
        self.assertEqual(links[0]["text"], "this link")
        
        # Check table
        self.assertEqual(len(tables), 1)
    
    def test_extract_markdown_metadata(self):
        """Test metadata extraction from markdown content."""
        metadata = extract_markdown_metadata(self.test_content, "test_document.md")
        
        # Check extracted metadata
        self.assertEqual(metadata["doc_type"], "markdown")
        self.assertEqual(metadata["title"], "Test Document Title")
        self.assertIn("John Doe", metadata["authors"])
        self.assertIn("Jane Smith", metadata["authors"])
        self.assertEqual(metadata["date_published"], "Published: 2023-05-15")
        self.assertEqual(metadata["source"], "test_document.md")
    
    def test_extract_minimal_content(self):
        """Test extraction with minimal content."""
        minimal_content = "Some basic markdown without structure."
        
        entities = extract_markdown_entities(minimal_content)
        self.assertEqual(len(entities), 0)
        
        metadata = extract_markdown_metadata(minimal_content, "minimal.md")
        self.assertEqual(metadata["doc_type"], "markdown")
        self.assertEqual(metadata["source"], "minimal.md")
        self.assertNotIn("authors", metadata)
        self.assertNotIn("date_published", metadata)
    
    def test_different_heading_formats(self):
        """Test various heading formats."""
        content = """# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6
"""
        entities = extract_markdown_entities(content)
        heading_counts = {}
        
        for entity in entities:
            if entity["type"].startswith("heading_"):
                level = int(entity["type"].split("_")[1])
                heading_counts[level] = heading_counts.get(level, 0) + 1
        
        # Check we found all 6 heading levels
        for level in range(1, 7):
            self.assertEqual(heading_counts.get(level, 0), 1)
    
    def test_multiple_code_blocks(self):
        """Test extraction of multiple code blocks with different languages."""
        content = """```python
def hello():
    return "Hello"
```

```javascript
function hello() {
    return "Hello";
}
```

```
Plain text code block
```
"""
        entities = extract_markdown_entities(content)
        code_blocks = [e for e in entities if e["type"] == "code_block"]
        
        self.assertEqual(len(code_blocks), 3)
        
        languages = [block["language"] for block in code_blocks]
        self.assertIn("python", languages)
        self.assertIn("javascript", languages)
        self.assertIn("text", languages)  # Default for unmarked code blocks


if __name__ == "__main__":
    unittest.main()
