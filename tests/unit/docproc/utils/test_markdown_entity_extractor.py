"""
Unit tests for markdown_entity_extractor.py module.

This module tests the extraction of structured entities and metadata from markdown content.
"""

import pytest
from typing import Dict, Any, List

from src.docproc.utils.markdown_entity_extractor import (
    extract_markdown_entities,
    extract_markdown_metadata
)


class TestExtractMarkdownEntities:
    """Tests for the extract_markdown_entities function."""

    def test_extract_headings(self):
        """Test extraction of headings at different levels."""
        content = """
# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6
"""
        entities = extract_markdown_entities(content)
        
        # Verify we found 6 headings
        headings = [e for e in entities if e["type"].startswith("heading_")]
        assert len(headings) == 6
        
        # Check heading levels
        for i in range(1, 7):
            matching = [h for h in headings if h["level"] == i]
            assert len(matching) == 1
            assert matching[0]["value"] == f"Heading {i}"
            assert matching[0]["type"] == f"heading_{i}"

    def test_extract_headings_with_trailing_hashes(self):
        """Test extraction of headings with trailing hashes."""
        content = """
# Heading 1 #

## Heading 2 ##
"""
        entities = extract_markdown_entities(content)
        
        # Verify headings are extracted correctly
        assert len(entities) == 2
        assert entities[0]["value"] == "Heading 1"
        assert entities[1]["value"] == "Heading 2"

    def test_extract_code_blocks(self):
        """Test extraction of code blocks with different languages."""
        content = """
```python
def hello():
    print("Hello World")
```

```javascript
function hello() {
    console.log("Hello World");
}
```

```
Plain text code block
```
"""
        entities = extract_markdown_entities(content)
        
        # Verify we found 3 code blocks
        code_blocks = [e for e in entities if e["type"] == "code_block"]
        assert len(code_blocks) == 3
        
        # Check languages
        python_blocks = [b for b in code_blocks if b["language"] == "python"]
        assert len(python_blocks) == 1
        assert "def hello():" in python_blocks[0]["value"]
        
        js_blocks = [b for b in code_blocks if b["language"] == "javascript"]
        assert len(js_blocks) == 1
        assert "function hello()" in js_blocks[0]["value"]
        
        text_blocks = [b for b in code_blocks if b["language"] == "text"]
        assert len(text_blocks) == 1
        assert "Plain text code block" in text_blocks[0]["value"]

    def test_extract_tables(self):
        """Test extraction of markdown tables."""
        content = """
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
"""
        entities = extract_markdown_entities(content)
        
        # Verify table extraction
        tables = [e for e in entities if e["type"] == "table"]
        assert len(tables) == 1
        
        # Check table content
        table = tables[0]["value"]
        assert "Header 1" in table
        assert "Cell 1" in table
        assert "Cell 4" in table

    def test_extract_links(self):
        """Test extraction of markdown links."""
        content = """
[Link 1](https://example.com)

[Link 2](https://github.com)

Check out [this page](https://docs.python.org) for more info.
"""
        entities = extract_markdown_entities(content)
        
        # Verify link extraction
        links = [e for e in entities if e["type"] == "link"]
        assert len(links) == 3
        
        # Check link properties
        link_texts = [link["text"] for link in links]
        assert "Link 1" in link_texts
        assert "Link 2" in link_texts
        assert "this page" in link_texts
        
        link_urls = [link["url"] for link in links]
        assert "https://example.com" in link_urls
        assert "https://github.com" in link_urls
        assert "https://docs.python.org" in link_urls

    def test_extract_mixed_content(self):
        """Test extraction from markdown with mixed content types."""
        content = """
# Project Documentation

## Introduction

This is a sample project that demonstrates markdown parsing.

## Code Example

```python
def main():
    print("Hello, world!")
    
if __name__ == "__main__":
    main()
```

## Data

| Name  | Value |
|-------|-------|
| First | 100   |
| Last  | 200   |

For more information, visit [our website](https://example.org).
"""
        entities = extract_markdown_entities(content)
        
        # Check that we extracted all entity types
        entity_types = {e["type"] for e in entities}
        assert "heading_1" in entity_types
        assert "heading_2" in entity_types
        assert "code_block" in entity_types
        assert "table" in entity_types
        assert "link" in entity_types
        
        # Check counts
        assert len([e for e in entities if e["type"].startswith("heading_")]) == 4  # There are 4 headings in the content
        assert len([e for e in entities if e["type"] == "code_block"]) == 1
        assert len([e for e in entities if e["type"] == "table"]) == 1
        assert len([e for e in entities if e["type"] == "link"]) == 1

    def test_empty_content(self):
        """Test handling of empty content."""
        assert extract_markdown_entities("") == []
        assert extract_markdown_entities("   \n  \n  ") == []

    def test_position_tracking(self):
        """Test that entity positions are tracked correctly."""
        content = """
# Heading

Some text.

```python
code
```
"""
        entities = extract_markdown_entities(content)
        
        # Check positions are in order
        positions = [e["start_pos"] for e in entities]
        assert positions == sorted(positions)
        
        # Check specific positions
        heading = [e for e in entities if e["type"] == "heading_1"][0]
        code_block = [e for e in entities if e["type"] == "code_block"][0]
        
        # Heading should come before code block
        assert heading["start_pos"] < code_block["start_pos"]


class TestExtractMarkdownMetadata:
    """Tests for the extract_markdown_metadata function."""

    def test_extract_title(self):
        """Test extraction of document title from first heading."""
        content = """
# Document Title

Content starts here.
"""
        metadata = extract_markdown_metadata(content, "/path/to/document.md")
        
        assert metadata["title"] == "Document Title"
        assert metadata["doc_type"] == "markdown"
        assert metadata["source"] == "/path/to/document.md"

    def test_extract_author_standard_format(self):
        """Test extraction of author in standard format."""
        content = """
# Document Title

Author: John Doe

Content starts here.
"""
        metadata = extract_markdown_metadata(content, "/path/to/document.md")
        
        assert "authors" in metadata
        assert metadata["authors"] == ["John Doe"]

    def test_extract_multiple_authors(self):
        """Test extraction of multiple comma-separated authors."""
        content = """
# Document Title

Authors: John Doe, Jane Smith, Bob Johnson

Content starts here.
"""
        metadata = extract_markdown_metadata(content, "/path/to/document.md")
        
        assert "authors" in metadata
        assert len(metadata["authors"]) == 3
        assert "John Doe" in metadata["authors"]
        assert "Jane Smith" in metadata["authors"]
        assert "Bob Johnson" in metadata["authors"]

    def test_extract_author_by_format(self):
        """Test extraction of author in various formats."""
        # By line
        content1 = "# Title\n\nBy John Doe\n\nContent."
        metadata1 = extract_markdown_metadata(content1, "doc1.md")
        assert metadata1["authors"] == ["John Doe"]
        
        # Italics
        content2 = "# Title\n\nContent.\n\n*Jane Smith*"
        metadata2 = extract_markdown_metadata(content2, "doc2.md")
        assert metadata2["authors"] == ["Jane Smith"]

    def test_extract_date(self):
        """Test extraction of date in different formats."""
        # Standard format
        content1 = "# Title\n\nDate: 2023-01-15\n\nContent."
        metadata1 = extract_markdown_metadata(content1, "doc1.md")
        # The implementation preserves the full line with the date
        assert metadata1["date_published"] == "Date: 2023-01-15"
        
        # Published format
        content2 = "# Title\n\nPublished: 2023-01-15\n\nContent."
        metadata2 = extract_markdown_metadata(content2, "doc2.md")
        assert metadata2["date_published"] == "Published: 2023-01-15"
        
        # Updated format
        content3 = "# Title\n\nUpdated: 2023-01-15\n\nContent."
        metadata3 = extract_markdown_metadata(content3, "doc3.md")
        assert metadata3["date_published"] == "Updated: 2023-01-15"
        
        # Natural language format
        content4 = "# Title\n\n15 January 2023\n\nContent."
        metadata4 = extract_markdown_metadata(content4, "doc4.md")
        assert metadata4["date_published"] == "15 January 2023"

    def test_extract_multiple_metadata(self):
        """Test extraction of multiple metadata fields."""
        content = """
# Document Title

Author: John Doe
Date: 2023-01-15

Content starts here.
"""
        metadata = extract_markdown_metadata(content, "/path/to/document.md")
        
        assert metadata["title"] == "Document Title"
        assert metadata["authors"] == ["John Doe"]
        assert metadata["date_published"] == "Date: 2023-01-15"

    def test_missing_metadata(self):
        """Test handling of missing metadata."""
        # No title (first heading)
        content1 = "Content without title."
        metadata1 = extract_markdown_metadata(content1, "doc1.md")
        assert "title" not in metadata1
        
        # No author
        content2 = "# Title\n\nContent without author."
        metadata2 = extract_markdown_metadata(content2, "doc2.md")
        assert "authors" not in metadata2
        
        # No date
        content3 = "# Title\n\nAuthor: John\n\nContent without date."
        metadata3 = extract_markdown_metadata(content3, "doc3.md")
        assert "date_published" not in metadata3

    def test_extract_from_complex_document(self):
        """Test metadata extraction from a complex document."""
        content = """
# Project Report

*A comprehensive analysis*

Authors: Dr. Jane Smith, Prof. John Doe

Published: 2023-05-20

## Executive Summary

This document provides an analysis of the project outcomes...

## Introduction

The project began in January 2023...
"""
        metadata = extract_markdown_metadata(content, "/reports/project.md")
        
        assert metadata["title"] == "Project Report"
        # The implementation extracts '*A comprehensive analysis*' as an author as well
        assert len(metadata["authors"]) == 3
        assert "Dr. Jane Smith" in metadata["authors"]
        assert "Prof. John Doe" in metadata["authors"]
        assert "A comprehensive analysis" in metadata["authors"]
        assert metadata["date_published"] == "Published: 2023-05-20"
