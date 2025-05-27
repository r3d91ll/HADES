#!/usr/bin/env python3
"""
Test suite for verifying the document processing adapters after type fixes.

This test suite ensures that the YAML, JSON, and Markdown adapters still
function correctly after the type annotation fixes.
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.docproc.adapters.yaml_adapter import YAMLAdapter
from src.docproc.adapters.json_adapter import JSONAdapter
from src.docproc.adapters.markdown_adapter import MarkdownAdapter


class TestYAMLAdapter(unittest.TestCase):
    """Test case for YAML adapter functionality after type fixes."""
    
    def setUp(self) -> None:
        """Set up the test case."""
        self.adapter = YAMLAdapter()
        
        # Create a test YAML file
        self.test_file = Path(__file__).parent / "test_data" / "test_yaml.yaml"
        self.test_file.parent.mkdir(exist_ok=True)
        
        # Simple YAML content for testing
        yaml_content = """
        key1: value1
        key2: 
          nested1: nestedvalue1
          nested2: nestedvalue2
        list_key:
          - item1
          - item2
        """
        
        with open(self.test_file, "w") as f:
            f.write(yaml_content)
    
    def tearDown(self) -> None:
        """Clean up after test."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_process_yaml(self) -> None:
        """Test YAML processing with type-fixed adapter."""
        result = self.adapter.process(self.test_file)
        
        # Verify the result structure
        self.assertIn("symbol_table", result)
        self.assertIn("relationships", result)
        self.assertIn("metadata", result)
        
        # Check if we can access the expected data
        self.assertGreater(len(result["symbol_table"]), 0)
        
        # Verify metadata counts are correct
        self.assertEqual(result["metadata"]["element_count"], len(result["symbol_table"]))
        self.assertEqual(result["metadata"]["relationship_count"], len(result["relationships"]))


class TestJSONAdapter(unittest.TestCase):
    """Test case for JSON adapter functionality after type fixes."""
    
    def setUp(self) -> None:
        """Set up the test case."""
        self.adapter = JSONAdapter()
        
        # Create a test JSON file
        self.test_file = Path(__file__).parent / "test_data" / "test_json.json"
        self.test_file.parent.mkdir(exist_ok=True)
        
        # Simple JSON content for testing
        json_content = """
        {
            "key1": "value1",
            "key2": {
                "nested1": "nestedvalue1",
                "nested2": "nestedvalue2"
            },
            "list_key": [
                "item1",
                "item2"
            ]
        }
        """
        
        with open(self.test_file, "w") as f:
            f.write(json_content)
    
    def tearDown(self) -> None:
        """Clean up after test."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_process_json(self) -> None:
        """Test JSON processing with type-fixed adapter."""
        result = self.adapter.process(self.test_file)
        
        # Verify the result structure
        self.assertIn("symbol_table", result)
        self.assertIn("relationships", result)
        self.assertIn("metadata", result)
        
        # Check if we can access the expected data
        self.assertGreater(len(result["symbol_table"]), 0)
        
        # Verify metadata counts are correct
        self.assertEqual(result["metadata"]["element_count"], len(result["symbol_table"]))
        self.assertEqual(result["metadata"]["relationship_count"], len(result["relationships"]))


class TestMarkdownAdapter(unittest.TestCase):
    """Test case for Markdown adapter functionality after type fixes."""
    
    def setUp(self) -> None:
        """Set up the test case."""
        self.adapter = MarkdownAdapter()
        
        # Create a test Markdown file
        self.test_file = Path(__file__).parent / "test_data" / "test_markdown.md"
        self.test_file.parent.mkdir(exist_ok=True)
        
        # Simple Markdown content for testing
        md_content = """
        # Test Heading
        
        This is a test paragraph.
        
        ## Subheading
        
        - List item 1
        - List item 2
        
        ### Code Example
        
        ```python
        def example_function():
            return "Hello World"
        ```
        """
        
        with open(self.test_file, "w") as f:
            f.write(md_content)
    
    def tearDown(self) -> None:
        """Clean up after test."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_process_markdown(self) -> None:
        """Test Markdown processing with type-fixed adapter."""
        result = self.adapter.process(self.test_file)
        
        # Verify the result structure
        self.assertIn("content", result)
        self.assertIn("parsed_content", result)
        self.assertIn("format", result)
        self.assertIn("metadata", result)
        
        # Check that the content is processed correctly
        self.assertEqual(result["format"], "markdown")
        self.assertIn("doc_type", result["metadata"])
        self.assertEqual(result["metadata"]["doc_type"], "markdown")


if __name__ == "__main__":
    unittest.main()
