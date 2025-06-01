"""
Tests for the Python code adapter module.

This module contains comprehensive tests for the PythonCodeAdapter class,
ensuring proper code analysis, entity extraction, and metadata generation.
"""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from typing import Dict, Any, List, Optional, Set

from src.docproc.adapters.python_code_adapter import PythonCodeAdapter


class TestPythonCodeAdapter(unittest.TestCase):
    """Test cases for the PythonCodeAdapter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = PythonCodeAdapter()
        
        # Sample Python code for testing
        self.sample_code = '''"""
Module docstring for test code.
"""

import os
import sys
from typing import List, Dict

def hello(name: str) -> str:
    """Say hello to someone.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}!"

class Person:
    """A simple person class."""
    
    def __init__(self, name: str, age: int):
        """Initialize a person.
        
        Args:
            name: The person's name
            age: The person's age
        """
        self.name = name
        self.age = age
        
    def greet(self) -> str:
        """Generate a greeting.
        
        Returns:
            A greeting message
        """
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
'''

    def test_init_with_default_options(self):
        """Test initialization with default options."""
        adapter = PythonCodeAdapter()
        self.assertIsNotNone(adapter)
        self.assertIsNotNone(adapter.python_adapter)
        self.assertEqual({}, adapter.options)

    def test_init_with_custom_options(self):
        """Test initialization with custom options."""
        options = {"custom_option": "value"}
        adapter = PythonCodeAdapter(options)
        self.assertEqual(options, adapter.options)

    def test_process_valid_file(self):
        """Test processing a valid Python file."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as tmp:
            tmp.write(self.sample_code)
            tmp.flush()
            
            result = self.adapter.process(tmp.name)
            
            # Check basic structure
            self.assertIsInstance(result, dict)
            self.assertEqual(str(Path(tmp.name)), result["source"])
            self.assertEqual(self.sample_code, result["content"])
            self.assertEqual("python", result["format"])
            self.assertIsNotNone(result["id"])
            self.assertIsNone(result["error"])
            
            # Check metadata
            self.assertIn("metadata", result)
            self.assertEqual(str(Path(tmp.name)), result["metadata"]["path"])
            self.assertEqual(Path(tmp.name).name, result["metadata"]["filename"])
            self.assertEqual(".py", result["metadata"]["extension"])
            self.assertEqual("python", result["metadata"]["language"])
            
            # Check code analysis
            self.assertIn("code_analysis", result["metadata"])
            
    def test_process_nonexistent_file(self):
        """Test processing a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.process("/path/to/nonexistent/file.py")
            
    def test_process_non_python_file(self):
        """Test processing a non-Python file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as tmp:
            tmp.write("This is not a Python file")
            tmp.flush()
            
            with self.assertRaises(ValueError):
                self.adapter.process(tmp.name)

    def test_process_text(self):
        """Test processing Python code text content."""
        result = self.adapter.process_text(self.sample_code)
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertEqual("text_content", result["source"])
        self.assertEqual(self.sample_code, result["content"])
        self.assertEqual("python", result["format"])
        self.assertIsNotNone(result["id"])
        self.assertIsNone(result["error"])
        
        # Check metadata
        self.assertIn("metadata", result)
        self.assertEqual("python", result["metadata"]["language"])
        
        # Check code analysis
        self.assertIn("code_analysis", result["metadata"])
        
    def test_extract_entities_from_text(self):
        """Test extracting entities from Python code text."""
        entities = self.adapter.extract_entities(self.sample_code)
        
        # Check that we have entities
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Find specific entities
        module_entity = next((e for e in entities if e["type"] == "module"), None)
        function_entity = next((e for e in entities if e["type"] == "function" and e["name"] == "hello"), None)
        class_entity = next((e for e in entities if e["type"] == "class" and e["name"] == "Person"), None)
        
        # Check module entity
        self.assertIsNotNone(module_entity)
        self.assertIn("Module docstring for test code", module_entity["content"])
        
        # Check function entity
        self.assertIsNotNone(function_entity)
        self.assertEqual("hello", function_entity["name"])
        self.assertIn("Say hello to someone", function_entity["content"])
        
        # Check class entity
        self.assertIsNotNone(class_entity)
        self.assertEqual("Person", class_entity["name"])
        self.assertIn("A simple person class", class_entity["content"])
        
    def test_extract_entities_from_dict(self):
        """Test extracting entities from already processed data."""
        # First process the text
        processed = self.adapter.process_text(self.sample_code)
        
        # Then extract entities from the processed data
        entities = self.adapter.extract_entities(processed)
        
        # Verify entities
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        function_entity = next((e for e in entities if e["type"] == "function" and e["name"] == "hello"), None)
        self.assertIsNotNone(function_entity)
        
    def test_extract_metadata_from_text(self):
        """Test extracting metadata from Python code text."""
        metadata = self.adapter.extract_metadata(self.sample_code)
        
        # Check basic metadata
        self.assertIsInstance(metadata, dict)
        self.assertEqual("python", metadata["language"])
        self.assertEqual("code", metadata["document_type"])
        self.assertEqual("python", metadata["code_type"])
        
        # Check counts
        self.assertEqual("1", metadata["function_count"])
        self.assertEqual("1", metadata["class_count"])
        self.assertIn("import_count", metadata)
        
    def test_extract_metadata_from_dict(self):
        """Test extracting metadata from already processed data."""
        # First process the text
        processed = self.adapter.process_text(self.sample_code)
        
        # Then extract metadata from the processed data
        metadata = self.adapter.extract_metadata(processed)
        
        # Verify metadata
        self.assertIsInstance(metadata, dict)
        self.assertEqual("python", metadata["language"])
        self.assertEqual("1", metadata["function_count"])
        self.assertEqual("1", metadata["class_count"])


if __name__ == "__main__":
    unittest.main()
