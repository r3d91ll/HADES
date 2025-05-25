#!/usr/bin/env python
"""
Script to run JSON serializer tests directly and measure code coverage.
"""

import sys
import os
import unittest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import importlib.util
import coverage

# Setup coverage
cov = coverage.Coverage(source=["src.docproc.serializers"])
cov.start()

# Manually import the JSON serializer module
json_serializer_path = str(Path(__file__).parents[3] / "src" / "docproc" / "serializers" / "json_serializer.py")
spec = importlib.util.spec_from_file_location("json_serializer", json_serializer_path)
json_serializer = importlib.util.module_from_spec(spec)
sys.modules["json_serializer"] = json_serializer
spec.loader.exec_module(json_serializer)

# Get functions from the module
serialize_to_json = json_serializer.serialize_to_json
save_to_json_file = json_serializer.save_to_json_file
_make_json_serializable = json_serializer._make_json_serializable


class TestJsonSerializerCoverage(unittest.TestCase):
    """Test cases for JSON serialization functions."""
    
    def test_make_json_serializable_simple_types(self):
        """Test _make_json_serializable with simple types."""
        # Simple types should be unchanged
        self.assertEqual(_make_json_serializable("string"), "string")
        self.assertEqual(_make_json_serializable(42), 42)
        self.assertEqual(_make_json_serializable(3.14), 3.14)
        self.assertEqual(_make_json_serializable(True), True)
        self.assertEqual(_make_json_serializable(None), None)
    
    def test_make_json_serializable_collections(self):
        """Test _make_json_serializable with collection types."""
        # Lists should be processed recursively
        self.assertEqual(_make_json_serializable([1, 2, "three"]), [1, 2, "three"])
        
        # Tuples should be converted to lists
        self.assertEqual(_make_json_serializable((1, 2, "three")), [1, 2, "three"])
        
        # Sets should be converted to lists
        result = _make_json_serializable({1, 2, 3})
        self.assertIsInstance(result, list)
        self.assertEqual(set(result), {1, 2, 3})
        
        # Dictionaries should be processed recursively
        self.assertEqual(
            _make_json_serializable({"a": 1, "b": [2, 3]}),
            {"a": 1, "b": [2, 3]}
        )
    
    def test_make_json_serializable_custom_objects(self):
        """Test _make_json_serializable with custom objects."""
        class TestObject:
            def __init__(self):
                self.a = 1
                self.b = "two"
        
        # Custom objects should be converted to dictionaries
        serialized = _make_json_serializable(TestObject())
        self.assertEqual(serialized["a"], 1)
        self.assertEqual(serialized["b"], "two")
    
    def test_make_json_serializable_complex_types(self):
        """Test _make_json_serializable with complex types."""
        # Non-serializable types should be converted to strings
        dt = datetime.now()
        self.assertTrue(isinstance(_make_json_serializable(dt), str))
    
    def test_serialize_to_json_basic(self):
        """Test basic serialization with minimal input."""
        result = {"content": "test content"}
        serialized = serialize_to_json(result)
        
        # Basic content should be preserved
        self.assertEqual(serialized["content"], "test content")
        
        # Version and timestamp should be added by default
        self.assertTrue("version" in serialized)
        self.assertTrue("timestamp" in serialized)
    
    def test_serialize_to_json_with_entities(self):
        """Test serialization with entities."""
        result = {
            "content": "test content",
            "entities": [
                {"name": "Entity1", "type": "Class", "line": 10},
                {"name": "Entity2", "type": "Function", "line": 20}
            ]
        }
        
        serialized = serialize_to_json(result)
        
        # Entities should be preserved
        self.assertEqual(len(serialized["entities"]), 2)
        self.assertEqual(serialized["entities"][0]["name"], "Entity1")
        self.assertEqual(serialized["entities"][1]["type"], "Function")
    
    def test_serialize_to_json_with_format(self):
        """Test serialization with format information."""
        result = {
            "content": "test content", 
            "format": "python"
        }
        
        serialized = serialize_to_json(result)
        
        # Format should be preserved
        self.assertEqual(serialized["format"], "python")
    
    def test_serialize_to_json_with_metadata(self):
        """Test serialization with metadata."""
        result = {
            "content": "test content",
            "metadata": {
                "filename": "test.py",
                "size": 1024,
                "author": "Test User"
            }
        }
        
        serialized = serialize_to_json(result)
        
        # Metadata should be preserved
        self.assertEqual(serialized["metadata"]["filename"], "test.py")
        self.assertEqual(serialized["metadata"]["size"], 1024)
    
    def test_serialize_to_json_with_processing_time(self):
        """Test serialization with processing time."""
        result = {
            "content": "test content",
            "processing_time": 1.23
        }
        
        serialized = serialize_to_json(result)
        
        # Processing time should be moved to metadata
        self.assertEqual(serialized["metadata"]["processing_time"], 1.23)
    
    def test_serialize_to_json_without_optional_fields(self):
        """Test serialization with optional fields disabled."""
        result = {"content": "test content"}
        
        serialized = serialize_to_json(
            result,
            include_metadata=False,
            include_timestamp=False,
            include_version=False
        )
        
        # Optional fields should be excluded
        self.assertNotIn("metadata", serialized)
        self.assertNotIn("timestamp", serialized)
        self.assertNotIn("version", serialized)
    
    def test_serialize_to_json_with_custom_version(self):
        """Test serialization with custom version."""
        result = {"content": "test content"}
        
        serialized = serialize_to_json(result, version="2.0.0")
        
        # Custom version should be used
        self.assertEqual(serialized["version"], "2.0.0")
    
    def test_serialize_to_json_with_extra_fields(self):
        """Test serialization with extra non-standard fields."""
        result = {
            "content": "test content",
            "extra_field1": "value1",
            "extra_field2": 42
        }
        
        serialized = serialize_to_json(result)
        
        # Extra fields should be preserved
        self.assertEqual(serialized["extra_field1"], "value1")
        self.assertEqual(serialized["extra_field2"], 42)
    
    def test_serialize_with_non_dict_metadata(self):
        """Test serialization with non-dictionary metadata."""
        result = {
            "content": "test content",
            "metadata": "string metadata"
        }
        
        serialized = serialize_to_json(result)
        
        # Non-dict metadata should not be included in the result
        self.assertNotIn("metadata", serialized)
        
        # Optional fields should still be included by default
        self.assertTrue("version" in serialized)
        self.assertTrue("timestamp" in serialized)
    
    def test_serialize_with_id_and_source(self):
        """Test serialization with id and source fields."""
        result = {
            "id": "doc123",
            "source": "/path/to/doc.txt",
            "content": "test content"
        }
        
        serialized = serialize_to_json(result)
        
        # ID and source should be preserved
        self.assertEqual(serialized["id"], "doc123")
        self.assertEqual(serialized["source"], "/path/to/doc.txt")
    
    def test_serialize_with_content_type(self):
        """Test serialization with content_type field."""
        result = {
            "content": "test content",
            "content_type": "code"
        }
        
        serialized = serialize_to_json(result)
        
        # Content type should be preserved
        self.assertEqual(serialized["content_type"], "code")
    
    def test_save_to_json_file(self):
        """Test saving serialized results to a file."""
        result = {
            "content": "test content",
            "format": "python",
            "entities": [{"name": "TestEntity", "type": "Class"}]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Save to file
            saved_path = save_to_json_file(result, output_path)
            
            # File should exist
            self.assertTrue(Path(saved_path).exists())
            
            # Content should be valid JSON
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                
            # Verify content
            self.assertEqual(loaded["content"], "test content")
            self.assertEqual(loaded["format"], "python")
            self.assertEqual(loaded["entities"][0]["name"], "TestEntity")
    
    def test_save_to_json_file_without_pretty_print(self):
        """Test saving serialized results to a file without pretty printing."""
        result = {"content": "test content"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Save to file without pretty printing
            saved_path = save_to_json_file(result, output_path, pretty_print=False)
            
            # File should exist
            self.assertTrue(Path(saved_path).exists())
            
            # Content should be valid JSON
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                
            # Verify content
            self.assertEqual(loaded["content"], "test content")
    
    def test_save_to_json_file_string_path(self):
        """Test saving serialized results to a file using a string path."""
        result = {"content": "test content"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.json")
            
            # Save to file using string path
            saved_path = save_to_json_file(result, output_path)
            
            # File should exist
            self.assertTrue(os.path.exists(saved_path))
            
            # Content should be valid JSON
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                
            # Verify content
            self.assertEqual(loaded["content"], "test content")


if __name__ == "__main__":
    # Run the tests
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestJsonSerializerCoverage)
    test_runner = unittest.TextTestRunner()
    test_result = test_runner.run(test_suite)
    
    # Stop coverage and save report
    cov.stop()
    cov.save()
    
    # Generate coverage report
    total_coverage = cov.report()
    
    print(f"\nTotal coverage: {total_coverage:.2f}%")
    
    # Exit with appropriate status code
    sys.exit(0 if test_result.wasSuccessful() else 1)
