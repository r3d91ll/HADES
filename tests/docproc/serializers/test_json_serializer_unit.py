"""
Unit tests for the JSON serializer functions.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Manually import the JSON serializer module without dependencies
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.docproc.serializers.json_serializer import (
    serialize_to_json,
    save_to_json_file,
    _make_json_serializable
)


class TestJsonSerializerUnit(unittest.TestCase):
    """Unit tests for the JSON serialization functions."""
    
    def test_serialize_basic(self):
        """Test basic serialization with minimal input."""
        result = serialize_to_json({"content": "test content"})
        self.assertEqual(result["content"], "test content")
        self.assertIn("version", result)
        self.assertIn("timestamp", result)
    
    def test_serialize_with_all_fields(self):
        """Test serialization with all possible fields."""
        input_data = {
            "id": "doc123",
            "source": "/path/to/file.txt",
            "format": "text",
            "content_type": "text",
            "content": "Test content",
            "entities": [{"name": "Entity1", "type": "Type1"}],
            "processing_time": 1.5,
            "metadata": {"author": "Test Author", "date": "2023-01-01"},
            "custom_field": "custom value"
        }
        
        result = serialize_to_json(input_data)
        
        # Check core fields are present
        self.assertEqual(result["id"], "doc123")
        self.assertEqual(result["source"], "/path/to/file.txt")
        self.assertEqual(result["format"], "text")
        self.assertEqual(result["content_type"], "text")
        self.assertEqual(result["content"], "Test content")
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["name"], "Entity1")
        self.assertEqual(result["metadata"]["author"], "Test Author")
        self.assertEqual(result["metadata"]["processing_time"], 1.5)
        self.assertEqual(result["custom_field"], "custom value")
    
    def test_serialize_without_optional_fields(self):
        """Test serialization with optional fields disabled."""
        input_data = {"content": "test content"}
        
        result = serialize_to_json(
            input_data,
            include_metadata=False,
            include_timestamp=False,
            include_version=False
        )
        
        self.assertEqual(result["content"], "test content")
        self.assertNotIn("metadata", result)
        self.assertNotIn("timestamp", result)
        self.assertNotIn("version", result)
    
    def test_json_serializable_conversion(self):
        """Test conversion of various types to JSON serializable formats."""
        # Simple types
        self.assertEqual(_make_json_serializable("string"), "string")
        self.assertEqual(_make_json_serializable(42), 42)
        self.assertEqual(_make_json_serializable(3.14), 3.14)
        self.assertEqual(_make_json_serializable(True), True)
        self.assertEqual(_make_json_serializable(None), None)
        
        # Collections
        self.assertEqual(_make_json_serializable([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_make_json_serializable((1, 2, 3)), [1, 2, 3])
        self.assertTrue(isinstance(_make_json_serializable({1, 2, 3}), list))
        
        # Complex types
        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = "two"
        
        obj_result = _make_json_serializable(TestObj())
        self.assertEqual(obj_result["a"], 1)
        self.assertEqual(obj_result["b"], "two")
        
        # Non-serializable types should be stringified
        dt = datetime.now()
        self.assertTrue(isinstance(_make_json_serializable(dt), str))
    
    def test_save_to_json_file(self):
        """Test saving serialized data to a JSON file."""
        input_data = {
            "id": "doc123",
            "content": "test content",
            "format": "text"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Path object
            path_obj = Path(temp_dir) / "test1.json"
            result_path = save_to_json_file(input_data, path_obj)
            
            self.assertTrue(os.path.exists(result_path))
            with open(result_path, 'r') as f:
                loaded = json.load(f)
                self.assertEqual(loaded["id"], "doc123")
                self.assertEqual(loaded["content"], "test content")
            
            # Test with string path
            str_path = os.path.join(temp_dir, "test2.json")
            result_path = save_to_json_file(input_data, str_path)
            
            self.assertTrue(os.path.exists(result_path))
            with open(result_path, 'r') as f:
                loaded = json.load(f)
                self.assertEqual(loaded["id"], "doc123")
            
            # Test without pretty print
            no_pretty_path = os.path.join(temp_dir, "test3.json")
            result_path = save_to_json_file(input_data, no_pretty_path, pretty_print=False)
            
            self.assertTrue(os.path.exists(result_path))
            with open(result_path, 'r') as f:
                loaded = json.load(f)
                self.assertEqual(loaded["id"], "doc123")


if __name__ == "__main__":
    unittest.main()
