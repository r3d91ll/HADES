"""
Unit tests for JSON serializer module.

This module tests the JSON serialization functionality for document processing results.
"""

import json
import pytest
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, mock_open

from src.docproc.serializers.json_serializer import (
    _make_json_serializable,
    serialize_to_json,
    save_to_json_file
)


class TestMakeJsonSerializable:
    """Tests for the _make_json_serializable function."""

    def test_basic_types(self):
        """Test serialization of basic types (strings, numbers, booleans, None)."""
        # Strings
        assert _make_json_serializable("test") == "test"
        # Numbers
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable(3.14) == 3.14
        # Booleans
        assert _make_json_serializable(True) is True
        assert _make_json_serializable(False) is False
        # None
        assert _make_json_serializable(None) is None

    def test_collections(self):
        """Test serialization of collections (lists, tuples, sets)."""
        # List
        assert _make_json_serializable([1, 2, 3]) == [1, 2, 3]
        # Tuple
        assert _make_json_serializable((1, 2, 3)) == [1, 2, 3]
        # Set
        assert _make_json_serializable({1, 2, 3}) == [1, 2, 3]
        # Nested collections
        assert _make_json_serializable([1, (2, 3), {4, 5}]) == [1, [2, 3], [4, 5]]

    def test_dictionaries(self):
        """Test serialization of dictionaries."""
        # Simple dictionary
        assert _make_json_serializable({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        # Dictionary with non-string keys
        assert _make_json_serializable({1: "a", 2: "b"}) == {"1": "a", "2": "b"}
        # Nested dictionary
        nested = {"a": {"b": [1, 2, {"c": 3}]}}
        assert _make_json_serializable(nested) == {"a": {"b": [1, 2, {"c": 3}]}}

    def test_custom_objects(self):
        """Test serialization of custom objects with __dict__ attribute."""
        class TestObj:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = TestObj()
        result = _make_json_serializable(obj)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_non_serializable_objects(self):
        """Test conversion of non-serializable objects to strings."""
        # Complex number
        complex_num = complex(1, 2)
        assert _make_json_serializable(complex_num) == str(complex_num)
        
        # Custom object without __dict__
        class CustomObj:
            __slots__ = ["value"]
            def __init__(self, value):
                self.value = value
            def __str__(self):
                return f"CustomObj({self.value})"
        
        obj = CustomObj(42)
        assert _make_json_serializable(obj) == str(obj)

    def test_mixed_complex_structure(self):
        """Test serialization of complex nested structures with mixed types."""
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
        
        data = {
            "people": [
                Person("Alice", 30),
                Person("Bob", 25)
            ],
            "locations": {1: "New York", 2: "London"},
            "active": True,
            "metadata": {
                "created": datetime(2023, 1, 1),
                "tags": {"important", "verified"}
            }
        }
        
        result = _make_json_serializable(data)
        
        # Check structure is preserved
        assert "people" in result
        assert isinstance(result["people"], list)
        assert len(result["people"]) == 2
        assert result["people"][0]["name"] == "Alice"
        assert result["people"][0]["age"] == 30
        assert result["people"][1]["name"] == "Bob"
        
        # Check non-string keys are converted
        assert "1" in result["locations"]
        assert result["locations"]["1"] == "New York"
        
        # Check sets are converted to lists
        assert isinstance(result["metadata"]["tags"], list)
        assert set(result["metadata"]["tags"]) == {"important", "verified"}
        
        # Check datetime is converted to string
        assert isinstance(result["metadata"]["created"], str)


class TestSerializeToJson:
    """Tests for the serialize_to_json function."""

    def test_basic_serialization(self):
        """Test basic serialization with minimal document."""
        input_doc = {
            "id": "doc123",
            "source": "test.txt",
            "format": "text",
            "content": "This is test content."
        }
        
        result = serialize_to_json(input_doc)
        
        # Check core fields
        assert result["id"] == "doc123"
        assert result["source"] == "test.txt"
        assert result["format"] == "text"
        assert result["content"] == "This is test content."
        
        # Check default fields
        assert "version" in result
        assert "timestamp" in result

    def test_with_metadata(self):
        """Test serialization with document metadata."""
        input_doc = {
            "id": "doc123",
            "content": "Test content",
            "metadata": {
                "title": "Test Document",
                "author": "Test Author",
                "created_date": "2023-01-01"
            }
        }
        
        result = serialize_to_json(input_doc, include_metadata=True)
        
        # Check metadata is included
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Document"
        assert result["metadata"]["author"] == "Test Author"
        
    def test_without_metadata(self):
        """Test serialization with metadata excluded."""
        input_doc = {
            "id": "doc123",
            "content": "Test content",
            "metadata": {
                "title": "Test Document",
                "author": "Test Author"
            }
        }
        
        result = serialize_to_json(input_doc, include_metadata=False)
        
        # Check metadata is excluded
        assert "metadata" not in result

    def test_with_entities(self):
        """Test serialization with document entities."""
        input_doc = {
            "id": "doc123",
            "content": "# Heading\n\nThis is a paragraph.",
            "entities": [
                {"type": "heading", "value": "Heading", "level": 1},
                {"type": "paragraph", "value": "This is a paragraph."}
            ]
        }
        
        result = serialize_to_json(input_doc)
        
        # Check entities are included
        assert "entities" in result
        assert len(result["entities"]) == 2
        assert result["entities"][0]["type"] == "heading"
        assert result["entities"][1]["type"] == "paragraph"

    def test_timestamp_and_version(self):
        """Test timestamp and version inclusion options."""
        input_doc = {"id": "doc123", "content": "Test"}
        
        # Test with timestamp and version
        result1 = serialize_to_json(
            input_doc, 
            include_timestamp=True,
            include_version=True,
            version="2.0.0"
        )
        assert "timestamp" in result1
        assert "version" in result1
        assert result1["version"] == "2.0.0"
        
        # Test without timestamp
        result2 = serialize_to_json(input_doc, include_timestamp=False)
        assert "timestamp" not in result2
        
        # Test without version
        result3 = serialize_to_json(input_doc, include_version=False)
        assert "version" not in result3

    def test_complex_metadata_serialization(self):
        """Test serialization with complex non-serializable metadata."""
        class CustomMetadata:
            def __init__(self):
                self.value = 42
        
        input_doc = {
            "id": "doc123",
            "content": "Test content",
            "metadata": {
                "custom": CustomMetadata(),
                "dates": {datetime(2023, 1, 1): "New Year"},
                "tags": {"important", "urgent"}
            }
        }
        
        result = serialize_to_json(input_doc)
        
        # Check complex metadata is properly serialized
        assert "metadata" in result
        assert isinstance(result["metadata"]["custom"], dict)
        assert result["metadata"]["custom"]["value"] == 42
        assert isinstance(result["metadata"]["tags"], list)

    def test_field_ordering(self):
        """Test that fields are ordered correctly in output."""
        input_doc = {
            "content": "This should come last",
            "format": "text",
            "id": "doc123",
            "tags": ["test"],
            "metadata": {"author": "Test"}
        }
        
        result = serialize_to_json(input_doc)
        
        # Convert to JSON string to check serialization order
        json_str = json.dumps(result)
        
        # Check core ID fields come before content
        assert json_str.find('"id"') < json_str.find('"content"')
        # Check metadata comes before content
        assert json_str.find('"metadata"') < json_str.find('"content"')

    def test_processing_time_in_metadata(self):
        """Test that processing_time is moved to metadata."""
        input_doc = {
            "id": "doc123",
            "content": "Test content",
            "processing_time": 0.42
        }
        
        result = serialize_to_json(input_doc)
        
        # Check processing_time is moved to metadata
        assert "metadata" in result
        assert result["metadata"]["processing_time"] == 0.42
        assert "processing_time" not in result  # Not in root

    def test_content_type_handling(self):
        """Test that content_type is correctly preserved."""
        input_doc = {
            "id": "doc123",
            "content": "# Markdown content",
            "content_type": "text/markdown"
        }
        
        result = serialize_to_json(input_doc)
        
        # Check content type is preserved
        assert result["content_type"] == "text/markdown"


class TestSaveToJsonFile:
    """Tests for the save_to_json_file function."""

    def test_save_with_pretty_print(self):
        """Test saving JSON with pretty print formatting."""
        input_doc = {
            "id": "doc123",
            "content": "Test content"
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_output.json")
            
            with patch("builtins.open", mock_open()) as mock_file, \
                 patch("json.dump") as mock_dump:
                save_to_json_file(input_doc, output_path)
                
                # Check file was opened for writing - path is converted to Path object inside
                mock_file.assert_called_once()
                # Get the path from the call
                call_args = mock_file.call_args
                assert call_args is not None
                # Verify right mode and encoding
                assert call_args[0][1] == 'w'
                assert call_args[1]['encoding'] == 'utf-8'
                
                # Check json.dump was called
                mock_dump.assert_called_once()

    def test_save_without_pretty_print(self):
        """Test saving JSON without pretty print formatting."""
        input_doc = {
            "id": "doc123",
            "content": "Test content"
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_output.json")
            
            with patch("json.dump") as mock_dump:
                save_to_json_file(input_doc, output_path, pretty_print=False)
                
                # Check json.dump was called
                assert mock_dump.called

    def test_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        input_doc = {"id": "doc123", "content": "Test"}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a path with nested directories
            output_path = os.path.join(tmp_dir, "nested", "dir", "test.json")
            
            with patch("pathlib.Path.mkdir") as mock_mkdir, \
                 patch("builtins.open", mock_open()), \
                 patch("json.dump"):
                
                save_to_json_file(input_doc, output_path)
                
                # Check mkdir was called with parents=True
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_path_object_handling(self):
        """Test that Path objects are handled properly."""
        input_doc = {"id": "doc123", "content": "Test"}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use a Path object
            output_path = Path(tmp_dir) / "test.json"
            
            with patch("builtins.open", mock_open()) as mock_file, \
                 patch("json.dump"):
                
                save_to_json_file(input_doc, output_path)
                
                # Check file was opened with the correct path
                mock_file.assert_called_once()
                file_path = mock_file.call_args[0][0]
                assert str(output_path) in str(file_path)

    def test_metadata_version_timestamp_options(self):
        """Test that options for metadata, version, and timestamp are passed to serialize_to_json."""
        input_doc = {"id": "doc123", "content": "Test"}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.json")
            
            with patch("src.docproc.serializers.json_serializer.serialize_to_json") as mock_serialize, \
                 patch("builtins.open", mock_open()), \
                 patch("json.dump"):
                
                save_to_json_file(
                    input_doc,
                    output_path,
                    include_metadata=False,
                    include_timestamp=False,
                    include_version=False,
                    version="3.0.0"
                )
                
                # Check serialize_to_json was called with the correct options
                mock_serialize.assert_called_once_with(
                    input_doc,
                    include_metadata=False,
                    include_timestamp=False,
                    include_version=False,
                    version="3.0.0"
                )

    def test_return_value(self):
        """Test that the function returns the absolute path to the saved file."""
        input_doc = {"id": "doc123", "content": "Test"}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.json")
            
            with patch("builtins.open", mock_open()), \
                 patch("json.dump"):
                
                result = save_to_json_file(input_doc, output_path)
                
                # Check return value is absolute path
                assert os.path.isabs(result)
                assert output_path in result
