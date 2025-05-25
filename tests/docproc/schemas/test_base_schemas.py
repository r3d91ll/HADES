"""
Unit tests for the base document schemas.

This module contains tests for the base document schemas that are used
across all document types.
"""

import unittest
from pydantic import ValidationError

from src.docproc.schemas.base import BaseDocument, BaseEntity, BaseMetadata


class TestBaseEntity(unittest.TestCase):
    """Test the BaseEntity model."""
    
    def test_valid_entity(self):
        """Test that a valid entity passes validation."""
        entity = BaseEntity(
            type="test",
            value="test value",
            line=10,
            confidence=0.8
        )
        
        self.assertEqual(entity.type, "test")
        self.assertEqual(entity.value, "test value")
        self.assertEqual(entity.line, 10)
        self.assertEqual(entity.confidence, 0.8)
    
    def test_minimal_entity(self):
        """Test that an entity with only required fields passes validation."""
        entity = BaseEntity(
            type="test",
            value="test value"
        )
        
        self.assertEqual(entity.type, "test")
        self.assertEqual(entity.value, "test value")
        self.assertIsNone(entity.line)
        self.assertEqual(entity.confidence, 1.0)  # Default value
    
    def test_invalid_confidence(self):
        """Test that an entity with invalid confidence fails validation."""
        with self.assertRaises(ValidationError):
            BaseEntity(
                type="test",
                value="test value",
                confidence=1.5  # Invalid: should be 0-1
            )
        
        with self.assertRaises(ValidationError):
            BaseEntity(
                type="test",
                value="test value",
                confidence=-0.5  # Invalid: should be 0-1
            )
    
    def test_extra_fields(self):
        """Test that extra fields are allowed and preserved."""
        entity = BaseEntity(
            type="test",
            value="test value",
            extra_field="extra value"
        )
        
        self.assertEqual(entity.extra_field, "extra value")  # Access via attribute
        self.assertEqual(entity.model_dump()["extra_field"], "extra value")  # Access via dict


class TestBaseMetadata(unittest.TestCase):
    """Test the BaseMetadata model."""
    
    def test_valid_metadata(self):
        """Test that valid metadata passes validation."""
        metadata = BaseMetadata(
            language="python",
            format="python",
            content_type="code",
            file_size=1024,
            line_count=100,
            char_count=5000,
            has_errors=False
        )
        
        self.assertEqual(metadata.language, "python")
        self.assertEqual(metadata.format, "python")
        self.assertEqual(metadata.content_type, "code")
        self.assertEqual(metadata.file_size, 1024)
        self.assertEqual(metadata.line_count, 100)
        self.assertEqual(metadata.char_count, 5000)
        self.assertFalse(metadata.has_errors)
    
    def test_minimal_metadata(self):
        """Test that metadata with only required fields passes validation."""
        metadata = BaseMetadata(
            language="python",
            format="python",
            content_type="code"
        )
        
        self.assertEqual(metadata.language, "python")
        self.assertEqual(metadata.format, "python")
        self.assertEqual(metadata.content_type, "code")
        self.assertIsNone(metadata.file_size)
        self.assertIsNone(metadata.line_count)
        self.assertFalse(metadata.has_errors)  # Default value
    
    def test_missing_required_fields(self):
        """Test that metadata without required fields fails validation."""
        with self.assertRaises(ValidationError):
            BaseMetadata(
                language="python",
                # Missing format
                content_type="code"
            )
        
        with self.assertRaises(ValidationError):
            BaseMetadata(
                # Missing language
                format="python",
                content_type="code"
            )
    
    def test_extra_fields(self):
        """Test that extra fields are allowed and preserved."""
        metadata = BaseMetadata(
            language="python",
            format="python",
            content_type="code",
            custom_field="custom value"
        )
        
        self.assertEqual(metadata.custom_field, "custom value")  # Access via attribute
        self.assertEqual(metadata.model_dump()["custom_field"], "custom value")  # Access via dict


class TestBaseDocument(unittest.TestCase):
    """Test the BaseDocument model."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_metadata = {
            "language": "python",
            "format": "python",
            "content_type": "code",
            "file_size": 1024,
            "line_count": 100
        }
        
        self.valid_entity = {
            "type": "function",
            "value": "test_function",
            "line": 10,
            "confidence": 0.9
        }
        
        self.valid_document_data = {
            "id": "doc_12345",
            "source": "/path/to/file.py",
            "content": "```python\ndef test_function():\n    pass\n```",
            "content_type": "markdown",
            "format": "python",
            "raw_content": "def test_function():\n    pass",
            "metadata": self.valid_metadata,
            "entities": [self.valid_entity]
        }
    
    def test_valid_document(self):
        """Test that a valid document passes validation."""
        document = BaseDocument(**self.valid_document_data)
        
        self.assertEqual(document.id, "doc_12345")
        self.assertEqual(document.source, "/path/to/file.py")
        self.assertEqual(document.content, "```python\ndef test_function():\n    pass\n```")
        self.assertEqual(document.content_type, "markdown")
        self.assertEqual(document.format, "python")
        self.assertEqual(document.raw_content, "def test_function():\n    pass")
        self.assertEqual(document.metadata.language, "python")
        self.assertEqual(len(document.entities), 1)
        self.assertEqual(document.entities[0].type, "function")
        self.assertEqual(document.entities[0].value, "test_function")
    
    def test_minimal_document(self):
        """Test that a document with only required fields passes validation."""
        minimal_data = {
            "id": "doc_12345",
            "source": "/path/to/file.py",
            "content": "test content",
            "format": "python",
            "raw_content": "test content",
            "metadata": {
                "language": "python",
                "format": "python",
                "content_type": "code"
            }
        }
        
        document = BaseDocument(**minimal_data)
        
        self.assertEqual(document.id, "doc_12345")
        self.assertEqual(document.source, "/path/to/file.py")
        self.assertEqual(document.content, "test content")
        self.assertEqual(document.content_type, "markdown")  # Default value
        self.assertEqual(document.format, "python")
        self.assertEqual(document.raw_content, "test content")
        self.assertEqual(document.metadata.language, "python")
        self.assertEqual(len(document.entities), 0)  # Default empty list
    
    def test_missing_required_fields(self):
        """Test that a document without required fields fails validation."""
        # Missing ID
        invalid_data = self.valid_document_data.copy()
        del invalid_data["id"]
        
        with self.assertRaises(ValidationError):
            BaseDocument(**invalid_data)
        
        # Missing source
        invalid_data = self.valid_document_data.copy()
        del invalid_data["source"]
        
        with self.assertRaises(ValidationError):
            BaseDocument(**invalid_data)
        
        # Missing metadata
        invalid_data = self.valid_document_data.copy()
        del invalid_data["metadata"]
        
        with self.assertRaises(ValidationError):
            BaseDocument(**invalid_data)
    
    def test_invalid_id(self):
        """Test that a document with an invalid ID fails validation."""
        # ID too short
        invalid_data = self.valid_document_data.copy()
        invalid_data["id"] = "abc"  # Less than 4 characters
        
        with self.assertRaises(ValidationError):
            BaseDocument(**invalid_data)
    
    def test_error_consistency(self):
        """Test that error state is consistent with metadata."""
        # Document with error but metadata.has_errors=False
        data_with_error = self.valid_document_data.copy()
        data_with_error["error"] = "Test error"
        
        document = BaseDocument(**data_with_error)
        
        # The model_validator should have set has_errors to True
        self.assertTrue(document.metadata.has_errors)
        self.assertEqual(document.error, "Test error")
    
    def test_model_dump(self):
        """Test that model_dump returns the expected structure."""
        document = BaseDocument(**self.valid_document_data)
        dump = document.model_dump()
        
        self.assertEqual(dump["id"], "doc_12345")
        self.assertEqual(dump["source"], "/path/to/file.py")
        self.assertEqual(dump["content"], "```python\ndef test_function():\n    pass\n```")
        self.assertEqual(dump["content_type"], "markdown")
        self.assertEqual(dump["format"], "python")
        self.assertEqual(dump["raw_content"], "def test_function():\n    pass")
        self.assertEqual(dump["metadata"]["language"], "python")
        self.assertEqual(len(dump["entities"]), 1)
        self.assertEqual(dump["entities"][0]["type"], "function")
        self.assertEqual(dump["entities"][0]["value"], "test_function")


if __name__ == "__main__":
    unittest.main()
