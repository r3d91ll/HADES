"""
Unit tests for the document schema validation utilities.

This module contains tests for the utilities used to validate document
data throughout the document processing pipeline.
"""

import unittest
import logging
from typing import Dict, Any

from pydantic import BaseModel, ValidationError

from src.docproc.schemas.base import BaseDocument
from src.docproc.schemas.python_document import PythonDocument
from src.docproc.schemas.utils import (
    validate_document, 
    validate_or_default,
    safe_validate,
    add_validation_to_adapter
)


class TestModel(BaseModel):
    """Simple test model for validation tests."""
    name: str
    value: int


class TestValidationUtils(unittest.TestCase):
    """Test the document validation utilities."""
    
    def setUp(self):
        """Set up test data."""
        # Valid Python document data
        self.valid_python_data = {
            "id": "python_test",
            "source": "/path/to/file.py",
            "content": "```python\ndef test_function():\n    pass\n```",
            "content_type": "markdown",
            "format": "python",
            "raw_content": "def test_function():\n    pass",
            "metadata": {
                "language": "python",
                "format": "python",
                "content_type": "code",
                "function_count": 1
            },
            "entities": [
                {
                    "type": "function",
                    "value": "test_function",
                    "line": 1
                }
            ]
        }
        
        # Valid base document data
        self.valid_base_data = {
            "id": "doc_test",
            "source": "/path/to/file.txt",
            "content": "Test content",
            "content_type": "text",
            "format": "text",
            "raw_content": "Test content",
            "metadata": {
                "language": "en",
                "format": "text",
                "content_type": "text"
            }
        }
        
        # Invalid document data (missing required fields)
        self.invalid_data = {
            "id": "invalid_test",
            "source": "/path/to/file.txt",
            # Missing content, format, etc.
        }
        
        # Simple test model data
        self.valid_test_data = {
            "name": "test",
            "value": 42
        }
        
        self.invalid_test_data = {
            "name": "test"
            # Missing value
        }
    
    def test_validate_document_python(self):
        """Test validating a Python document."""
        # Valid Python document
        result = validate_document(self.valid_python_data)
        
        self.assertIsInstance(result, BaseDocument)
        self.assertIsInstance(result, PythonDocument)
        self.assertEqual(result.id, "python_test")
        self.assertEqual(result.format, "python")
        self.assertEqual(result.metadata.function_count, 1)
    
    def test_validate_document_base(self):
        """Test validating a base document."""
        # Valid base document
        result = validate_document(self.valid_base_data)
        
        self.assertIsInstance(result, BaseDocument)
        self.assertEqual(result.id, "doc_test")
        self.assertEqual(result.format, "text")
    
    def test_validate_document_invalid(self):
        """Test validating an invalid document."""
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            validate_document(self.invalid_data)
    
    def test_validate_or_default(self):
        """Test the validate_or_default function."""
        # Valid data
        result = validate_or_default(self.valid_test_data, TestModel)
        
        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.value, 42)
        
        # Invalid data with default
        default = TestModel(name="default", value=0)
        result = validate_or_default(self.invalid_test_data, TestModel, default)
        
        self.assertIs(result, default)
        
        # Invalid data without default
        result = validate_or_default(self.invalid_test_data, TestModel)
        
        self.assertIsNone(result)
    
    def test_safe_validate(self):
        """Test the safe_validate function."""
        # Valid data
        result = safe_validate(self.valid_test_data, TestModel)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 42)
        
        # Invalid data
        result = safe_validate(self.invalid_test_data, TestModel)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertIn("_validation_error", result)
    
    def test_add_validation_to_adapter(self):
        """Test the add_validation_to_adapter function."""
        # Test with Python document
        result = add_validation_to_adapter(self.valid_python_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "python_test")
        self.assertEqual(result["format"], "python")
        
        # Test with base document
        result = add_validation_to_adapter(self.valid_base_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "doc_test")
        self.assertEqual(result["format"], "text")
        
        # Test with invalid document
        result = add_validation_to_adapter(self.invalid_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "invalid_test")
        self.assertIn("_validation_error", result)


if __name__ == "__main__":
    unittest.main()
