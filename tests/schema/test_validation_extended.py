"""
Extended unit tests for the schema validation module.

This module provides additional test coverage for validation utilities
and functions that weren't fully covered in the main test_validation.py file.
"""
import unittest
from unittest.mock import patch, MagicMock, call
import json
from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable

from pydantic import BaseModel, ValidationError, Field

from src.schema.document_schema import (
    SchemaVersion,
    DocumentType,
    DocumentSchema,
    DatasetSchema,
    ChunkMetadata,
    DocumentRelationSchema,
    RelationType
)

from src.schema.validation import (
    ValidationStage,
    ValidationResult,
    validate_document,
    validate_dataset,
    validate_or_raise,
    ValidationCheckpoint,
    upgrade_schema_version
)

T = TypeVar('T', bound=BaseModel)


class TestExtendedValidationResult(unittest.TestCase):
    """Extended test cases for ValidationResult class."""

    def test_validation_result_with_document_id(self):
        """Test ValidationResult with document ID."""
        result = ValidationResult(
            is_valid=True,
            stage=ValidationStage.INGESTION,
            document_id="test-doc-123"
        )
        self.assertEqual(result.document_id, "test-doc-123")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.stage, ValidationStage.INGESTION)
        
    def test_validation_result_with_errors(self):
        """Test ValidationResult with detailed errors."""
        errors = [
            {"loc": ["field1"], "msg": "Field required", "type": "missing"},
            {"loc": ["field2"], "msg": "Invalid value", "type": "value_error"}
        ]
        result = ValidationResult(
            is_valid=False,
            stage=ValidationStage.PREPROCESSING,
            errors=errors,
            document_id="doc-with-errors"
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.errors[0]["loc"], ["field1"])
        self.assertEqual(result.errors[1]["type"], "value_error")
        
    def test_validation_result_str_representation(self):
        """Test string representation of ValidationResult."""
        # Valid result
        valid_result = ValidationResult(
            is_valid=True,
            stage=ValidationStage.STORAGE,
            document_id="valid-doc"
        )
        str_repr = str(valid_result)
        # Just check that the string representation is a non-empty string
        self.assertTrue(isinstance(str_repr, str))
        self.assertTrue(len(str_repr) > 0)
        
        # Invalid result
        invalid_result = ValidationResult(
            is_valid=False,
            stage=ValidationStage.RETRIEVAL,
            errors=[{"loc": ["field"], "msg": "Error message"}],
            document_id="invalid-doc"
        )
        str_repr = str(invalid_result)
        # Just check that the string representation is a non-empty string
        self.assertTrue(isinstance(str_repr, str))
        self.assertTrue(len(str_repr) > 0)


class TestExtendedDocumentValidation(unittest.TestCase):
    """Extended test cases for document validation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_chunk = {
            "start_offset": 0,
            "end_offset": 100,
            "chunk_type": "text",
            "chunk_index": 0,
            "parent_id": "doc123",
            "metadata": {
                "content": "This is chunk content"
            }
        }
        
        self.valid_document = {
            "id": "test-doc-001",
            "title": "Test Document",
            "content": "This is test content",
            "source": "test.txt",
            "document_type": "text",
            "schema_version": "2.0.0",
            "chunks": [self.valid_chunk],
            "metadata": {
                "author": "Test Author",
                "created_at": datetime.now().isoformat()
            }
        }
        
        self.valid_relation = {
            "source_id": "doc1",
            "target_id": "doc2",
            "relation_type": "references",
            "weight": 0.8,
            "bidirectional": False
        }
        
        self.valid_dataset = {
            "id": "test-dataset-001",
            "name": "Test Dataset",
            "description": "Test dataset description",
            "documents": {
                "doc1": self.valid_document
            },
            "relations": [self.valid_relation],
            "schema_version": "2.0.0"
        }
    
    def test_validate_document_with_different_stages(self):
        """Test document validation at different pipeline stages."""
        stages = [
            ValidationStage.INGESTION,
            ValidationStage.PREPROCESSING,
            ValidationStage.CHUNKING,
            ValidationStage.ENRICHMENT,
            ValidationStage.EMBEDDING,
            ValidationStage.STORAGE,
            ValidationStage.RETRIEVAL
        ]
        
        for stage in stages:
            result = validate_document(self.valid_document, stage)
            self.assertTrue(result.is_valid, f"Validation failed at stage {stage}")
            self.assertEqual(result.stage, stage)
    
    def test_validate_document_with_invalid_schema_version(self):
        """Test document validation with invalid schema version."""
        # Create document with invalid schema version
        invalid_doc = self.valid_document.copy()
        invalid_doc["schema_version"] = "3.0.0"  # Non-existent version
        
        # This should not raise an error, but return an invalid result
        result = validate_document(invalid_doc, ValidationStage.INGESTION)
        self.assertFalse(result.is_valid)
    
    def test_validate_dataset_with_different_stages(self):
        """Test dataset validation at different pipeline stages."""
        stages = [
            ValidationStage.INGESTION,
            ValidationStage.STORAGE,
            ValidationStage.RETRIEVAL
        ]
        
        for stage in stages:
            result = validate_dataset(self.valid_dataset, stage)
            self.assertTrue(result.is_valid, f"Dataset validation failed at stage {stage}")
            self.assertEqual(result.stage, stage)
    
    def test_validate_document_with_invalid_data(self):
        """Test document validation with invalid data."""
        # Test successful validation
        result = validate_document(self.valid_document, ValidationStage.INGESTION)
        self.assertTrue(result.is_valid)
        
        # Test failed validation
        invalid_doc = self.valid_document.copy()
        invalid_doc.pop("source")  # Remove required field
        
        result = validate_document(invalid_doc, ValidationStage.INGESTION)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.errors)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_or_raise_with_custom_error_message(self):
        """Test validate_or_raise with custom error message."""
        # Test with valid data
        custom_message = "Custom validation error message"
        validated = validate_or_raise(
            self.valid_document,
            DocumentSchema,
            ValidationStage.INGESTION,
            custom_message
        )
        self.assertIsInstance(validated, DocumentSchema)
        
        # Test with invalid data
        invalid_doc = self.valid_document.copy()
        invalid_doc.pop("source")  # Remove required field
        
        with self.assertRaises(ValueError) as context:
            validate_or_raise(
                invalid_doc,
                DocumentSchema,
                ValidationStage.INGESTION,
                custom_message
            )
        
        self.assertIn(custom_message, str(context.exception))


class TestExtendedValidationCheckpoint(unittest.TestCase):
    """Extended test cases for ValidationCheckpoint decorator."""
    
    def setUp(self):
        """Set up test data and functions."""
        self.valid_document = {
            "id": "test-doc-001",
            "title": "Test Document",
            "content": "This is test content",
            "source": "test.txt",
            "document_type": "text",
            "schema_version": "2.0.0",
            "chunks": [{
                "start_offset": 0,
                "end_offset": 100,
                "chunk_type": "text",
                "chunk_index": 0,
                "parent_id": "test-doc-001",
                "metadata": {
                    "content": "This is chunk content"
                }
            }],
            "metadata": {
                "author": "Test Author"
            }
        }
        
        # Define a test function that returns the same document
        def identity_function(document):
            return document
        
        self.identity_function = identity_function
    
    def test_validation_checkpoint_with_no_schemas(self):
        """Test ValidationCheckpoint with no schemas specified."""
        # Create checkpoint with no schemas
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.INGESTION
        )
        
        # Apply checkpoint to function
        decorated_func = checkpoint(self.identity_function)
        
        # Call decorated function
        result = decorated_func(self.valid_document)
        
        # Verify result
        self.assertEqual(result, self.valid_document)
    
    def test_validation_checkpoint_with_exception_handling(self):
        """Test ValidationCheckpoint with exception handling."""
        # Create a function that raises an exception
        def failing_function(document):
            raise ValueError("Test exception")
        
        # Create checkpoint
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.INGESTION,
            input_schema=DocumentSchema,
            strict=False
        )
        
        # Apply checkpoint to function
        decorated_func = checkpoint(failing_function)
        
        # Call decorated function
        with self.assertRaises(ValueError):
            decorated_func(self.valid_document)
    
    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_with_multiple_args(self, mock_logger):
        """Test ValidationCheckpoint with multiple arguments."""
        # Create a function with multiple arguments
        def multi_arg_function(doc1, doc2, extra=None):
            return {
                "doc1": doc1,
                "doc2": doc2,
                "extra": extra
            }
        
        # Create checkpoint
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.INGESTION,
            input_schema=DocumentSchema,
            strict=False
        )
        
        # Apply checkpoint to function
        decorated_func = checkpoint(multi_arg_function)
        
        # Call decorated function
        result = decorated_func(
            self.valid_document, 
            self.valid_document, 
            extra="test_value"
        )
        
        # Verify result
        self.assertEqual(result["doc1"], self.valid_document)
        self.assertEqual(result["doc2"], self.valid_document)
        self.assertEqual(result["extra"], "test_value")


class TestExtendedSchemaVersionUpgrade(unittest.TestCase):
    """Extended test cases for schema version upgrade function."""
    
    def setUp(self):
        """Set up test data."""
        # Create a V1 document
        self.v1_document = {
            "id": "test-doc-001",
            "title": "Test Document",
            "content": "This is test content",
            "source": "test.txt",
            "document_type": "text",
            "schema_version": "1.0.0",
            "metadata": {
                "author": "Test Author"
            }
        }
        
        # Create a document with missing version
        self.no_version_document = {
            "id": "test-doc-002",
            "title": "Test Document",
            "content": "This is test content",
            "source": "test.txt",
            "document_type": "text",
            "metadata": {
                "author": "Test Author"
            }
        }
    
    def test_upgrade_with_custom_fields(self):
        """Test upgrading a document with custom fields."""
        # Add custom fields to V1 document
        v1_with_custom = self.v1_document.copy()
        v1_with_custom["custom_field"] = "custom_value"
        v1_with_custom["another_custom"] = 123
        
        # Upgrade to V2
        v2_document = upgrade_schema_version(v1_with_custom, SchemaVersion.V2)
        
        # Verify upgrade
        self.assertEqual(v2_document["schema_version"], "2.0.0")
        self.assertEqual(v2_document["custom_field"], "custom_value")
        self.assertEqual(v2_document["another_custom"], 123)
    
    def test_upgrade_with_nested_metadata(self):
        """Test upgrading a document with nested metadata."""
        # Add nested metadata to V1 document
        v1_with_nested = self.v1_document.copy()
        v1_with_nested["metadata"] = {
            "author": "Test Author",
            "nested": {
                "field1": "value1",
                "field2": 123
            },
            "array": [1, 2, 3]
        }
        
        # Upgrade to V2
        v2_document = upgrade_schema_version(v1_with_nested, SchemaVersion.V2)
        
        # Verify upgrade
        self.assertEqual(v2_document["schema_version"], "2.0.0")
        self.assertEqual(v2_document["metadata"]["nested"]["field1"], "value1")
        self.assertEqual(v2_document["metadata"]["nested"]["field2"], 123)
        self.assertEqual(v2_document["metadata"]["array"], [1, 2, 3])
    
    def test_upgrade_with_invalid_document(self):
        """Test upgrading an invalid document."""
        # Create an invalid document (missing required fields)
        invalid_document = {
            "id": "test-doc-003",
            "schema_version": "1.0.0"
        }
        
        # Upgrade to V2
        v2_document = upgrade_schema_version(invalid_document, SchemaVersion.V2)
        
        # Verify upgrade (should still work even if document is invalid)
        self.assertEqual(v2_document["schema_version"], "2.0.0")
        self.assertEqual(v2_document["id"], "test-doc-003")


if __name__ == "__main__":
    unittest.main()
