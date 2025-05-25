"""
Unit tests for the schema validation module.

This module tests the validation utilities and checkpoint functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid
from typing import Dict, Any, List

from pydantic import ValidationError

from src.schema.document_schema import (
    SchemaVersion,
    DocumentType,
    DocumentSchema,
    DatasetSchema
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


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult class."""

    def test_boolean_conversion(self):
        """Test boolean conversion of validation result."""
        # Valid result
        valid_result = ValidationResult(
            is_valid=True,
            stage=ValidationStage.INGESTION
        )
        self.assertTrue(bool(valid_result))
        
        # Invalid result
        invalid_result = ValidationResult(
            is_valid=False,
            stage=ValidationStage.INGESTION,
            errors=[{"loc": ["field"], "msg": "Error message"}]
        )
        self.assertFalse(bool(invalid_result))

    def test_format_errors(self):
        """Test error formatting."""
        # No errors
        result = ValidationResult(
            is_valid=True,
            stage=ValidationStage.PREPROCESSING
        )
        self.assertEqual(result.format_errors(), "No validation errors")
        
        # With errors, no document ID
        result = ValidationResult(
            is_valid=False,
            stage=ValidationStage.CHUNKING,
            errors=[
                {"loc": ["content"], "msg": "field required"},
                {"loc": ["source"], "msg": "field required"}
            ]
        )
        formatted = result.format_errors()
        self.assertIn("Validation errors at chunking stage:", formatted)
        self.assertIn("content: field required", formatted)
        self.assertIn("source: field required", formatted)
        
        # With errors and document ID
        result = ValidationResult(
            is_valid=False,
            stage=ValidationStage.STORAGE,
            errors=[{"loc": ["embedding"], "msg": "invalid embedding"}],
            document_id="test-doc-id"
        )
        formatted = result.format_errors()
        self.assertIn("for document test-doc-id", formatted)
        self.assertIn("embedding: invalid embedding", formatted)


class TestDocumentValidation(unittest.TestCase):
    """Test cases for document validation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_document_data = {
            "id": str(uuid.uuid4()),
            "content": "This is test content.",
            "source": "test.txt",
            "document_type": "text"
        }
        
        self.invalid_document_data = {
            "id": str(uuid.uuid4()),
            # Missing required fields
            "content": "Invalid document."
        }
        
        self.valid_dataset_data = {
            "id": str(uuid.uuid4()),
            "name": "Test Dataset"
        }
        
        self.invalid_dataset_data = {
            # Missing required fields
            "description": "Invalid dataset."
        }

    def test_validate_document(self):
        """Test document validation."""
        # Valid document
        result = validate_document(
            document_data=self.valid_document_data,
            stage=ValidationStage.INGESTION
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.stage, ValidationStage.INGESTION)
        self.assertEqual(result.errors, [])
        
        # Invalid document
        result = validate_document(
            document_data=self.invalid_document_data,
            stage=ValidationStage.INGESTION
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.stage, ValidationStage.INGESTION)
        self.assertGreater(len(result.errors), 0)

    def test_validate_dataset(self):
        """Test dataset validation."""
        # Valid dataset
        result = validate_dataset(
            dataset_data=self.valid_dataset_data,
            stage=ValidationStage.INGESTION
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.stage, ValidationStage.INGESTION)
        self.assertEqual(result.errors, [])
        
        # Invalid dataset
        result = validate_dataset(
            dataset_data=self.invalid_dataset_data,
            stage=ValidationStage.INGESTION
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.stage, ValidationStage.INGESTION)
        self.assertGreater(len(result.errors), 0)

    def test_validate_or_raise(self):
        """Test validate_or_raise function."""
        # Valid data should return model instance
        model = validate_or_raise(
            data=self.valid_document_data,
            schema_class=DocumentSchema,
            stage=ValidationStage.PREPROCESSING
        )
        self.assertIsInstance(model, DocumentSchema)
        
        # Invalid data should raise ValueError
        with self.assertRaises(ValueError):
            validate_or_raise(
                data=self.invalid_document_data,
                schema_class=DocumentSchema,
                stage=ValidationStage.PREPROCESSING,
                error_message="Custom error message"
            )


class TestValidationCheckpoint(unittest.TestCase):
    """Test cases for ValidationCheckpoint decorator."""
    
    def setUp(self):
        """Set up test data and functions."""
        self.valid_document_data = {
            "id": str(uuid.uuid4()),
            "content": "This is test content.",
            "source": "test.txt",
            "document_type": "text"
        }
        
        self.invalid_document_data = {
            "id": str(uuid.uuid4()),
            # Missing required fields
            "content": "Invalid document."
        }
        
        # Define test functions to decorate
        def process_document(self, document):
            return document
        
        def process_with_data(self, data):
            return data
        
        def transform_document(self, document):
            return {"id": document["id"], "transformed": True}
        
        self.process_document = process_document
        self.process_with_data = process_with_data
        self.transform_document = transform_document

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_valid_input_output(self, mock_logger):
        """Test validation checkpoint with valid input and output."""
        # Create decorated function
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.PREPROCESSING,
            input_schema=DocumentSchema,
            output_schema=DocumentSchema
        )
        decorated = checkpoint(self.process_document)
        
        # Call with valid document data
        result = decorated(self, self.valid_document_data)
        
        # Verify result and logger not called for warnings
        self.assertEqual(result, self.valid_document_data)
        mock_logger.warning.assert_not_called()

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_invalid_input(self, mock_logger):
        """Test validation checkpoint with invalid input."""
        # Create decorated function with non-strict validation
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.PREPROCESSING,
            input_schema=DocumentSchema,
            strict=False
        )
        decorated = checkpoint(self.process_document)
        
        # Call with invalid document data
        result = decorated(self, self.invalid_document_data)
        
        # Verify result and logger called for warnings
        self.assertEqual(result, self.invalid_document_data)
        mock_logger.warning.assert_called()

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_strict_mode(self, mock_logger):
        """Test validation checkpoint in strict mode."""
        # Create decorated function with strict validation
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.PREPROCESSING,
            input_schema=DocumentSchema,
            strict=True
        )
        decorated = checkpoint(self.process_document)
        
        # Call with invalid document data should raise
        with self.assertRaises(ValueError):
            decorated(self, self.invalid_document_data)

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_keyword_args(self, mock_logger):
        """Test validation checkpoint with keyword arguments."""
        # Create decorated function
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.ENRICHMENT,
            input_schema=DocumentSchema
        )
        decorated = checkpoint(self.process_document)
        
        # Call with document as keyword argument
        result = decorated(self, document=self.valid_document_data)
        
        # Verify result
        self.assertEqual(result, self.valid_document_data)
        mock_logger.warning.assert_not_called()

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_invalid_output(self, mock_logger):
        """Test validation checkpoint with invalid output."""
        # Create decorated function that returns invalid output
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.ENRICHMENT,
            output_schema=DocumentSchema,
            strict=False
        )
        decorated = checkpoint(self.transform_document)
        
        # Call with valid document data, but output is invalid
        result = decorated(self, self.valid_document_data)
        
        # Verify result and logger called for warnings
        self.assertEqual(result["transformed"], True)
        mock_logger.warning.assert_called()

    @patch('src.schema.validation.logger')
    def test_validation_checkpoint_data_keyword(self, mock_logger):
        """Test validation checkpoint with 'data' keyword argument."""
        checkpoint = ValidationCheckpoint(
            stage=ValidationStage.RETRIEVAL,
            input_schema=DocumentSchema
        )
        decorated = checkpoint(self.process_with_data)
        
        # Call with data as keyword argument
        result = decorated(self, data=self.valid_document_data)
        
        # Verify result
        self.assertEqual(result, self.valid_document_data)
        mock_logger.warning.assert_not_called()


class TestSchemaVersionUpgrade(unittest.TestCase):
    """Test cases for schema version upgrade function."""
    
    def setUp(self):
        """Set up test data."""
        self.v1_document = {
            "id": str(uuid.uuid4()),
            "content": "V1 document content.",
            "source": "v1.txt",
            "document_type": "text",
            "schema_version": "1.0.0",
            "chunks": [
                {
                    "content": "Chunk content without required fields"
                }
            ]
        }
        
        self.v2_document = {
            "id": str(uuid.uuid4()),
            "content": "V2 document content.",
            "source": "v2.txt",
            "document_type": "text",
            "schema_version": "2.0.0",
            "chunks": [
                {
                    "start_offset": 0,
                    "end_offset": 20,
                    "chunk_type": "text",
                    "chunk_index": 0,
                    "parent_id": "parent_id",
                    "content": "Chunk with all fields"
                }
            ]
        }

    def test_upgrade_v1_to_v2(self):
        """Test upgrading from schema V1 to V2."""
        upgraded = upgrade_schema_version(
            document_data=self.v1_document,
            target_version=SchemaVersion.V2
        )
        
        # Verify version updated
        self.assertEqual(upgraded["schema_version"], SchemaVersion.V2.value)
        
        # Verify chunks upgraded
        chunk = upgraded["chunks"][0]
        self.assertIn("chunk_index", chunk)
        self.assertIn("parent_id", chunk)
        self.assertIn("start_offset", chunk)
        self.assertIn("end_offset", chunk)
        self.assertEqual(chunk["parent_id"], self.v1_document["id"])

    def test_already_at_target_version(self):
        """Test upgrading document already at target version."""
        # Should return same document without changes
        result = upgrade_schema_version(
            document_data=self.v2_document,
            target_version=SchemaVersion.V2
        )
        
        # Verify document unchanged
        self.assertEqual(result, self.v2_document)

    def test_no_version_specified(self):
        """Test upgrading document with no version specified."""
        # Create document with no version
        doc_no_version = self.v1_document.copy()
        doc_no_version.pop("schema_version")
        
        # Upgrade should assume V1
        upgraded = upgrade_schema_version(
            document_data=doc_no_version,
            target_version=SchemaVersion.V2
        )
        
        # Verify version updated
        self.assertEqual(upgraded["schema_version"], SchemaVersion.V2.value)
        
        # Original document should be unchanged
        self.assertNotIn("schema_version", doc_no_version)

    def test_upgrade_with_empty_chunks(self):
        """Test upgrading a document with no chunks."""
        # Create document with no chunks
        doc_no_chunks = self.v1_document.copy()
        doc_no_chunks.pop("chunks")
        
        # Upgrade should not add chunks
        upgraded = upgrade_schema_version(
            document_data=doc_no_chunks,
            target_version=SchemaVersion.V2
        )
        
        # Verify version updated but no chunks added
        self.assertEqual(upgraded["schema_version"], SchemaVersion.V2.value)
        self.assertNotIn("chunks", upgraded)


if __name__ == "__main__":
    unittest.main()
