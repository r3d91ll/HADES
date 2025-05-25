"""
Tests for document validation in the docproc module.

This module tests the Pydantic validation integration in the document processing pipeline.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from pydantic import ValidationError

# Mock Docling dependency before importing docproc modules
import sys
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.document_converter.DocumentConverter'] = MagicMock()

import src.docproc.core as docproc_core
from src.docproc.schemas.base import BaseDocument, BaseMetadata, BaseEntity
from src.docproc.schemas.utils import validate_document
from src.docproc.adapters.base import BaseAdapter


class MockValidAdapter(BaseAdapter):
    """Mock adapter that returns valid document data."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the mock adapter."""
        self.options = options or {}
        
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation that returns valid document data."""
        options = options or {}
        return {
            'id': 'test-doc-123',
            'source': str(file_path),
            'content': f"Processed content from {file_path}",
            'content_type': 'markdown',
            'format': 'text',
            'raw_content': f"Raw content from {file_path}",
            'metadata': {
                'language': 'en',
                'format': 'text',
                'content_type': 'text',
                'file_size': 100,
                'line_count': 5,
                'char_count': 100,
                'has_errors': False
            },
            'entities': []
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation that returns valid document data."""
        options = options or {}
        return {
            'id': 'test-text-123',
            'source': 'direct_text',
            'content': f"Processed text: {text[:20]}...",
            'content_type': 'markdown',
            'format': 'text',
            'raw_content': text,
            'metadata': {
                'language': 'en',
                'format': 'text',
                'content_type': 'text',
                'file_size': len(text),
                'line_count': text.count('\n') + 1,
                'char_count': len(text),
                'has_errors': False
            },
            'entities': []
        }
    
    def extract_entities(self, content: Any) -> list:
        """Mock implementation of extract_entities method."""
        return []
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Mock implementation of extract_metadata method."""
        return {
            'language': 'en',
            'format': 'text',
            'content_type': 'text'
        }


class MockInvalidAdapter(BaseAdapter):
    """Mock adapter that returns invalid document data."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the mock adapter."""
        self.options = options or {}
        
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation that returns invalid document data (missing required fields)."""
        options = options or {}
        return {
            # Missing 'id' field
            'source': str(file_path),
            'content': f"Processed content from {file_path}",
            # Missing 'content_type' field
            # Missing 'format' field
            'raw_content': f"Raw content from {file_path}",
            # Missing 'metadata' field
            'entities': []
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock implementation that returns invalid document data."""
        options = options or {}
        return {
            # Missing required fields
            'content': f"Processed text: {text[:20]}...",
            'raw_content': text,
        }
    
    def extract_entities(self, content: Any) -> list:
        """Mock implementation of extract_entities method."""
        return []
    
    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Mock implementation of extract_metadata method."""
        return {}


class TestDocumentValidation(unittest.TestCase):
    """Test cases for document validation in the docproc module."""

    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create test file
        self.test_text_file = self.temp_dir_path / "test.txt"
        with open(self.test_text_file, 'w') as f:
            f.write("This is a test text file.")
    
    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
    
    def test_validate_document_valid(self):
        """Test validating a valid document."""
        # Create a valid document
        valid_doc = {
            'id': 'test-doc-123',
            'source': 'test-source',
            'content': 'Test content',
            'content_type': 'markdown',
            'format': 'text',
            'raw_content': 'Raw test content',
            'metadata': {
                'language': 'en',
                'format': 'text',
                'content_type': 'text',
                'file_size': 100,
                'line_count': 5,
                'char_count': 100,
                'has_errors': False
            },
            'entities': []
        }
        
        # Validate the document
        validated_doc = validate_document(valid_doc)
        
        # Check that validation succeeded
        self.assertIsInstance(validated_doc, BaseDocument)
        self.assertEqual(validated_doc.id, 'test-doc-123')
        self.assertEqual(validated_doc.source, 'test-source')
        self.assertEqual(validated_doc.content, 'Test content')
        
    def test_validate_document_python_format(self):
        """Test validating a document with python format."""
        # Create a valid python document
        python_doc = {
            'id': 'test-python-123',
            'source': 'test.py',
            'content': 'def test(): pass',
            'content_type': 'markdown',
            'format': 'python',  # This should trigger python document validation
            'raw_content': 'def test(): pass',
            'metadata': {
                'language': 'python',
                'format': 'python',
                'content_type': 'code',
                'file_size': 100,
                'line_count': 1,
                'char_count': 15,
                'has_errors': False
            },
            'entities': []
        }
        
        # Instead of mocking the validation error, let's just test the normal flow
        # where it validates as a Python document
        validated_doc = validate_document(python_doc)
        
        # Check that validation succeeded
        self.assertIsInstance(validated_doc, BaseDocument)
        self.assertEqual(validated_doc.id, 'test-python-123')
        self.assertEqual(validated_doc.format, 'python')
    
    def test_validate_document_invalid(self):
        """Test validating an invalid document."""
        # Create an invalid document (missing required fields)
        invalid_doc = {
            'content': 'Test content',
            'raw_content': 'Raw test content'
        }
        
        # Validation should raise an error
        with self.assertRaises(ValidationError):
            validate_document(invalid_doc)
            
    def test_validate_or_default_valid(self):
        """Test validate_or_default with valid data."""
        from src.docproc.schemas.utils import validate_or_default
        
        # Create valid metadata
        valid_metadata = {
            'language': 'en',
            'format': 'text',
            'content_type': 'text',
            'file_size': 100,
            'line_count': 5,
            'char_count': 100,
            'has_errors': False
        }
        
        # Validate against BaseMetadata
        result = validate_or_default(valid_metadata, BaseMetadata)
        
        # Check validation succeeded
        self.assertIsInstance(result, BaseMetadata)
        self.assertEqual(result.language, 'en')
        self.assertEqual(result.format, 'text')
        
    def test_validate_or_default_invalid(self):
        """Test validate_or_default with invalid data."""
        from src.docproc.schemas.utils import validate_or_default
        
        # Create invalid metadata (missing required fields)
        invalid_metadata = {
            'file_size': 100,
            'line_count': 5
        }
        
        # Default value
        default_metadata = BaseMetadata(
            language='unknown',
            format='unknown',
            content_type='unknown'
        )
        
        # Validate against BaseMetadata with default
        result = validate_or_default(invalid_metadata, BaseMetadata, default_metadata)
        
        # Check default was returned
        self.assertIsInstance(result, BaseMetadata)
        self.assertEqual(result.language, 'unknown')
        self.assertEqual(result.format, 'unknown')
        
    def test_safe_validate_valid(self):
        """Test safe_validate with valid data."""
        from src.docproc.schemas.utils import safe_validate
        
        # Create valid entity
        valid_entity = {
            'type': 'keyword',
            'value': 'test',
            'confidence': 0.95
        }
        
        # Validate against BaseEntity
        result = safe_validate(valid_entity, BaseEntity)
        
        # Check validation succeeded and returned dict
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'keyword')
        self.assertEqual(result['value'], 'test')
        self.assertEqual(result['confidence'], 0.95)
        
    def test_safe_validate_invalid(self):
        """Test safe_validate with invalid data."""
        from src.docproc.schemas.utils import safe_validate
        
        # Create invalid entity (missing required field)
        invalid_entity = {
            'value': 'test',
            'confidence': 1.5  # Invalid confidence (should be 0-1)
        }
        
        # Validate against BaseEntity
        result = safe_validate(invalid_entity, BaseEntity)
        
        # Check original data was returned with validation error
        self.assertIsInstance(result, dict)
        self.assertEqual(result['value'], 'test')
        self.assertEqual(result['confidence'], 1.5)
        self.assertIn('_validation_error', result)
        
    def test_add_validation_to_adapter_valid_base(self):
        """Test add_validation_to_adapter with valid base document."""
        from src.docproc.schemas.utils import add_validation_to_adapter
        
        # Create valid document
        valid_doc = {
            'id': 'test-doc-123',
            'source': 'test-source',
            'content': 'Test content',
            'content_type': 'markdown',
            'format': 'text',  # Not python format
            'raw_content': 'Raw test content',
            'metadata': {
                'language': 'en',
                'format': 'text',
                'content_type': 'text',
                'file_size': 100,
                'line_count': 5,
                'char_count': 100,
                'has_errors': False
            },
            'entities': []
        }
        
        # Validate with adapter validation
        result = add_validation_to_adapter(valid_doc)
        
        # Check validation succeeded
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'test-doc-123')
        self.assertNotIn('_validation_error', result)
        
    def test_add_validation_to_adapter_valid_python(self):
        """Test add_validation_to_adapter with valid python document."""
        from src.docproc.schemas.utils import add_validation_to_adapter
        
        # Create valid python document
        valid_python_doc = {
            'id': 'test-python-123',
            'source': 'test.py',
            'content': 'def test(): pass',
            'content_type': 'markdown',
            'format': 'python',  # Python format
            'raw_content': 'def test(): pass',
            'metadata': {
                'language': 'python',
                'format': 'python',
                'content_type': 'code',
                'file_size': 100,
                'line_count': 1,
                'char_count': 15,
                'has_errors': False
            },
            'entities': []
        }
        
        # Validate with adapter validation
        result = add_validation_to_adapter(valid_python_doc)
        
        # Check validation succeeded
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 'test-python-123')
        self.assertNotIn('_validation_error', result)
        
    def test_add_validation_to_adapter_invalid(self):
        """Test add_validation_to_adapter with invalid document."""
        from src.docproc.schemas.utils import add_validation_to_adapter
        
        # Create invalid document
        invalid_doc = {
            'content': 'Test content',
            'format': 'text',
            # Missing required fields
        }
        
        # Validate with adapter validation
        result = add_validation_to_adapter(invalid_doc)
        
        # Check original data was returned with validation error
        self.assertIsInstance(result, dict)
        self.assertEqual(result['content'], 'Test content')
        self.assertIn('_validation_error', result)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_with_valid_adapter(self, mock_get_adapter):
        """Test processing a document with an adapter that returns valid data."""
        # Setup mock
        mock_adapter = MockValidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Process the document
        result = docproc_core.process_document(self.test_text_file)
        
        # Verify the document was validated successfully
        self.assertNotIn('_validation_error', result)
        self.assertEqual(result['id'], 'test-doc-123')
        self.assertEqual(result['source'], str(self.test_text_file))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_document_with_invalid_adapter(self, mock_get_adapter):
        """Test processing a document with an adapter that returns invalid data."""
        # Setup mock
        mock_adapter = MockInvalidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Process the document - should not raise an exception but include validation error
        result = docproc_core.process_document(self.test_text_file)
        
        # Verify validation error was captured
        self.assertIn('_validation_error', result)
        self.assertIn('id', result['_validation_error'])  # Error should mention missing 'id' field
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text_with_valid_adapter(self, mock_get_adapter):
        """Test processing text with an adapter that returns valid data."""
        # Setup mock
        mock_adapter = MockValidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Process text
        text = "This is sample text content to process."
        result = docproc_core.process_text(text, 'text')
        
        # Verify the document was validated successfully
        self.assertNotIn('_validation_error', result)
        self.assertEqual(result['id'], 'test-text-123')
        self.assertEqual(result['source'], 'direct_text')
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_text_with_invalid_adapter(self, mock_get_adapter):
        """Test processing text with an adapter that returns invalid data."""
        # Setup mock
        mock_adapter = MockInvalidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Process text - should not raise an exception but include validation error
        text = "This is sample text content to process."
        result = docproc_core.process_text(text, 'text')
        
        # Verify validation error was captured
        self.assertIn('_validation_error', result)
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_save_processed_document_with_validation(self, mock_get_adapter):
        """Test saving a processed document with validation."""
        # Setup mock
        mock_adapter = MockValidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Process the document
        result = docproc_core.process_document(self.test_text_file)
        
        # Save the document
        output_path = self.temp_dir_path / "output.json"
        saved_path = docproc_core.save_processed_document(result, output_path)
        
        # Verify the document was saved
        self.assertTrue(saved_path.exists())
        
        # Load the saved document and verify it has the _validated flag
        with open(saved_path, 'r') as f:
            saved_doc = json.load(f)
        
        self.assertTrue(saved_doc.get('_validated', False))
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_documents_batch_with_validation(self, mock_get_adapter):
        """Test batch processing documents with validation."""
        # Setup mock
        mock_adapter = MockValidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = self.temp_dir_path / f"test_{i}.txt"
            with open(file_path, 'w') as f:
                f.write(f"This is test file {i}.")
            test_files.append(file_path)
        
        # Process the batch
        output_dir = self.temp_dir_path / "output"
        stats = docproc_core.process_documents_batch(test_files, output_dir=output_dir)
        
        # Verify stats
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['success'], 3)
        self.assertEqual(stats['error'], 0)
        self.assertEqual(stats['saved'], 3)
        self.assertEqual(stats['validation_failures'], 0)
        
        # Verify output files were created
        for i in range(3):
            output_file = output_dir / f"test_{i}.json"
            self.assertTrue(output_file.exists())
    
    @patch('src.docproc.core.get_adapter_for_format')
    def test_process_documents_batch_with_invalid_data(self, mock_get_adapter):
        """Test batch processing documents with invalid data."""
        # Setup mock
        mock_adapter = MockInvalidAdapter()
        mock_get_adapter.return_value = mock_adapter
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = self.temp_dir_path / f"test_{i}.txt"
            with open(file_path, 'w') as f:
                f.write(f"This is test file {i}.")
            test_files.append(file_path)
        
        # Process the batch
        output_dir = self.temp_dir_path / "output"
        stats = docproc_core.process_documents_batch(test_files, output_dir=output_dir)
        
        # Verify stats
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['success'], 3)  # Processing still succeeds even with validation failures
        self.assertEqual(stats['error'], 0)
        self.assertEqual(stats['saved'], 3)
        self.assertEqual(stats['validation_failures'], 3)  # All documents should have validation failures


if __name__ == '__main__':
    unittest.main()
