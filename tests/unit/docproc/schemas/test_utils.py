"""
Unit tests for schema utility functions.

This module tests the validation utilities in the schemas/utils.py module,
which provides functionality for document validation across the processing pipeline.
"""

import pytest
import logging
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, ValidationError

from src.docproc.schemas.utils import (
    validate_document,
    validate_or_default,
    safe_validate,
    add_validation_to_adapter
)
from src.docproc.schemas.base import BaseDocument, BaseMetadata, BaseEntity
from src.docproc.schemas.python_document import PythonDocument, PythonMetadata


class TestModel(BaseModel):
    """Simple test model for validation tests."""
    name: str
    age: int


class TestValidateOrDefault:
    """Tests for the validate_or_default function."""
    
    def test_successful_validation(self):
        """Test successful validation against a model."""
        # Valid data
        data = {"name": "Test User", "age": 30}
        
        # Validate against model
        result = validate_or_default(data, TestModel)
        
        # Verify result
        assert isinstance(result, TestModel)
        assert result.name == "Test User"
        assert result.age == 30
    
    def test_failed_validation_no_default(self):
        """Test validation failure with no default provided."""
        # Invalid data (missing required field)
        data = {"name": "Test User"}
        
        # Validate against model
        result = validate_or_default(data, TestModel)
        
        # Verify result is None (default when no default is provided)
        assert result is None
    
    def test_failed_validation_with_default(self):
        """Test validation failure with a default value."""
        # Invalid data (missing required field)
        data = {"name": "Test User"}
        
        # Create default instance
        default = TestModel(name="Default", age=0)
        
        # Validate against model
        result = validate_or_default(data, TestModel, default)
        
        # Verify result is the default
        assert result is default
        assert result.name == "Default"
        assert result.age == 0
    
    def test_validation_logs_warning(self):
        """Test that validation failure logs a warning."""
        # Invalid data (missing required field)
        data = {"name": "Test User"}
        
        # Validate against model with mocked logger
        with patch('src.docproc.schemas.utils.logger') as mock_logger:
            validate_or_default(data, TestModel)
            
            # Verify warning was logged
            assert mock_logger.warning.called


class TestSafeValidate:
    """Tests for the safe_validate function."""
    
    def test_successful_validation(self):
        """Test successful validation of data."""
        # Valid data
        data = {"name": "Test User", "age": 30}
        
        # Safe validate
        result = safe_validate(data, TestModel)
        
        # Verify result matches expected
        assert isinstance(result, dict)
        assert result["name"] == "Test User"
        assert result["age"] == 30
        assert "_validation_error" not in result
    
    def test_failed_validation(self):
        """Test returning original data when validation fails."""
        # Invalid data (missing required field)
        data = {"name": "Test User"}
        
        # Safe validate
        result = safe_validate(data, TestModel)
        
        # Verify result is original data with error
        assert isinstance(result, dict)
        assert result["name"] == "Test User"
        assert "_validation_error" in result
        assert "age" in result["_validation_error"].lower()
    
    def test_validation_logs_warning(self):
        """Test that validation failure logs a warning."""
        # Invalid data (missing required field)
        data = {"name": "Test User"}
        
        # Safe validate with mocked logger
        with patch('src.docproc.schemas.utils.logger') as mock_logger:
            safe_validate(data, TestModel)
            
            # Verify warning was logged
            assert mock_logger.warning.called


class TestAddValidationToAdapter:
    """Tests for the add_validation_to_adapter function."""
    
    def test_validation_failure_with_mock(self):
        """Test that validation failure is handled gracefully."""
        # Create a simple adapter result
        adapter_result = {
            "format": "python",
            "content": "def test(): pass"
        }
        
        # Mock both PythonDocument and BaseDocument validation to fail
        with patch('src.docproc.schemas.python_document.PythonDocument.model_validate') as mock_py_validate:
            # Make PythonDocument validation raise a ValidationError
            mock_py_validate.side_effect = ValidationError.from_exception_data(
                title="ValidationError",
                line_errors=[{"loc": ("content",), "msg": "field required", "type": "missing"}]
            )
            
            # Also mock BaseDocument validation in case it's called as a fallback
            with patch('src.docproc.schemas.base.BaseDocument.model_validate') as mock_base_validate:
                mock_base_validate.side_effect = ValidationError.from_exception_data(
                    title="ValidationError",
                    line_errors=[{"loc": ("content",), "msg": "field required", "type": "missing"}]
                )
                
                # Also mock logger to verify warning is logged
                with patch('src.docproc.schemas.utils.logger') as mock_logger:
                    # Call the function
                    result = add_validation_to_adapter(adapter_result)
                    
                    # Verify validation failure was handled
                    assert mock_py_validate.called
                    assert mock_logger.warning.called
                    assert isinstance(result, dict)
                    assert result["format"] == "python"
                    assert "_validation_error" in result


def create_test_python_document():
    """Create a valid test Python document for testing."""
    return {
        "id": "test123",
        "source": "test.py",
        "format": "python",
        "content": "def test(): pass",
        "content_type": "code",
        "raw_content": "def test(): pass",
        "metadata": {
            "language": "python",
            "format": "python",
            "content_type": "code",
            "function_count": 1,
            "class_count": 0,
            "import_count": 0,
            "method_count": 0, 
            "has_module_docstring": False,
            "has_syntax_errors": False
        }
    }


class TestDocumentValidation:
    """Integration tests for document validation."""
    
    def test_validate_document_with_mocks(self):
        """Test document validation with mocks."""
        # Create test data
        data = {
            "format": "python"
        }
        
        # Create a logger patch to capture warnings
        with patch('src.docproc.schemas.utils.logger') as mock_logger:
            # Mock both validation methods to control behavior
            with patch('src.docproc.schemas.python_document.PythonDocument.model_validate') as mock_python:
                # Make PythonDocument validation raise a ValidationError
                mock_python.side_effect = ValidationError.from_exception_data(
                    title="ValidationError",
                    line_errors=[{"loc": ("content",), "msg": "field required", "type": "missing"}]
                )
                
                with patch('src.docproc.schemas.base.BaseDocument.model_validate') as mock_base:
                    # Return a mock BaseDocument
                    mock_base_doc = MagicMock(spec=BaseDocument)
                    mock_base.return_value = mock_base_doc
                    
                    # Call validate_document
                    result = validate_document(data)
                    
                    # Verify Python validation was attempted first
                    assert mock_python.called
                    
                    # Verify warning was logged for Python validation failure
                    assert mock_logger.warning.called
                    
                    # Verify fallback to BaseDocument
                    assert mock_base.called
                    assert result is mock_base_doc
