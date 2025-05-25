"""
Tests for the BasePreProcessor class.

This module provides test coverage for the base pre-processor functionality.
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path

from src.ingest.pre_processor.base_pre_processor import BasePreProcessor, ProcessError


# Create a concrete implementation of BasePreProcessor for testing
class TestablePreProcessor(BasePreProcessor):
    """Concrete implementation of BasePreProcessor for testing."""
    
    def __init__(self, should_raise_error: bool = False):
        super().__init__()
        self.should_raise_error = should_raise_error
        self.process_file_called = False
        self.last_file_path = None
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Implementation of abstract method for testing."""
        self.process_file_called = True
        self.last_file_path = file_path
        
        if self.should_raise_error:
            raise ValueError("Test error")
            
        return {
            "path": file_path,
            "content": "Test content",
            "id": "test-id",
            "type": "test-type",
            "metadata": {"test": "metadata"}
        }
        
    # We now inherit process_batch and get_errors from BasePreProcessor


class TestBasePreProcessor:
    """Tests for the BasePreProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a TestablePreProcessor instance."""
        return TestablePreProcessor()
    
    @pytest.fixture
    def error_processor(self):
        """Create a TestablePreProcessor that raises errors."""
        return TestablePreProcessor(should_raise_error=True)
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Test content")
            path = f.name
        yield path
        os.unlink(path)
    
    def test_initialization(self, processor):
        """Test that the BasePreProcessor initializes correctly."""
        assert processor.errors == []
        assert hasattr(processor, "process_file")
        assert callable(processor.process_file)
    
    def test_process_batch_empty(self, processor):
        """Test processing an empty batch."""
        results = processor.process_batch([])
        assert results == []
        assert processor.errors == []
    
    def test_process_batch_success(self, processor, temp_file):
        """Test processing a batch with successful file processing."""
        results = processor.process_batch([temp_file])
        
        assert len(results) == 1
        assert processor.process_file_called
        assert processor.last_file_path == temp_file
        assert results[0]["path"] == temp_file
        assert processor.errors == []
    
    def test_process_batch_multiple_files(self, processor, temp_file):
        """Test processing multiple files in a batch."""
        # Create a second temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"Another test file")
            second_file = f2.name
        
        try:
            files = [temp_file, second_file]
            results = processor.process_batch(files)
            
            assert len(results) == 2
            assert processor.process_file_called
            assert processor.last_file_path in files
            assert len([r for r in results if r["path"] in files]) == 2
            assert processor.errors == []
        finally:
            os.unlink(second_file)
    
    def test_process_batch_with_error(self, error_processor, temp_file):
        """Test processing a batch with an error."""
        with patch("traceback.format_exc", return_value="Test traceback"):
            results = error_processor.process_batch([temp_file])
            
            assert results == []
            assert error_processor.process_file_called
            assert error_processor.last_file_path == temp_file
            assert len(error_processor.errors) == 1
            assert error_processor.errors[0]["file_path"] == temp_file
            assert "Test error" in error_processor.errors[0]["error"]
            assert error_processor.errors[0]["traceback"] == "Test traceback"
    
    def test_process_batch_nonexistent_file(self, processor):
        """Test processing a non-existent file."""
        non_existent_file = "/path/does/not/exist.txt"
        
        # Mock os.path.exists to return False for our test file
        with patch("os.path.exists", return_value=False):
            with patch("logging.Logger.warning") as mock_warning:
                results = processor.process_batch([non_existent_file])
                
                assert results == []
                assert not processor.process_file_called
                assert processor.errors == []
                mock_warning.assert_called_once()
    
    def test_process_batch_multiple_with_errors(self, error_processor, temp_file):
        """Test processing multiple files with some errors."""
        # Create a valid file
        with tempfile.NamedTemporaryFile(delete=False) as valid_file:
            valid_file.write(b"Valid content")
            valid_path = valid_file.name
            
        # Create a non-existent file path
        non_existent = "/path/does/not/exist.txt"
        
        try:
            # Mock os.path.exists to return True for valid file, False for non-existent
            def mock_exists(path):
                return path != non_existent
                
            with patch("os.path.exists", side_effect=mock_exists):
                with patch("traceback.format_exc", return_value="Test traceback"):
                    # First file will raise an error during processing
                    # Second file doesn't exist
                    results = error_processor.process_batch([temp_file, non_existent])
                    
                    assert results == []
                    assert error_processor.process_file_called
                    assert len(error_processor.errors) == 1
                    assert error_processor.errors[0]["file_path"] == temp_file
        finally:
            os.unlink(valid_path)
    
    def test_get_errors(self, error_processor, temp_file):
        """Test getting errors after processing."""
        with patch("traceback.format_exc", return_value="Test traceback"):
            error_processor.process_batch([temp_file])
            errors = error_processor.get_errors()
            
            assert len(errors) == 1
            assert errors[0]["file_path"] == temp_file
            assert isinstance(errors[0], dict)
            assert isinstance(errors, list)
