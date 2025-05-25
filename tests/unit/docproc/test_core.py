"""
Unit tests for the docproc.core module.

These tests cover the core functionality of the document processing module:
- Processing documents from files
- Processing text content directly
- Batch processing multiple documents
- Format detection
"""

import os
import tempfile
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.docproc.core import (
    process_document,
    process_text,
    get_format_for_document,
    process_documents_batch
)


def create_test_file(content, suffix=".txt"):
    """Create a temporary test file with given content and suffix."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as f:
        f.write(content)
        return Path(f.name)


def cleanup_test_file(file_path):
    """Clean up a temporary test file."""
    try:
        os.unlink(file_path)
    except Exception:
        pass


class TestGetFormatForDocument:
    """Tests for the get_format_for_document function."""
    
    def test_format_detection_by_extension(self):
        """Test format detection based on file extension."""
        # Create test files with different extensions
        md_file = create_test_file("# Markdown content", suffix=".md")
        txt_file = create_test_file("Plain text content", suffix=".txt")
        py_file = create_test_file("def test(): pass", suffix=".py")
        
        try:
            # Test detection
            assert get_format_for_document(md_file) == "markdown"
            assert get_format_for_document(txt_file) == "text"
            assert get_format_for_document(py_file) == "python"
        finally:
            # Clean up
            cleanup_test_file(md_file)
            cleanup_test_file(txt_file)
            cleanup_test_file(py_file)
    
    def test_format_detection_unknown_extension(self):
        """Test format detection with unknown extension."""
        # Create test file with unknown extension
        unknown_file = create_test_file("Some content", suffix=".unknown")
        
        try:
            # Should default to text
            assert get_format_for_document(unknown_file) == "text"
        finally:
            cleanup_test_file(unknown_file)


class TestProcessText:
    """Tests for the process_text function."""
    
    @patch("src.docproc.core.get_adapter_for_format")
    def test_process_text_markdown(self, mock_get_adapter):
        """Test processing markdown text."""
        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.process_text.return_value = {
            "id": "test_doc",
            "content": "# Test Document\n\nContent",
            "format": "markdown",
            "metadata": {"title": "Test Document"}
        }
        mock_get_adapter.return_value = mock_adapter
        
        # Test processing markdown
        result = process_text("# Test Document\n\nContent", format_type="markdown")
        
        # Verify adapter was called correctly
        mock_get_adapter.assert_called_once_with("markdown")
        mock_adapter.process_text.assert_called_once()
        
        # Verify result
        assert result["id"] is not None
        assert result["format"] == "markdown"
        assert "content" in result
        assert "metadata" in result
    
    @patch("src.docproc.core.get_adapter_for_format")
    def test_process_text_with_options(self, mock_get_adapter):
        """Test processing text with options."""
        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.process_text.return_value = {
            "id": "test_doc",
            "content": "Plain text",
            "format": "text",
            "metadata": {}
        }
        mock_get_adapter.return_value = mock_adapter
        
        # Test with options
        options = {"extract_metadata": False}
        result = process_text("Plain text", format_type="text", options=options)
        
        # Verify options were passed
        mock_adapter.process_text.assert_called_once_with("Plain text", options)
        
        # Verify result
        assert result["format"] == "text"
        assert result["content"] == "Plain text"
    
    @patch("src.docproc.core.get_adapter_for_format")
    def test_process_text_format_or_options(self, mock_get_adapter):
        """Test process_text with format_or_options parameter."""
        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.process_text.return_value = {
            "id": "test_doc",
            "content": "Content",
            "format": "html",
            "metadata": {}
        }
        mock_get_adapter.return_value = mock_adapter
        
        # Test with format as string
        process_text("Content", "text", "html")
        mock_get_adapter.assert_called_with("html")
        
        # Reset mock
        mock_get_adapter.reset_mock()
        
        # Test with format_or_options as dict
        options = {"option1": "value1"}
        process_text("Content", "text", options)
        mock_get_adapter.assert_called_with("text")
        mock_adapter.process_text.assert_called_with("Content", options)


class TestProcessDocument:
    """Tests for the process_document function."""
    
    @patch("src.docproc.core.get_adapter_for_format")
    def test_process_document_file(self, mock_get_adapter):
        """Test processing a document file."""
        # Create test file
        test_file = create_test_file("Test content")
        
        try:
            # Create mock adapter
            mock_adapter = MagicMock()
            mock_adapter.process.return_value = {
                "id": "test_doc",
                "content": "Test content",
                "format": "text",
                "metadata": {}
            }
            mock_get_adapter.return_value = mock_adapter
            
            # Test processing
            result = process_document(test_file)
            
            # Verify adapter was called correctly
            mock_get_adapter.assert_called_once()
            mock_adapter.process.assert_called_once()
            
            # Verify result
            assert result["id"] is not None
            assert "content" in result
            assert "format" in result
        finally:
            cleanup_test_file(test_file)
    
    def test_process_document_nonexistent_file(self):
        """Test processing a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            process_document("/nonexistent/file.txt")
    
    @patch("src.docproc.core.get_adapter_for_format")
    def test_process_document_with_options(self, mock_get_adapter):
        """Test processing a document with options."""
        # Create test file
        test_file = create_test_file("Test content")
        
        try:
            # Create mock adapter
            mock_adapter = MagicMock()
            mock_adapter.process.return_value = {
                "id": "test_doc",
                "content": "Test content",
                "format": "text",
                "metadata": {}
            }
            mock_get_adapter.return_value = mock_adapter
            
            # Test with options
            options = {"extract_metadata": False}
            result = process_document(test_file, options=options)
            
            # Verify options were passed
            args, kwargs = mock_adapter.process.call_args
            assert kwargs.get("options") == options or args[1] == options
            
            # Verify result
            assert result["format"] == "text"
        finally:
            cleanup_test_file(test_file)


class TestProcessDocumentsBatch:
    """Tests for the process_documents_batch function."""
    
    @patch("src.docproc.core.process_document")
    def test_batch_processing(self, mock_process_document):
        """Test batch processing of documents."""
        # Create test files
        file1 = create_test_file("Content 1")
        file2 = create_test_file("Content 2")
        
        try:
            # Setup mock
            doc1 = {"id": "doc1", "content": "Content 1"}
            doc2 = {"id": "doc2", "content": "Content 2"}
            mock_process_document.side_effect = [doc1, doc2]
            
            # Test batch processing
            result = process_documents_batch([file1, file2])
            
            # Verify process_document was called for each file
            assert mock_process_document.call_count == 2
            
            # Verify stats based on the actual implementation
            assert result["total"] == 2
            assert result["success"] == 2
            assert result["error"] == 0
            assert result["saved"] == 0
        finally:
            cleanup_test_file(file1)
            cleanup_test_file(file2)
    
    @patch("src.docproc.core.process_document")
    def test_batch_processing_with_errors(self, mock_process_document):
        """Test batch processing with some failures."""
        # Create test files
        file1 = create_test_file("Content 1")
        file2 = create_test_file("Content 2")
        
        try:
            # Setup mock to succeed for first file and fail for second
            doc1 = {"id": "doc1", "content": "Content 1"}
            mock_process_document.side_effect = [
                doc1,
                Exception("Processing error")
            ]
            
            # Test batch processing
            result = process_documents_batch([file1, file2])
            
            # Verify stats based on the actual implementation
            assert result["total"] == 2
            assert result["success"] == 1
            assert result["error"] == 1
            assert result["saved"] == 0
        finally:
            cleanup_test_file(file1)
            cleanup_test_file(file2)
    
    @patch("src.docproc.core.process_document")
    @patch("src.docproc.core.save_processed_document")
    def test_batch_processing_with_output_dir(self, mock_save_document, mock_process_document):
        """Test batch processing with output directory."""
        # Create test files and output directory
        file1 = create_test_file("Content 1")
        
        with tempfile.TemporaryDirectory() as output_dir:
            # Setup mocks
            doc_result = {"id": "doc1", "content": "Content 1"}
            mock_process_document.return_value = doc_result
            mock_save_document.return_value = True
            
            # Test with output directory
            process_documents_batch([file1], output_dir=output_dir)
            
            # Verify save_processed_document was called with the right parameters
            mock_save_document.assert_called_once()
            args, kwargs = mock_save_document.call_args
            assert args[0] == doc_result  # First arg should be the doc result
            assert str(args[1]).startswith(output_dir)  # Second arg should be output path
        
        # Clean up
        cleanup_test_file(file1)
    
    @patch("src.docproc.core.process_document")
    def test_batch_processing_with_callbacks(self, mock_process_document):
        """Test batch processing with success and error callbacks."""
        # Create test files
        file1 = create_test_file("Content 1")
        file2 = create_test_file("Content 2")
        
        try:
            # Setup mock to succeed for first file and fail for second
            doc_result = {"id": "doc1", "content": "Content 1", "path": str(file1)}
            mock_process_document.side_effect = [
                doc_result,
                Exception("Processing error")
            ]
            
            # Setup tracking for callbacks
            success_calls = []
            error_calls = []
            
            def on_success(doc, path):
                success_calls.append((doc, path))
            
            def on_error(path, error):
                error_calls.append((path, error))
            
            # Test with callbacks
            process_documents_batch(
                [file1, file2],
                on_success=on_success,
                on_error=on_error
            )
            
            # Verify callbacks were called
            assert len(success_calls) == 1
            assert success_calls[0][0] == doc_result
            # The path might be None since we're not providing output_dir
            
            assert len(error_calls) == 1
            # First arg is a string with the file path
            assert str(file2) in error_calls[0][0] 
            assert isinstance(error_calls[0][1], Exception)
        finally:
            cleanup_test_file(file1)
            cleanup_test_file(file2)
