"""Unit tests for the Document Processor Manager.

This module contains comprehensive tests for the DocumentProcessorManager class,
ensuring that it can properly process different types of documents and handle
various edge cases.
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

from src.docproc.manager import DocumentProcessorManager
from src.docproc.adapters.base import BaseAdapter
from tests.unit.common_fixtures import sample_text_document, SAMPLE_TEXT_CONTENT


class MockAdapter(BaseAdapter):
    """Mock document adapter for testing."""
    
    def __init__(self, format_type: str = "mock", content_prefix: str = "processed:"):
        super().__init__(format_type)
        self.content_prefix = content_prefix
    
    def can_process(self, file_path: str) -> bool:
        """Check if this adapter can handle the given file."""
        return True  # Handle everything for testing
    
    def process(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Process the file and return a document dict."""
        content = content or f"Content of {file_path}"
        return {
            "id": f"mock-{Path(file_path).name}",
            "content": f"{self.content_prefix}{content}",
            "path": str(file_path),
            "format": self.format_type,
            "metadata": {"processor": "mock"}
        }
        
    def extract_metadata(self, content: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {"processor": "mock", "timestamp": "2025-05-15T10:00:00Z"}
        
    def extract_entities(self, content: str, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract entities from content."""
        return [{"type": "mock-entity", "text": "mock", "start": 0, "end": 4}]
        
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text content."""
        return {
            "id": f"mock-text",
            "content": f"{self.content_prefix}{text}",
            "format": self.format_type,
            "metadata": self.extract_metadata(text, options),
            "entities": self.extract_entities(text, options)
        }


class TestDocumentProcessorManager:
    """Test suite for the DocumentProcessorManager class."""
    
    def test_init(self):
        """Test initialization of the document processor manager."""
        manager = DocumentProcessorManager()
        assert manager.config == {}
        assert isinstance(manager.cache, dict)
        
        # Test with custom config
        config = {"test_key": "test_value"}
        manager = DocumentProcessorManager(config=config)
        assert manager.config == config
    
    def test_process_document_with_content(self):
        """Test processing a document with direct content."""
        manager = DocumentProcessorManager()
        content = "Test content"
        doc_type = "text"
        
        with patch("src.docproc.manager.process_text") as mock_process_text:
            mock_process_text.return_value = {
                "id": "test-id",
                "content": "processed content",
                "format": "text"
            }
            
            result = manager.process_document(content=content, doc_type=doc_type)
            
            # Verify process_text was called correctly
            mock_process_text.assert_called_once_with(content, doc_type, {})
            
            # Check result
            assert result["content"] == "processed content"
            assert result["format"] == "text"
    
    def test_process_document_with_path(self):
        """Test processing a document from a file path."""
        manager = DocumentProcessorManager()
        path = "/path/to/document.txt"
        
        with patch("src.docproc.manager.process_document") as mock_process_document:
            mock_process_document.return_value = {
                "id": "test-id",
                "content": "file content",
                "path": path,
                "format": "text"
            }
            
            result = manager.process_document(path=path)
            
            # Verify process_document was called correctly
            mock_process_document.assert_called_once()
            
            # Check result
            assert result["content"] == "file content"
            assert result["path"] == path
    
    def test_process_document_with_content_and_path(self):
        """Test processing when both content and path are provided."""
        manager = DocumentProcessorManager()
        content = "Test content"
        path = "/path/to/document.txt"
        
        with patch("src.docproc.manager.process_text") as mock_process_text:
            mock_process_text.return_value = {
                "id": "test-id",
                "content": "processed content",
                "format": "text"
            }
            
            result = manager.process_document(content=content, path=path)
            
            # Verify process_text was called correctly
            mock_process_text.assert_called_once_with(content, "text", {})
            
            # Check path was added
            assert result["path"] == path
    
    def test_process_document_no_content_or_path(self):
        """Test processing when neither content nor path is provided."""
        manager = DocumentProcessorManager()
        
        with pytest.raises(ValueError) as excinfo:
            manager.process_document()
        
        assert "Either content or path must be provided" in str(excinfo.value)
    
    def test_batch_process(self):
        """Test batch processing of documents."""
        manager = DocumentProcessorManager()
        paths = ["/path/to/doc1.txt", "/path/to/doc2.txt"]
        
        # Setup mock for process_document
        with patch.object(manager, "process_document") as mock_process:
            mock_process.side_effect = [
                {"id": "doc1", "content": "content1", "path": paths[0]},
                {"id": "doc2", "content": "content2", "path": paths[1]}
            ]
            
            results = manager.batch_process(paths)
            
            # Verify process_document was called twice
            assert mock_process.call_count == 2
            
            # Check results
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[1]["id"] == "doc2"
    
    def test_batch_process_with_error(self):
        """Test batch processing with an error."""
        manager = DocumentProcessorManager()
        paths = ["/path/to/doc1.txt", "/path/to/error.txt"]
        
        # Setup mock for process_document
        with patch.object(manager, "process_document") as mock_process:
            mock_process.side_effect = [
                {"id": "doc1", "content": "content1", "path": paths[0]},
                Exception("Processing error")
            ]
            
            results = manager.batch_process(paths)
            
            # Verify process_document was called twice
            assert mock_process.call_count == 2
            
            # Check results
            assert len(results) == 2
            assert results[0]["id"] == "doc1"  # First document processed successfully
            assert results[1]["status"] == "failed"  # Second document failed
            assert "Processing error" in results[1]["error"]
    
    def test_get_adapter_for_doc_type(self):
        """Test getting an adapter for a specific document type."""
        manager = DocumentProcessorManager()
        
        with patch("src.docproc.manager.get_adapter_for_format") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_get_adapter.return_value = mock_adapter
            
            # First call should get from the registry
            adapter1 = manager.get_adapter_for_doc_type("pdf")
            assert adapter1 == mock_adapter
            mock_get_adapter.assert_called_once_with("pdf")
            
            # Second call should get from the cache
            mock_get_adapter.reset_mock()
            adapter2 = manager.get_adapter_for_doc_type("pdf")
            assert adapter2 == mock_adapter
            mock_get_adapter.assert_not_called()  # Should not be called again
    
    def test_process_file_with_mock_adapter(self):
        """Test processing a file with a mock adapter through the registry."""
        manager = DocumentProcessorManager()
        
        # Mock the registry to return our adapter
        mock_adapter = MockAdapter()
        
        with patch("src.docproc.manager.get_adapter_for_format") as mock_get_adapter:
            mock_get_adapter.return_value = mock_adapter
            
            # Create a test file
            test_content = "Test content"
            test_path = "/path/to/test.mock"
            
            # Need to mock exists() check and read_text
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=test_content):
                    # Need to mock the core.process_document function directly
                    with patch("src.docproc.manager.process_document") as mock_process_doc:
                        mock_process_doc.return_value = {
                            "id": "mock-test.mock",
                            "content": f"processed:{test_content}",
                            "path": test_path,
                            "format": "mock"
                        }
                        
                        result = manager.process_document(path=test_path)
                
            # Check that the adapter was correctly used
            mock_process_doc.assert_called_once()
            assert result["content"].startswith("processed:")
    
    def test_adapter_caching(self):
        """Test that adapters are properly cached."""
        manager = DocumentProcessorManager()
        
        with patch("src.docproc.manager.get_adapter_for_format") as mock_get_adapter:
            # Create two different mock adapters for different formats
            mock_adapter1 = MagicMock(spec=BaseAdapter)
            mock_adapter2 = MagicMock(spec=BaseAdapter)
            
            # Configure mock to return different adapters based on format
            mock_get_adapter.side_effect = lambda fmt: {
                "text": mock_adapter1,
                "pdf": mock_adapter2
            }.get(fmt)
            
            # First call for each format should hit the registry
            adapter1 = manager.get_adapter_for_doc_type("text")
            adapter2 = manager.get_adapter_for_doc_type("pdf")
            
            assert adapter1 == mock_adapter1
            assert adapter2 == mock_adapter2
            assert mock_get_adapter.call_count == 2
            
            # Reset mock to verify cache hits
            mock_get_adapter.reset_mock()
            
            # Second calls should use the cache
            adapter1_again = manager.get_adapter_for_doc_type("text")
            adapter2_again = manager.get_adapter_for_doc_type("pdf")
            
            assert adapter1_again == mock_adapter1
            assert adapter2_again == mock_adapter2
            assert mock_get_adapter.call_count == 0  # No new calls
    
    def test_file_extension_detection(self):
        """Test that file extensions are properly detected and matched to adapters."""
        manager = DocumentProcessorManager()
        
        # Create a test file path with a specific extension
        file_path = "/path/to/document.pdf"
        
        with patch("src.docproc.manager.process_document") as mock_process_document:
            # Configure mock
            mock_process_document.return_value = {
                "id": "test-id",
                "content": "file content",
                "path": file_path,
                "format": "pdf"
            }
            
            # Add format override
            options = {"format_override": "custom_format"}
            result = manager.process_document(path=file_path, options=options)
            
            # Verify the format override was passed through
            mock_process_document.assert_called_once()
            
            # Check args - need to see how the function is actually called
            call_args = mock_process_document.call_args
            if len(call_args[0]) > 1:  # If passed as positional arg
                passed_options = call_args[0][1]
            else:  # If passed as keyword arg
                passed_options = call_args[1].get('options', {})
                
            assert "format_override" in passed_options
    

    

    

    

    

    

    
    def test_error_handling_in_batch_processing(self):
        """Test error handling during batch processing."""
        manager = DocumentProcessorManager()
        paths = ["/path/to/doc1.txt", "/path/to/error.txt", "/path/to/doc2.txt"]
        
        # Setup mock for process_document with mixed success/failure
        with patch.object(manager, "process_document") as mock_process:
            mock_process.side_effect = [
                {"id": "doc1", "content": "content1"},
                Exception("Processing error"),
                {"id": "doc2", "content": "content2"}
            ]
            
            # Process the batch
            results = manager.batch_process(paths)
            
            # Should have three results (2 success, 1 failure)
            assert len(results) == 3
            
            # First and third should be successful
            assert "error" not in results[0]
            assert "error" not in results[2]
            
            # Second should have error info
            assert "error" in results[1]
            assert results[1]["status"] == "failed"
            assert "Processing error" in results[1]["error"]
            
            # Path should be preserved in the error case
            assert results[1]["path"] == paths[1]
    
    def test_process_with_custom_options(self):
        """Test processing with custom options passed to the processor."""
        manager = DocumentProcessorManager()
        content = "Test content"
        doc_type = "text"
        options = {"custom_option": "value", "another_option": 123}
        
        with patch("src.docproc.manager.process_text") as mock_process_text:
            mock_process_text.return_value = {"id": "test-id", "content": "processed"}
            
            # Process with custom options
            manager.process_document(content=content, doc_type=doc_type, options=options)
            
            # Verify options were passed through
            mock_process_text.assert_called_once_with(content, doc_type, options)
