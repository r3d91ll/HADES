"""
Tests for the DoclingPreProcessor class.

This module provides comprehensive test coverage for the Docling preprocessor functionality.
"""
import os
import pytest
import tempfile
import re
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any, List
from pathlib import Path

from src.ingest.pre_processor.docling_pre_processor import DoclingPreProcessor, _BS4_AVAILABLE


class TestDoclingPreProcessor:
    """Tests for the DoclingPreProcessor class."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock DoclingAdapter for testing."""
        mock = MagicMock()
        mock.analyze_text.return_value = {"sentences": [{"text": "Test sentence"}]}
        mock.analyze_file.return_value = {"sentences": [{"text": "Test file content"}]}
        mock.extract_entities.return_value = [{"type": "PERSON", "text": "John Doe"}]
        mock.extract_relationships.return_value = [{"source": "John", "target": "report", "relation": "wrote"}]
        mock.extract_keywords.return_value = [{"text": "test", "score": 0.9}]
        return mock
    
    @pytest.fixture
    def mock_doc_adapter(self):
        """Create a mock DocProcAdapter for testing."""
        mock = MagicMock()
        # Setup mock to return a realistic docproc result
        mock.process_file.return_value = {
            "id": "test-doc-id",
            "path": "/mock/path/file.txt",
            "content": "Test content",
            "metadata": {"title": "Test Document"},
            "entities": [{"type": "PERSON", "text": "John Doe"}],
            "format": "html"
        }
        return mock
    
    @pytest.fixture
    def processor(self, mock_adapter, mock_doc_adapter):
        """Create a DoclingPreProcessor with mocked adapters."""
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter", return_value=mock_adapter), \
             patch("src.ingest.pre_processor.docling_pre_processor.DocProcAdapter", return_value=mock_doc_adapter):
            yield DoclingPreProcessor()
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("This is a test document for Docling analysis.")
            path = f.name
        yield path
        os.unlink(path)
    
    def test_initialization(self):
        """Test initialization of DoclingPreProcessor."""
        # Test with default options
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter") as mock_adapter_class:
            processor = DoclingPreProcessor()
            mock_adapter_class.assert_called_once_with(None)
            
        # Test with custom options
        custom_options = {"model": "test-model", "language": "en"}
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter") as mock_adapter_class:
            processor = DoclingPreProcessor(options=custom_options)
            mock_adapter_class.assert_called_once_with(custom_options)
    
    def test_analyze_text(self, processor, mock_adapter):
        """Test analyze_text method."""
        text = "This is test text."
        result = processor.analyze_text(text)
        
        mock_adapter.analyze_text.assert_called_once_with(text)
        assert result == {"sentences": [{"text": "Test sentence"}]}
    
    def test_analyze_file(self, processor, mock_adapter, temp_file):
        """Test analyze_file method."""
        result = processor.analyze_file(temp_file)
        
        mock_adapter.analyze_file.assert_called_once_with(temp_file)
        assert result == {"sentences": [{"text": "Test file content"}]}
        
        # Test with Path object
        processor.analyze_file(Path(temp_file))
        mock_adapter.analyze_file.assert_called_with(Path(temp_file))
    
    def test_extract_entities(self, processor, mock_adapter, temp_file):
        """Test extract_entities method."""
        result = processor.extract_entities(temp_file)
        
        mock_adapter.extract_entities.assert_called_once_with(temp_file)
        assert result == [{"type": "PERSON", "text": "John Doe"}]
        
        # Test with Path object
        processor.extract_entities(Path(temp_file))
        mock_adapter.extract_entities.assert_called_with(Path(temp_file))
    
    def test_extract_relationships(self, processor, mock_adapter):
        """Test extract_relationships method."""
        text = "John wrote a report."
        result = processor.extract_relationships(text)
        
        mock_adapter.extract_relationships.assert_called_once_with(text)
        assert result == [{"source": "John", "target": "report", "relation": "wrote"}]
    
    def test_extract_keywords(self, processor, mock_adapter):
        """Test extract_keywords method."""
        text = "This is a test document."
        result = processor.extract_keywords(text)
        
        mock_adapter.extract_keywords.assert_called_once_with(text)
        assert result == [{"text": "test", "score": 0.9}]
    
    def test_process_file_success(self, processor, temp_file, mock_doc_adapter):
        """Test successful file processing."""
        # Update mock_doc_adapter to use the actual file path
        mock_doc_adapter.process_file.return_value = {
            "id": "test-doc-id",
            "path": temp_file,
            "content": "Test content",
            "metadata": {"title": "Test Document"},
            "entities": [{"type": "PERSON", "text": "John Doe"}],
            "format": "html"
        }
        
        # Process the file
        result = processor.process_file(temp_file)
            
        # Verify the DocProcAdapter was called
        mock_doc_adapter.process_file.assert_called_once_with(temp_file)
            
        # Verify the result has expected structure
        assert result["path"] == temp_file
        assert result["content"] == "Test content"
        assert result["metadata"]["title"] == "Test Document"
        assert result["entities"] == [{"type": "PERSON", "text": "John Doe"}]
    
    def test_process_file_nonexistent(self, processor, mock_doc_adapter):
        """Test process_file with a nonexistent file."""
        # Configure the DocProcAdapter to handle nonexistent files
        mock_doc_adapter.process_file.return_value = None
        
        # The DoclingPreProcessor wraps FileNotFoundError in a general Exception with a detailed message
        with pytest.raises(Exception) as exc_info:
            processor.process_file("/nonexistent/file.txt")
        
        # Verify the error message contains both "File not found" and the file path
        assert "File not found" in str(exc_info.value)
        assert "/nonexistent/file.txt" in str(exc_info.value)
        
        # Verify the DocProcAdapter was called
        mock_doc_adapter.process_file.assert_called_once_with("/nonexistent/file.txt")
    
    def test_process_file_with_error(self, processor, temp_file, mock_doc_adapter):
        """Test process_file with an error in the adapter."""
        # Configure the DocProcAdapter to simulate an error
        mock_doc_adapter.process_file.side_effect = ValueError("Analysis error")
        
        # With our new adapter pattern, errors are handled and re-raised with additional context
        with pytest.raises(Exception) as exc_info:
            processor.process_file(temp_file)
        
        # Verify the error message contains the original error
        assert "Analysis error" in str(exc_info.value)
        # Verify the adapter method was called
        mock_doc_adapter.process_file.assert_called_once_with(temp_file)
    
    def test_process_file_with_read_error(self, processor, temp_file, mock_doc_adapter):
        """Test process_file with a file reading error."""
        # Configure the DocProcAdapter to simulate a read error
        mock_doc_adapter.process_file.side_effect = IOError("Read error")
        
        # With our new adapter pattern, errors are handled and re-raised with additional context
        with pytest.raises(Exception) as exc_info:
            processor.process_file(temp_file)
        
        # Verify the error message contains the original error
        assert "Read error" in str(exc_info.value)
        # Verify the adapter method was called
        mock_doc_adapter.process_file.assert_called_once_with(temp_file)
    
    def test_process_file_with_html_links(self, processor, mock_doc_adapter):
        """Test HTML link extraction with our new adapter pattern."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Document</h1>
            <p>This is a <a href="local_link.html">local link</a>.</p>
            <p>This is an <a href="https://example.com">external link</a>.</p>
        </body>
        </html>
        """
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(html_content.encode('utf-8'))
            html_file_path = f.name
        
        try:
            # Set up mock DocProcAdapter response with relationships
            mock_doc_adapter.process_file.return_value = {
                "id": "html_12345678_test",
                "path": html_file_path,
                "content": html_content,
                "metadata": {"title": "Test Page"},
                "entities": [],
                "format": "html",
                "relationships": [
                    {
                        "source_id": "html_12345678_test",
                        "target_id": "html_87654321_local_link_html",
                        "type": "references",
                        "metadata": {"href": "local_link.html"}
                    }
                ]
            }
            
            # Process the file
            result = processor.process_file(html_file_path)
            
            # Check if relationships were returned
            assert "relationships" in result
            assert len(result["relationships"]) > 0
            
            # Check the relationship details
            relationship = result["relationships"][0]
            assert relationship["type"] == "references"
            assert relationship["metadata"]["href"] == "local_link.html"
            
            # Verify the DocProcAdapter was called
            mock_doc_adapter.process_file.assert_called_once_with(html_file_path)
        finally:
            # Clean up the temporary file
            os.unlink(html_file_path)
    
    def test_process_file_with_regex_fallback(self, processor, mock_doc_adapter):
        """Test HTML link extraction with regex fallback using our new adapter pattern."""
        html_content = '<a href="relative/path.html">Link</a>'
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(html_content.encode('utf-8'))
            html_file_path = f.name
        
        try:
            # Set up mock DocProcAdapter response with relationships
            mock_doc_adapter.process_file.return_value = {
                "id": "html_12345678_test",
                "path": html_file_path,
                "content": html_content,
                "metadata": {"title": "Test Document"},
                "entities": [],
                "format": "html",
                "relationships": [
                    {
                        "source_id": "html_12345678_test",
                        "target_id": "html_87654321_relative_path_html",
                        "type": "references",
                        "metadata": {"href": "relative/path.html"}
                    }
                ]
            }
            
            # Process the file
            result = processor.process_file(html_file_path)
            
            # Verify relationships were extracted
            assert "relationships" in result
            assert len(result["relationships"]) > 0
            
            # Verify the relationship details
            relationship = result["relationships"][0]
            assert relationship["type"] == "references"
            assert relationship["metadata"]["href"] == "relative/path.html"
            
            # Verify the DocProcAdapter was called
            mock_doc_adapter.process_file.assert_called_once_with(html_file_path)
        finally:
            # Clean up the temporary file
            os.unlink(html_file_path)
    
    def test_process_file_with_html_parsing_error(self, processor, mock_doc_adapter):
        """Test error handling during HTML parsing with our new adapter pattern."""
        malformed_html = "<html><body><div></html>"
        
        # Create a temporary HTML file with malformed content
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(malformed_html.encode('utf-8'))
            html_file_path = f.name
        
        try:
            # Configure the DocProcAdapter to simulate a parsing error
            mock_doc_adapter.process_file.side_effect = ValueError("Malformed HTML: unclosed div tag")
            
            # The processor should handle the error gracefully but re-raise with context
            with pytest.raises(Exception) as exc_info:
                processor.process_file(html_file_path)
                
            # Check that the error message is informative
            assert "Error processing file" in str(exc_info.value)
            assert "Malformed HTML" in str(exc_info.value)
            
            # Verify the DocProcAdapter was called
            mock_doc_adapter.process_file.assert_called_once_with(html_file_path)
        finally:
            # Clean up the temporary file
            os.unlink(html_file_path)
    
    def test_bs4_import_handling(self):
        """Test that the module correctly handles BS4 availability."""
        # This test verifies that the module level _BS4_AVAILABLE flag is set
        # and that appropriate fallback classes are defined
        
        # We can't directly test the import behavior, but we can verify
        # the resulting variables and classes
        
        # Check if the module has the expected classes for both scenarios
        from src.ingest.pre_processor.docling_pre_processor import Tag, NavigableString, PageElement
        
        # These classes should be defined whether BS4 is available or not
        assert Tag is not None
        assert NavigableString is not None
        assert PageElement is not None
        
        # _BS4_AVAILABLE should be set to a boolean value
        assert isinstance(_BS4_AVAILABLE, bool)
