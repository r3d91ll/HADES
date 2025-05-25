"""
Tests for the DoclingAdapter class in src/ingest/adapters/docling_adapter.py
"""
import os
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch, mock_open

from src.ingest.adapters.docling_adapter import DoclingAdapter


class MockDocumentConverter:
    """Mock DocumentConverter for testing DoclingAdapter without docling dependency."""
    
    def convert(self, file_path: str, input_format: Optional[Any] = None) -> Any:
        """Mock conversion method."""
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Mock Markdown Document"
        
        mock_result = MagicMock()
        mock_result.document = mock_doc
        return mock_result


class MockInputFormat:
    """Mock InputFormat for testing format inference."""
    
    class PDF:
        name = "pdf"
    
    class HTML:
        name = "html"
    
    class MARKDOWN:
        name = "markdown"
    
    class DOCX:
        name = "docx"


class TestDoclingAdapter:
    """Test suite for DoclingAdapter."""
    
    @pytest.fixture
    def mock_imports(self):
        """Mock the docling imports to avoid actual dependency."""
        with patch("src.ingest.adapters.docling_adapter.DocumentConverter", MockDocumentConverter), \
             patch("src.ingest.adapters.docling_adapter.InputFormat", MockInputFormat):
            yield
    
    @pytest.fixture
    def adapter(self, mock_imports):
        """Create a DoclingAdapter instance with mocked dependencies."""
        return DoclingAdapter()
    
    @pytest.fixture
    def adapter_with_options(self, mock_imports):
        """Create a DoclingAdapter instance with custom options."""
        return DoclingAdapter(options={"custom_option": "value"})
    
    def test_init_with_default_options(self, mock_imports):
        """Test initialization with default options."""
        adapter = DoclingAdapter()
        assert adapter.options == {}
        assert isinstance(adapter.converter, MockDocumentConverter)
    
    def test_init_with_custom_options(self, mock_imports):
        """Test initialization with custom options."""
        options = {"lang": "en", "model_size": "large"}
        adapter = DoclingAdapter(options=options)
        assert adapter.options == options
        assert isinstance(adapter.converter, MockDocumentConverter)
    
    def test_init_raises_import_error(self):
        """Test initialization raises ImportError when docling is not available."""
        with patch("src.ingest.adapters.docling_adapter.DocumentConverter", None), \
             patch("src.ingest.adapters.docling_adapter.InputFormat", None):
            with pytest.raises(ImportError) as excinfo:
                DoclingAdapter()
            assert "Docling is not installed" in str(excinfo.value)
    
    def test_parse(self, adapter):
        """Test parse method with mock file."""
        file_path = Path("/path/to/document.pdf")
        
        result = adapter.parse(file_path)
        
        assert isinstance(result, dict)
        assert result["source"] == str(file_path)
        assert result["content"] == "# Mock Markdown Document"
        # docling_document field has been removed
        assert result["format"] == "PDF"
    
    def test_parse_with_str_path(self, adapter):
        """Test parse method with string file path."""
        file_path = "/path/to/document.pdf"
        
        result = adapter.parse(file_path)
        
        assert isinstance(result, dict)
        assert result["source"] == file_path
        assert result["content"] == "# Mock Markdown Document"
    
    def test_analyze_text(self, adapter):
        """Test analyze_text method."""
        text = "Sample text for analysis"
        
        result = adapter.analyze_text(text)
        
        assert isinstance(result, dict)
        assert result["text"] == text
        assert result["analysis"]["type"] == "basic_analysis"
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample file content")
    def test_analyze_file(self, mock_file, adapter):
        """Test analyze_file method."""
        file_path = Path("/path/to/file.txt")
        
        result = adapter.analyze_file(file_path)
        
        mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
        assert isinstance(result, dict)
        assert result["text"] == "Sample file content"
        assert result["analysis"]["type"] == "basic_analysis"
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample file content")
    def test_analyze_file_with_str_path(self, mock_file, adapter):
        """Test analyze_file method with string path."""
        file_path = "/path/to/file.txt"
        
        result = adapter.analyze_file(file_path)
        
        mock_file.assert_called_once_with(Path(file_path), 'r', encoding='utf-8')
        assert isinstance(result, dict)
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample file content")
    def test_analyze_file_filenotfound_error(self, mock_file, adapter):
        """Test analyze_file method with FileNotFoundError."""
        file_path = Path("/path/to/nonexistent.txt")
        mock_file.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            adapter.analyze_file(file_path)
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample file content")
    def test_extract_entities(self, mock_file, adapter):
        """Test extract_entities method."""
        file_path = Path("/path/to/file.txt")
        
        result = adapter.extract_entities(file_path)
        
        mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["type"] == "entity"
        assert result[0]["value"] == "placeholder"
        assert result[0]["confidence"] == 0.9
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample file content")
    def test_extract_entities_with_str_path(self, mock_file, adapter):
        """Test extract_entities method with string path."""
        file_path = "/path/to/file.txt"
        
        result = adapter.extract_entities(file_path)
        
        mock_file.assert_called_once_with(Path(file_path), 'r', encoding='utf-8')
        assert isinstance(result, list)
        assert len(result) > 0
    
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_entities_file_error(self, mock_file, adapter):
        """Test extract_entities method with file error."""
        file_path = Path("/path/to/file.txt")
        mock_file.side_effect = IOError("File error")
        
        with pytest.raises(IOError):
            adapter.extract_entities(file_path)
    
    def test_extract_relationships(self, adapter):
        """Test extract_relationships method."""
        text = "Sample text for relationship extraction"
        
        result = adapter.extract_relationships(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["type"] == "relationship"
        assert result[0]["source"] == "placeholder_source"
        assert result[0]["target"] == "placeholder_target"
        assert result[0]["confidence"] == 0.8
    
    def test_extract_keywords(self, adapter):
        """Test extract_keywords method."""
        text = "Sample text for keyword extraction"
        
        result = adapter.extract_keywords(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["keyword"] == "placeholder_keyword"
        assert result[0]["score"] == 0.95
    
    def test_infer_format_pdf(self, mock_imports):
        """Test _infer_format with PDF file."""
        adapter = DoclingAdapter()
        file_path = Path("document.pdf")
        
        result = adapter._infer_format(file_path)
        
        assert result == "PDF"
    
    def test_infer_format_html(self, mock_imports):
        """Test _infer_format with HTML files."""
        adapter = DoclingAdapter()
        
        assert adapter._infer_format(Path("document.html")) == "HTML"
        assert adapter._infer_format(Path("document.htm")) == "HTML"
    
    def test_infer_format_markdown(self, mock_imports):
        """Test _infer_format with Markdown file."""
        adapter = DoclingAdapter()
        file_path = Path("document.md")
        
        result = adapter._infer_format(file_path)
        
        assert result == "MARKDOWN"
    
    def test_infer_format_docx(self, mock_imports):
        """Test _infer_format with DOCX file."""
        adapter = DoclingAdapter()
        file_path = Path("document.docx")
        
        result = adapter._infer_format(file_path)
        
        assert result == "DOCX"
    
    def test_infer_format_unknown(self, mock_imports):
        """Test _infer_format with unknown file type."""
        adapter = DoclingAdapter()
        file_path = Path("document.unknown")
        
        result = adapter._infer_format(file_path)
        
        assert result is None
    
    def test_infer_format_no_input_format(self):
        """Test _infer_format when InputFormat is None."""
        with patch("src.ingest.adapters.docling_adapter.DocumentConverter", MockDocumentConverter), \
             patch("src.ingest.adapters.docling_adapter.DOCLING_AVAILABLE", False):
            adapter = DoclingAdapter()
            file_path = Path("document.pdf")
            
            result = adapter._infer_format(file_path)
            
            assert result is None
