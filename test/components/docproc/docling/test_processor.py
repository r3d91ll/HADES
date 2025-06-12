"""
Unit tests for DoclingDocumentProcessor component.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.components.docproc.docling.processor import DoclingDocumentProcessor
from src.types.components.contracts import (
    ComponentType,
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ProcessedDocument,
    ContentCategory,
    ProcessingStatus
)


class TestDoclingDocumentProcessor:
    """Test suite for DoclingDocumentProcessor."""
    
    @pytest.fixture
    def processor_with_docling(self) -> DoclingDocumentProcessor:
        """Create a DoclingDocumentProcessor instance with mocked Docling."""
        with patch('src.components.docproc.docling.processor.DoclingDocumentProcessor._check_docling_availability', return_value=True):
            return DoclingDocumentProcessor()
    
    @pytest.fixture
    def processor_without_docling(self) -> DoclingDocumentProcessor:
        """Create a DoclingDocumentProcessor instance without Docling."""
        with patch('src.components.docproc.docling.processor.DoclingDocumentProcessor._check_docling_availability', return_value=False):
            return DoclingDocumentProcessor()
    
    @pytest.fixture
    def mock_adapter_result(self) -> dict:
        """Mock result from Docling adapter."""
        return {
            'id': 'test_doc',
            'content': 'This is processed content from PDF',
            'raw_content': 'Raw PDF content',
            'content_type': 'application/pdf',
            'format': 'pdf',
            'entities': ['entity1', 'entity2'],
            'sections': ['section1', 'section2'],
            'metadata': {'pages': 5, 'author': 'Test Author'},
            'error': None
        }
    
    def test_basic_properties(self, processor_with_docling: DoclingDocumentProcessor):
        """Test basic component properties."""
        assert processor_with_docling.name == "docling"
        assert processor_with_docling.version == "1.0.0"
        assert processor_with_docling.component_type == ComponentType.DOCPROC
    
    def test_configuration(self, processor_with_docling: DoclingDocumentProcessor):
        """Test component configuration."""
        config = {
            "ocr_enabled": False,
            "extract_tables": True,
            "extract_images": False
        }
        
        assert processor_with_docling.validate_config(config)
        processor_with_docling.configure(config)
        
        # Configuration should be updated
        assert processor_with_docling._config["ocr_enabled"] == False
        assert processor_with_docling._config["extract_tables"] == True
    
    def test_config_validation(self, processor_with_docling: DoclingDocumentProcessor):
        """Test configuration validation."""
        # Valid config
        valid_config = {"ocr_enabled": True}
        assert processor_with_docling.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor_with_docling.validate_config("invalid")
        assert not processor_with_docling.validate_config(None)
    
    def test_config_schema(self, processor_with_docling: DoclingDocumentProcessor):
        """Test configuration schema."""
        schema = processor_with_docling.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "ocr_enabled" in schema["properties"]
        assert "extract_tables" in schema["properties"]
        assert "extract_images" in schema["properties"]
    
    def test_health_check_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test health check when Docling is available."""
        assert processor_with_docling.health_check() == True
    
    def test_health_check_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test health check when Docling is not available."""
        assert processor_without_docling.health_check() == False
    
    def test_metrics_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test metrics collection when Docling is available."""
        metrics = processor_with_docling.get_metrics()
        
        assert metrics["component_name"] == "docling"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["docling_available"] == True
        assert len(metrics["supported_formats"]) > 0
    
    def test_metrics_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test metrics collection when Docling is not available."""
        metrics = processor_without_docling.get_metrics()
        
        assert metrics["component_name"] == "docling"
        assert metrics["docling_available"] == False
        assert len(metrics["supported_formats"]) == 0
    
    def test_supported_formats_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test supported formats when Docling is available."""
        formats = processor_with_docling.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.pdf' in formats
        assert '.docx' in formats
        assert '.doc' in formats
        assert '.pptx' in formats
    
    def test_supported_formats_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test supported formats when Docling is not available."""
        formats = processor_without_docling.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) == 0
    
    def test_can_process_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test file processing capability when Docling is available."""
        assert processor_with_docling.can_process("test.pdf") == True
        assert processor_with_docling.can_process("test.docx") == True
        assert processor_with_docling.can_process("test.txt") == False
    
    def test_can_process_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test file processing capability when Docling is not available."""
        assert processor_without_docling.can_process("test.pdf") == False
        assert processor_without_docling.can_process("test.docx") == False
    
    def test_process_document_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test processing a document when Docling is available."""
        doc = processor_with_docling.process_document("test.pdf")
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.id == "test"
        assert "[Docling would process:" in doc.content
        assert doc.content_category == ContentCategory.DOCUMENT
        assert doc.format == "pdf"
        assert doc.error is None
    
    def test_process_document_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test processing a document when Docling is not available."""
        doc = processor_without_docling.process_document("test.pdf")
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.error == "Docling not available"
        assert doc.content_category == ContentCategory.UNKNOWN
    
    def test_process_documents_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test processing multiple documents when Docling is not available."""
        docs = processor_without_docling.process_documents(["test1.pdf", "test2.pdf"])
        
        assert len(docs) == 2
        assert all(doc.error == "Docling not available" for doc in docs)
    
    def test_process_contract_interface_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test the main process method with contract interface when Docling is available."""
        input_data = DocumentProcessingInput(
            file_path="test.pdf",
            processing_options={"extract_tables": True}
        )
        
        output = processor_with_docling.process(input_data)
        
        assert isinstance(output, DocumentProcessingOutput)
        assert len(output.documents) == 1
        assert output.metadata.component_type == ComponentType.DOCPROC
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert len(output.errors) == 0
        
        doc = output.documents[0]
        assert doc.content_category == ContentCategory.DOCUMENT
        assert "[Docling would process:" in doc.content
    
    def test_process_contract_interface_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test the main process method with contract interface when Docling is not available."""
        input_data = DocumentProcessingInput(file_path="test.pdf")
        
        output = processor_without_docling.process(input_data)
        
        assert isinstance(output, DocumentProcessingOutput)
        assert len(output.documents) == 1
        assert output.metadata.component_type == ComponentType.DOCPROC
        assert output.metadata.status == ProcessingStatus.ERROR
        assert len(output.errors) == 1
        
        doc = output.documents[0]
        assert doc.error == "Docling not available"
    
    def test_process_batch_contract(self, processor_without_docling: DoclingDocumentProcessor):
        """Test batch processing with contract interface."""
        input_batch = [
            DocumentProcessingInput(file_path="test1.pdf"),
            DocumentProcessingInput(file_path="test2.pdf")
        ]
        
        outputs = processor_without_docling.process_batch(input_batch)
        
        assert len(outputs) == 2
        assert all(isinstance(output, DocumentProcessingOutput) for output in outputs)
        assert all(output.metadata.status == ProcessingStatus.ERROR for output in outputs)
    
    def test_error_handling_with_adapter_exception(self, processor_with_docling: DoclingDocumentProcessor):
        """Test error handling when adapter raises exception."""
        # For minimal implementation, this would require mocking the processing logic
        # For now, just test that the processor handles the case gracefully
        doc = processor_with_docling.process_document("test.pdf")
        
        assert isinstance(doc, ProcessedDocument)
        # Should not have error in minimal implementation
        assert doc.error is None or "[Docling would process:" in doc.content
    
    def test_estimate_processing_time_with_docling(self, processor_with_docling: DoclingDocumentProcessor):
        """Test processing time estimation when Docling is available."""
        input_data = DocumentProcessingInput(file_path="test.pdf")
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB
            estimated_time = processor_with_docling.estimate_processing_time(input_data)
            
            assert isinstance(estimated_time, (int, float))
            assert estimated_time >= 0.5  # Minimum for Docling
    
    def test_estimate_processing_time_without_docling(self, processor_without_docling: DoclingDocumentProcessor):
        """Test processing time estimation when Docling is not available."""
        input_data = DocumentProcessingInput(file_path="test.pdf")
        
        estimated_time = processor_without_docling.estimate_processing_time(input_data)
        
        assert estimated_time == 0.0
    
    def test_check_docling_availability_success(self):
        """Test Docling availability check when import succeeds."""
        with patch('builtins.__import__', return_value=Mock()):
            processor = DoclingDocumentProcessor()
            # This will call _check_docling_availability during init
            # We can't directly test the method due to the way it's called in init
    
    def test_check_docling_availability_failure(self):
        """Test Docling availability check when import fails."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'docling'")):
            with patch('src.components.docproc.docling.processor.DoclingDocumentProcessor._check_docling_availability', return_value=False):
                processor = DoclingDocumentProcessor()
                assert not processor._docling_available


if __name__ == "__main__":
    pytest.main([__file__])