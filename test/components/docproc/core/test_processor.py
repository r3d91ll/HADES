"""
Unit tests for CoreDocumentProcessor component.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from src.components.docproc.core.processor import CoreDocumentProcessor
from src.types.components.contracts import (
    ComponentType,
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ProcessedDocument,
    ContentCategory,
    ProcessingStatus
)


class TestCoreDocumentProcessor:
    """Test suite for CoreDocumentProcessor."""
    
    @pytest.fixture
    def processor(self) -> CoreDocumentProcessor:
        """Create a CoreDocumentProcessor instance for testing."""
        return CoreDocumentProcessor()
    
    @pytest.fixture
    def processor_with_config(self) -> CoreDocumentProcessor:
        """Create a CoreDocumentProcessor with configuration."""
        config = {
            "processing_options": {
                "extract_metadata": True,
                "extract_entities": False
            }
        }
        return CoreDocumentProcessor(config=config)
    
    @pytest.fixture
    def temp_python_file(self) -> str:
        """Create a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def hello():\n    print("Hello, world!")\n')
            return f.name
    
    @pytest.fixture
    def temp_markdown_file(self) -> str:
        """Create a temporary Markdown file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write('# Test Document\n\nThis is a test markdown document.\n')
            return f.name
    
    @pytest.fixture
    def temp_json_file(self) -> str:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"name": "test", "value": 123}\n')
            return f.name
    
    def test_basic_properties(self, processor: CoreDocumentProcessor):
        """Test basic component properties."""
        assert processor.name == "core"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.DOCPROC
    
    def test_configuration(self, processor: CoreDocumentProcessor):
        """Test component configuration."""
        config = {
            "processing_options": {
                "extract_metadata": False,
                "preserve_formatting": True
            }
        }
        
        assert processor.validate_config(config)
        processor.configure(config)
        
        # Configuration should be updated
        assert processor._config["processing_options"]["extract_metadata"] == False
        assert processor._config["processing_options"]["preserve_formatting"] == True
    
    def test_config_validation(self, processor: CoreDocumentProcessor):
        """Test configuration validation."""
        # Valid config
        valid_config = {"processing_options": {"extract_metadata": True}}
        assert processor.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor.validate_config("invalid")
        assert not processor.validate_config(None)
        
        # Invalid nested config
        invalid_config = {"processing_options": "not_a_dict"}
        assert not processor.validate_config(invalid_config)
    
    def test_config_schema(self, processor: CoreDocumentProcessor):
        """Test configuration schema."""
        schema = processor.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "manager_config" in schema["properties"]
        assert "processing_options" in schema["properties"]
    
    def test_health_check(self, processor: CoreDocumentProcessor):
        """Test health check."""
        assert processor.health_check() == True
    
    def test_metrics(self, processor: CoreDocumentProcessor):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert metrics["component_name"] == "core"
        assert metrics["component_version"] == "1.0.0"
        assert "supported_formats" in metrics
        assert "format_count" in metrics
        assert metrics["format_count"] > 0
    
    def test_supported_formats(self, processor: CoreDocumentProcessor):
        """Test supported formats."""
        formats = processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert '.py' in formats
        assert '.md' in formats
        assert '.json' in formats
        assert '.yaml' in formats
    
    def test_can_process(self, processor: CoreDocumentProcessor):
        """Test file processing capability check."""
        assert processor.can_process("test.py") == True
        assert processor.can_process("test.md") == True
        assert processor.can_process("test.json") == True
        assert processor.can_process("test.yaml") == True
        
        # Should handle most files (current implementation is permissive)
        assert processor.can_process("test.unknown") == True
    
    def test_process_python_document(self, processor: CoreDocumentProcessor, temp_python_file: str):
        """Test processing a Python document."""
        try:
            doc = processor.process_document(temp_python_file)
            
            assert isinstance(doc, ProcessedDocument)
            assert doc.content_category == ContentCategory.CODE
            assert doc.format == "py"
            assert doc.content_type == "text/x-python"
            assert "def hello():" in doc.content
            assert doc.error is None
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)
    
    def test_process_markdown_document(self, processor: CoreDocumentProcessor, temp_markdown_file: str):
        """Test processing a Markdown document."""
        try:
            doc = processor.process_document(temp_markdown_file)
            
            assert isinstance(doc, ProcessedDocument)
            assert doc.content_category == ContentCategory.MARKDOWN
            assert doc.format == "md"
            assert doc.content_type == "text/markdown"
            assert "# Test Document" in doc.content
            assert doc.error is None
            
        finally:
            Path(temp_markdown_file).unlink(missing_ok=True)
    
    def test_process_json_document(self, processor: CoreDocumentProcessor, temp_json_file: str):
        """Test processing a JSON document."""
        try:
            doc = processor.process_document(temp_json_file)
            
            assert isinstance(doc, ProcessedDocument)
            assert doc.content_category == ContentCategory.DATA
            assert doc.format == "json"
            assert doc.content_type == "application/json"
            assert "test" in doc.content
            assert doc.error is None
            
        finally:
            Path(temp_json_file).unlink(missing_ok=True)
    
    def test_process_documents_batch(self, processor: CoreDocumentProcessor, temp_python_file: str, temp_markdown_file: str):
        """Test processing multiple documents."""
        try:
            docs = processor.process_documents([temp_python_file, temp_markdown_file])
            
            assert len(docs) == 2
            assert all(isinstance(doc, ProcessedDocument) for doc in docs)
            
            # Check that both documents were processed
            categories = [doc.content_category for doc in docs]
            assert ContentCategory.CODE in categories
            assert ContentCategory.MARKDOWN in categories
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)
            Path(temp_markdown_file).unlink(missing_ok=True)
    
    def test_process_contract_interface(self, processor: CoreDocumentProcessor, temp_python_file: str):
        """Test the main process method with contract interface."""
        try:
            input_data = DocumentProcessingInput(
                file_path=temp_python_file,
                processing_options={"extract_metadata": True}
            )
            
            output = processor.process(input_data)
            
            assert isinstance(output, DocumentProcessingOutput)
            assert len(output.documents) == 1
            assert output.metadata.component_type == ComponentType.DOCPROC
            assert output.metadata.status == ProcessingStatus.SUCCESS
            assert output.metadata.processing_time is not None
            assert len(output.errors) == 0
            
            doc = output.documents[0]
            assert doc.content_category == ContentCategory.CODE
            assert "def hello():" in doc.content
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)
    
    def test_process_with_content(self, processor: CoreDocumentProcessor):
        """Test processing from content string."""
        input_data = DocumentProcessingInput(
            file_path="test.py",
            content="print('Hello from content')",
            file_type="text/x-python",
            processing_options={}
        )
        
        output = processor.process(input_data)
        
        assert isinstance(output, DocumentProcessingOutput)
        assert len(output.documents) == 1
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        doc = output.documents[0]
        assert doc.content_category == ContentCategory.CODE
        assert "Hello from content" in doc.content
    
    def test_process_batch_contract(self, processor: CoreDocumentProcessor, temp_python_file: str, temp_json_file: str):
        """Test batch processing with contract interface."""
        try:
            input_batch = [
                DocumentProcessingInput(file_path=temp_python_file),
                DocumentProcessingInput(file_path=temp_json_file)
            ]
            
            outputs = processor.process_batch(input_batch)
            
            assert len(outputs) == 2
            assert all(isinstance(output, DocumentProcessingOutput) for output in outputs)
            assert all(len(output.documents) == 1 for output in outputs)
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)
            Path(temp_json_file).unlink(missing_ok=True)
    
    def test_error_handling(self, processor: CoreDocumentProcessor):
        """Test error handling for non-existent files."""
        doc = processor.process_document("nonexistent_file.py")
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.error is not None
        assert "error_" in doc.id
        assert doc.content_category == ContentCategory.UNKNOWN
    
    def test_estimate_processing_time(self, processor: CoreDocumentProcessor, temp_python_file: str):
        """Test processing time estimation."""
        try:
            input_data = DocumentProcessingInput(file_path=temp_python_file)
            
            estimated_time = processor.estimate_processing_time(input_data)
            
            assert isinstance(estimated_time, (int, float))
            assert estimated_time > 0
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)
    
    def test_estimate_processing_time_content(self, processor: CoreDocumentProcessor):
        """Test processing time estimation for content."""
        input_data = DocumentProcessingInput(
            file_path="test.py",
            content="print('test')"
        )
        
        estimated_time = processor.estimate_processing_time(input_data)
        
        assert isinstance(estimated_time, (int, float))
        assert estimated_time >= 0.1  # Minimum estimate
    
    def test_configuration_with_processing_options(self, processor_with_config: CoreDocumentProcessor, temp_python_file: str):
        """Test processing with pre-configured options."""
        try:
            doc = processor_with_config.process_document(temp_python_file)
            
            assert isinstance(doc, ProcessedDocument)
            assert doc.error is None
            
        finally:
            Path(temp_python_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])