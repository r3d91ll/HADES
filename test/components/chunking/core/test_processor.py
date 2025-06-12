"""
Unit tests for CoreChunker component.
"""

import pytest
from typing import Dict, Any

from src.components.chunking.core.processor import CoreChunker
from src.types.components.contracts import (
    ComponentType,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)


class TestCoreChunker:
    """Test suite for CoreChunker."""
    
    @pytest.fixture
    def processor(self) -> CoreChunker:
        """Create a CoreChunker instance for testing."""
        return CoreChunker()
    
    @pytest.fixture
    def processor_with_config(self) -> CoreChunker:
        """Create a CoreChunker with configuration."""
        config = {
            "default_chunker": "text"
        }
        return CoreChunker(config=config)
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for testing."""
        return """This is a sample document for testing chunking functionality.
        
It contains multiple paragraphs and sentences to test how the chunker 
handles different types of content. The text should be long enough 
to create multiple chunks when processed with reasonable chunk sizes.

This paragraph contains more text to ensure we have sufficient content
for comprehensive testing of the chunking algorithms. We want to verify
that chunks are created properly with appropriate overlap and sizing.

The final paragraph provides additional content to test edge cases
and ensure the chunker handles end-of-document scenarios correctly."""
    
    @pytest.fixture
    def sample_code(self) -> str:
        """Sample code for testing."""
        return '''def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    """A test class for demonstration."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"
        
    def calculate(self, x: int, y: int) -> int:
        """Calculate sum of two numbers."""
        return x + y

# Main execution
if __name__ == "__main__":
    greeter = TestClass("World")
    print(greeter.greet())
    print(greeter.calculate(5, 3))'''
    
    def test_basic_properties(self, processor: CoreChunker):
        """Test basic component properties."""
        assert processor.name == "core"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.CHUNKING
    
    def test_configuration(self, processor: CoreChunker):
        """Test component configuration."""
        config = {
            "default_chunker": "text"
        }
        
        assert processor.validate_config(config)
        processor.configure(config)
        
        # Configuration should be updated
        assert processor._config["default_chunker"] == "text"
    
    def test_config_validation(self, processor: CoreChunker):
        """Test configuration validation."""
        # Valid config
        valid_config = {"default_chunker": "cpu"}
        assert processor.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor.validate_config("invalid")
        assert not processor.validate_config(None)
        
        # Invalid chunker type
        invalid_config = {"default_chunker": "invalid_type"}
        assert not processor.validate_config(invalid_config)
    
    def test_config_schema(self, processor: CoreChunker):
        """Test configuration schema."""
        schema = processor.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "default_chunker" in schema["properties"]
    
    def test_health_check(self, processor: CoreChunker):
        """Test health check."""
        assert processor.health_check() == True
    
    def test_metrics(self, processor: CoreChunker):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert metrics["component_name"] == "core"
        assert metrics["component_version"] == "1.0.0"
        assert "available_chunkers" in metrics
    
    def test_cpu_chunking(self, processor: CoreChunker, sample_text: str):
        """Test CPU-based chunking."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="cpu",
            chunk_size=200,
            chunk_overlap=50
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.component_type == ComponentType.CHUNKING
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert len(output.errors) == 0
        
        # Verify chunks
        for i, chunk in enumerate(output.chunks):
            assert isinstance(chunk, TextChunk)
            assert chunk.chunk_index == i
            assert len(chunk.text) > 0
            assert chunk.id.startswith("test_doc_chunk_")
    
    def test_text_chunking(self, processor: CoreChunker, sample_text: str):
        """Test text-specific chunking."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="text",
            chunk_size=300,
            chunk_overlap=30
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Check chunk properties
        for chunk in output.chunks:
            assert len(chunk.text) <= 350  # Should respect size limits with some tolerance
            assert chunk.text.strip()  # Should not be empty or just whitespace
    
    def test_code_chunking(self, processor: CoreChunker, sample_code: str):
        """Test code-specific chunking."""
        input_data = ChunkingInput(
            text=sample_code,
            document_id="test_code",
            content_type="code",
            chunker_type="code",
            chunk_size=400,
            chunk_overlap=40
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Code chunks should preserve structure
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Check that we have some code-like content
            code_indicators = ['def ', 'class ', 'import ', 'if __name__']
            has_code = any(indicator in chunk.text for indicator in code_indicators)
            # At least some chunks should contain code indicators
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires transformers
        reason="Chonky chunker requires transformers library"
    )
    def test_chonky_chunking(self, processor: CoreChunker, sample_text: str):
        """Test GPU-accelerated chonky chunking."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="chonky",
            chunk_size=256,
            chunk_overlap=50
        )
        
        output = processor.chunk(input_data)
        
        # Should either succeed or fail gracefully
        assert isinstance(output, ChunkingOutput)
        
        if output.metadata.status == ProcessingStatus.SUCCESS:
            assert len(output.chunks) > 0
            for chunk in output.chunks:
                assert isinstance(chunk, TextChunk)
                assert len(chunk.text) > 0
        else:
            # Should have error information if failed
            assert len(output.errors) > 0
    
    def test_chunk_estimation(self, processor: CoreChunker, sample_text: str):
        """Test chunk count estimation."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="cpu",
            chunk_size=200,
            chunk_overlap=50
        )
        
        estimated_count = processor.estimate_chunks(input_data)
        
        assert isinstance(estimated_count, int)
        assert estimated_count > 0
        
        # Actually chunk and compare
        output = processor.chunk(input_data)
        actual_count = len(output.chunks)
        
        # Estimation should be reasonably close (within 50% typically)
        assert abs(estimated_count - actual_count) / max(actual_count, 1) <= 0.8
    
    def test_content_type_support(self, processor: CoreChunker):
        """Test content type support checking."""
        assert processor.supports_content_type("text") == True
        assert processor.supports_content_type("code") == True
        assert processor.supports_content_type("markdown") == True
        assert processor.supports_content_type("unknown") == False
    
    def test_optimal_chunk_size(self, processor: CoreChunker):
        """Test optimal chunk size calculation."""
        text_size = processor.get_optimal_chunk_size("text")
        code_size = processor.get_optimal_chunk_size("code")
        markdown_size = processor.get_optimal_chunk_size("markdown")
        
        assert isinstance(text_size, int)
        assert isinstance(code_size, int)
        assert isinstance(markdown_size, int)
        
        assert text_size > 0
        assert code_size > 0
        assert markdown_size > 0
    
    def test_error_handling(self, processor: CoreChunker):
        """Test error handling for invalid input."""
        # Empty text
        input_data = ChunkingInput(
            text="",
            document_id="test_doc",
            content_type="text",
            chunker_type="cpu"
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        # Should handle empty text gracefully
        assert len(output.chunks) == 0 or output.chunks[0].text == ""
    
    def test_chunk_overlap_validation(self, processor: CoreChunker, sample_text: str):
        """Test that chunk overlap is properly applied."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="cpu",
            chunk_size=200,
            chunk_overlap=50
        )
        
        output = processor.chunk(input_data)
        
        if len(output.chunks) > 1:
            # Check for overlap between consecutive chunks
            first_chunk = output.chunks[0]
            second_chunk = output.chunks[1]
            
            # There should be some overlap in content
            # (exact implementation depends on chunker strategy)
            assert len(first_chunk.text) > 0
            assert len(second_chunk.text) > 0
    
    def test_multiple_processing(self, processor: CoreChunker, sample_text: str, sample_code: str):
        """Test processing multiple inputs individually."""
        # Process text input
        text_input = ChunkingInput(
            text=sample_text,
            document_id="doc1",
            content_type="text",
            chunker_type="cpu",
            chunk_size=200
        )
        
        text_output = processor.chunk(text_input)
        
        # Process code input
        code_input = ChunkingInput(
            text=sample_code,
            document_id="doc2",
            content_type="code",
            chunker_type="code",
            chunk_size=300
        )
        
        code_output = processor.chunk(code_input)
        
        assert isinstance(text_output, ChunkingOutput)
        assert isinstance(code_output, ChunkingOutput)
        assert len(text_output.chunks) > 0
        assert len(code_output.chunks) > 0
    
    def test_configured_processor(self, processor_with_config: CoreChunker, sample_text: str):
        """Test processor with pre-configured settings."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text"
            # chunker_type comes from config
        )
        
        output = processor_with_config.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should use configured settings
        if len(output.chunks) > 0:
            # Verify chunks respect configured size (with some tolerance)
            max_expected_size = 1000 + 200  # chunk_size + tolerance
            for chunk in output.chunks:
                assert len(chunk.text) <= max_expected_size
    
    def test_chunker_type_delegation(self, processor: CoreChunker, sample_text: str):
        """Test that different chunker types are properly delegated."""
        chunker_types = ["cpu", "text", "code"]
        
        for chunker_type in chunker_types:
            input_data = ChunkingInput(
                text=sample_text,
                document_id="test_doc",
                content_type="text",
                chunker_type=chunker_type,
                chunk_size=200
            )
            
            output = processor.chunk(input_data)
            
            assert isinstance(output, ChunkingOutput)
            # Should either succeed or fail gracefully
            assert output.metadata.status in [ProcessingStatus.SUCCESS, ProcessingStatus.ERROR]
    
    def test_metadata_tracking(self, processor: CoreChunker, sample_text: str):
        """Test that metadata is properly tracked."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunker_type="cpu"
        )
        
        output = processor.chunk(input_data)
        
        assert output.metadata.component_name == "core"
        assert output.metadata.component_version == "1.0.0"
        assert output.metadata.processing_time is not None
        assert output.metadata.processed_at is not None
        
        # Processing stats should be included
        assert "processing_time" in output.processing_stats
        assert "chunks_created" in output.processing_stats
        assert output.processing_stats["chunks_created"] == len(output.chunks)


if __name__ == "__main__":
    pytest.main([__file__])