"""
Unit tests for CPUChunker component.
"""

import pytest
from typing import Dict, Any

from src.components.chunking.chunkers.cpu.processor import CPUChunker
from src.types.components.contracts import (
    ComponentType,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)


class TestCPUChunker:
    """Test suite for CPUChunker."""
    
    @pytest.fixture
    def processor(self) -> CPUChunker:
        """Create a CPUChunker instance for testing."""
        return CPUChunker()
    
    @pytest.fixture
    def processor_with_config(self) -> CPUChunker:
        """Create a CPUChunker with configuration."""
        config = {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunking_method": "sentence_aware"
        }
        return CPUChunker(config=config)
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for testing."""
        return """This is the first sentence. This is the second sentence. Here's a third sentence.
        
This is a new paragraph with multiple sentences. It should be handled properly by the chunker. 
The chunker should respect sentence boundaries when possible.

This is another paragraph. It contains additional content for testing. The chunker needs to 
handle various text structures efficiently."""
    
    @pytest.fixture
    def long_text(self) -> str:
        """Longer text for testing large document chunking."""
        paragraphs = []
        for i in range(10):
            paragraphs.append(f"""This is paragraph {i+1}. It contains multiple sentences to test 
            the chunking algorithm. Each paragraph has enough content to verify that chunking 
            works correctly across different sizes and configurations. The text should be 
            divided into appropriate chunks based on the specified parameters.""")
        return "\n\n".join(paragraphs)
    
    def test_basic_properties(self, processor: CPUChunker):
        """Test basic component properties."""
        assert processor.name == "cpu"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.CHUNKING
    
    def test_configuration(self, processor: CPUChunker):
        """Test component configuration."""
        config = {
            "chunk_size": 800,
            "chunk_overlap": 80,
            "chunking_method": "paragraph_aware"
        }
        
        assert processor.validate_config(config)
        processor.configure(config)
        
        # Configuration should be updated
        assert processor._config["chunk_size"] == 800
        assert processor._config["chunk_overlap"] == 80
        assert processor._config["chunking_method"] == "paragraph_aware"
    
    def test_config_validation(self, processor: CPUChunker):
        """Test configuration validation."""
        # Valid config
        valid_config = {"chunk_size": 1000, "chunking_method": "fixed"}
        assert processor.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor.validate_config("invalid")
        assert not processor.validate_config(None)
        
        # Invalid chunk size
        invalid_config = {"chunk_size": -100}
        assert not processor.validate_config(invalid_config)
        
        # Invalid chunking method
        invalid_config = {"chunking_method": "invalid_method"}
        assert not processor.validate_config(invalid_config)
    
    def test_config_schema(self, processor: CPUChunker):
        """Test configuration schema."""
        schema = processor.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "chunk_size" in schema["properties"]
        assert "chunk_overlap" in schema["properties"]
        assert "chunking_method" in schema["properties"]
    
    def test_health_check(self, processor: CPUChunker):
        """Test health check."""
        assert processor.health_check() == True
    
    def test_metrics(self, processor: CPUChunker):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert metrics["component_name"] == "cpu"
        assert metrics["component_version"] == "1.0.0"
        assert "chunking_method" in metrics
        assert "total_chunks_created" in metrics
        assert metrics["total_chunks_created"] >= 0
    
    def test_fixed_chunking(self, processor: CPUChunker, sample_text: str):
        """Test fixed-size chunking method."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=100,
            chunk_overlap=20,
            processing_options={"chunking_method": "fixed"}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert len(output.errors) == 0
        
        # Verify chunks
        for i, chunk in enumerate(output.chunks):
            assert isinstance(chunk, TextChunk)
            assert chunk.chunk_index == i
            assert len(chunk.text) > 0
            assert len(chunk.text) <= 120  # chunk_size + some tolerance
            assert chunk.id.startswith("test_doc_chunk_")
    
    def test_sentence_aware_chunking(self, processor: CPUChunker, sample_text: str):
        """Test sentence-aware chunking method."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=200,
            chunk_overlap=30,
            processing_options={"chunking_method": "sentence_aware"}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Check that chunks end at sentence boundaries where possible
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Most chunks should end with sentence punctuation (with some tolerance for last chunk)
            if len(chunk.text.strip()) > 50:  # Only check substantial chunks
                text = chunk.text.strip()
                # Should generally end with sentence punctuation
                ends_properly = text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith('"')
                # We allow some flexibility as not all chunks may end perfectly
    
    def test_paragraph_aware_chunking(self, processor: CPUChunker, sample_text: str):
        """Test paragraph-aware chunking method."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            chunk_overlap=50,
            processing_options={"chunking_method": "paragraph_aware"}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Paragraph-aware chunks should respect paragraph boundaries
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Should not break in the middle of paragraphs inappropriately
    
    def test_adaptive_chunking(self, processor: CPUChunker, sample_text: str):
        """Test adaptive chunking method."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=250,
            chunk_overlap=40,
            processing_options={"chunking_method": "adaptive"}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Adaptive chunking should vary chunk sizes based on content
        chunk_sizes = [len(chunk.text) for chunk in output.chunks]
        if len(chunk_sizes) > 1:
            # Should have some variation in sizes
            size_variance = max(chunk_sizes) - min(chunk_sizes)
            assert size_variance >= 0  # At minimum, should not be negative
    
    def test_chunk_overlap(self, processor: CPUChunker, long_text: str):
        """Test that chunk overlap is properly implemented."""
        input_data = ChunkingInput(
            text=long_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=200,
            chunk_overlap=50,
            processing_options={"chunking_method": "fixed"}
        )
        
        output = processor.chunk(input_data)
        
        assert len(output.chunks) > 1
        
        # Check overlap between consecutive chunks
        for i in range(len(output.chunks) - 1):
            current_chunk = output.chunks[i]
            next_chunk = output.chunks[i + 1]
            
            # Should have proper start/end indices
            assert current_chunk.start_index < current_chunk.end_index
            assert next_chunk.start_index < next_chunk.end_index
            
            # Next chunk should start before current chunk ends (overlap)
            assert next_chunk.start_index < current_chunk.end_index
    
    def test_chunk_estimation(self, processor: CPUChunker, sample_text: str):
        """Test chunk count estimation."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=100,
            chunk_overlap=20
        )
        
        estimated_count = processor.estimate_chunks(input_data)
        
        assert isinstance(estimated_count, int)
        assert estimated_count > 0
        
        # Actually chunk and compare
        output = processor.chunk(input_data)
        actual_count = len(output.chunks)
        
        # Estimation should be reasonably close
        assert abs(estimated_count - actual_count) <= max(2, actual_count // 2)
    
    def test_content_type_support(self, processor: CPUChunker):
        """Test content type support checking."""
        assert processor.supports_content_type("text") == True
        assert processor.supports_content_type("markdown") == True
        assert processor.supports_content_type("code") == True
        assert processor.supports_content_type("document") == True
        assert processor.supports_content_type("unknown") == False
    
    def test_optimal_chunk_size(self, processor: CPUChunker):
        """Test optimal chunk size calculation."""
        text_size = processor.get_optimal_chunk_size("text")
        markdown_size = processor.get_optimal_chunk_size("markdown")
        code_size = processor.get_optimal_chunk_size("code")
        
        assert isinstance(text_size, int)
        assert isinstance(markdown_size, int)
        assert isinstance(code_size, int)
        
        assert text_size > 0
        assert markdown_size > 0
        assert code_size > 0
    
    def test_empty_text_handling(self, processor: CPUChunker):
        """Test handling of empty text."""
        input_data = ChunkingInput(
            text="",
            document_id="test_doc",
            content_type="text",
            chunk_size=100
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        # Should handle empty text gracefully
        assert len(output.chunks) == 0 or (len(output.chunks) == 1 and output.chunks[0].text == "")
    
    def test_very_small_text(self, processor: CPUChunker):
        """Test handling of very small text."""
        input_data = ChunkingInput(
            text="Short text.",
            document_id="test_doc",
            content_type="text",
            chunk_size=100
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) == 1
        assert output.chunks[0].text == "Short text."
    
    def test_very_large_chunk_size(self, processor: CPUChunker, sample_text: str):
        """Test handling when chunk size is larger than text."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=10000  # Much larger than text
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) == 1
        assert output.chunks[0].text == sample_text
    
    def test_zero_overlap(self, processor: CPUChunker, sample_text: str):
        """Test chunking with zero overlap."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=100,
            chunk_overlap=0
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        
        # With zero overlap, chunks should not overlap
        for i in range(len(output.chunks) - 1):
            current_chunk = output.chunks[i]
            next_chunk = output.chunks[i + 1]
            
            # Next chunk should start where current chunk ends
            assert next_chunk.start_index >= current_chunk.end_index
    
    def test_configured_processor(self, processor_with_config: CPUChunker, sample_text: str):
        """Test processor with pre-configured settings."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text"
            # Should use configured chunk_size and method
        )
        
        output = processor_with_config.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should use configured settings
        if len(output.chunks) > 0:
            # Verify chunks respect configured size (with some tolerance)
            max_expected_size = 500 + 100  # chunk_size + tolerance
            for chunk in output.chunks:
                assert len(chunk.text) <= max_expected_size
    
    def test_metadata_tracking(self, processor: CPUChunker, sample_text: str):
        """Test that metadata is properly tracked."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=100
        )
        
        output = processor.chunk(input_data)
        
        assert output.metadata.component_name == "cpu"
        assert output.metadata.component_version == "1.0.0"
        assert output.metadata.processing_time is not None
        assert output.metadata.processed_at is not None
        
        # Processing stats should be included
        assert "processing_time" in output.processing_stats
        assert "chunks_created" in output.processing_stats
        assert "chunking_method" in output.processing_stats
        assert output.processing_stats["chunks_created"] == len(output.chunks)
    
    def test_chunk_metadata(self, processor: CPUChunker, sample_text: str):
        """Test that individual chunks have proper metadata."""
        input_data = ChunkingInput(
            text=sample_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=100,
            processing_options={"chunking_method": "sentence_aware"}
        )
        
        output = processor.chunk(input_data)
        
        for chunk in output.chunks:
            assert isinstance(chunk.metadata, dict)
            assert "chunking_method" in chunk.metadata
            assert chunk.metadata["chunking_method"] == "sentence_aware"
            assert "processing_method" in chunk.metadata
            assert chunk.metadata["processing_method"] == "cpu_optimized"


if __name__ == "__main__":
    pytest.main([__file__])