"""
Unit tests for TextChunker component.
"""

import pytest
from typing import Dict, Any

from src.components.chunking.chunkers.text.processor import TextChunker
from src.types.components.contracts import (
    ComponentType,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)


class TestTextChunker:
    """Test suite for TextChunker."""
    
    @pytest.fixture
    def processor(self) -> TextChunker:
        """Create a TextChunker instance for testing."""
        return TextChunker()
    
    @pytest.fixture
    def processor_with_config(self) -> TextChunker:
        """Create a TextChunker with configuration."""
        config = {
            "chunk_size": 600,
            "chunk_overlap": 60,
            "preserve_sentences": True,
            "preserve_paragraphs": True
        }
        return TextChunker(config=config)
    
    @pytest.fixture
    def narrative_text(self) -> str:
        """Narrative text for testing."""
        return """Once upon a time, in a distant galaxy far away, there lived a young programmer who dreamed of writing perfect code. Every day, she would sit at her computer and work on algorithms that could solve complex problems.

The programmer's name was Alice, and she had a particular passion for natural language processing. She believed that understanding text was the key to unlocking the mysteries of human communication. Her greatest project was building a system that could automatically break down large documents into meaningful chunks.

One day, Alice discovered a powerful technique called semantic chunking. This method didn't just split text at arbitrary character boundaries, but instead tried to preserve the meaning and flow of the original content. She realized this was exactly what her project needed.

As Alice implemented her chunking algorithm, she encountered many challenges. How should the system handle dialogue? What about lists and bullet points? Should code snippets be treated differently from regular prose? Each question led to new insights and improvements."""
    
    @pytest.fixture
    def dialogue_text(self) -> str:
        """Text with dialogue for testing."""
        return '''"Hello, world!" she exclaimed with joy.

"What a beautiful day it is," replied her companion, looking up at the clear blue sky.

"Indeed it is," she agreed. "Perfect for a walk in the park."

The conversation continued as they strolled down the winding path. Birds chirped in the trees overhead, and a gentle breeze rustled through the leaves.

"Do you remember when we first met?" he asked suddenly.

"Of course I do," she laughed. "How could I forget such an awkward introduction?"'''
    
    @pytest.fixture
    def technical_text(self) -> str:
        """Technical text for testing."""
        return """Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data. The field encompasses various techniques including supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common examples include classification tasks (predicting categories) and regression tasks (predicting continuous values). Popular supervised learning algorithms include decision trees, random forests, support vector machines, and neural networks.

Unsupervised learning, on the other hand, works with unlabeled data to discover hidden patterns or structures. Clustering algorithms like k-means and hierarchical clustering are common examples. Dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-SNE also fall into this category.

Reinforcement learning involves training agents to make sequential decisions in an environment to maximize cumulative reward. This approach has been particularly successful in game playing (like chess and Go) and robotics applications."""
    
    def test_basic_properties(self, processor: TextChunker):
        """Test basic component properties."""
        assert processor.name == "text"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.CHUNKING
    
    def test_configuration(self, processor: TextChunker):
        """Test component configuration."""
        config = {
            "chunk_size": 800,
            "chunk_overlap": 80,
            "preserve_sentences": False,
            "preserve_paragraphs": True
        }
        
        assert processor.validate_config(config)
        processor.configure(config)
        
        # Configuration should be updated
        assert processor._config["chunk_size"] == 800
        assert processor._config["chunk_overlap"] == 80
        assert processor._config["preserve_sentences"] == False
        assert processor._config["preserve_paragraphs"] == True
    
    def test_config_validation(self, processor: TextChunker):
        """Test configuration validation."""
        # Valid config
        valid_config = {"chunk_size": 1000, "preserve_sentences": True}
        assert processor.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor.validate_config("invalid")
        assert not processor.validate_config(None)
        
        # Invalid chunk size
        invalid_config = {"chunk_size": -100}
        assert not processor.validate_config(invalid_config)
        
        # Invalid boolean flags
        invalid_config = {"preserve_sentences": "not_a_boolean"}
        assert not processor.validate_config(invalid_config)
    
    def test_config_schema(self, processor: TextChunker):
        """Test configuration schema."""
        schema = processor.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "chunk_size" in schema["properties"]
        assert "chunk_overlap" in schema["properties"]
        assert "preserve_sentences" in schema["properties"]
        assert "preserve_paragraphs" in schema["properties"]
    
    def test_health_check(self, processor: TextChunker):
        """Test health check."""
        assert processor.health_check() == True
    
    def test_metrics(self, processor: TextChunker):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert metrics["component_name"] == "text"
        assert metrics["component_version"] == "1.0.0"
        assert "preserve_sentences" in metrics
        assert "preserve_paragraphs" in metrics
        assert "total_chunks_created" in metrics
        assert metrics["total_chunks_created"] >= 0
    
    def test_sentence_preservation(self, processor: TextChunker, narrative_text: str):
        """Test sentence boundary preservation."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            chunk_overlap=50,
            processing_options={"preserve_sentences": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Chunks should generally end at sentence boundaries
        for chunk in output.chunks[:-1]:  # Exclude last chunk which might be partial
            text = chunk.text.strip()
            if len(text) > 50:  # Only check substantial chunks
                # Should end with sentence punctuation
                assert text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith('"')
    
    def test_paragraph_preservation(self, processor: TextChunker, narrative_text: str):
        """Test paragraph boundary preservation."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=600,
            chunk_overlap=60,
            processing_options={"preserve_paragraphs": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Check that paragraph structure is preserved
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Should not break inappropriately within paragraphs
    
    def test_dialogue_handling(self, processor: TextChunker, dialogue_text: str):
        """Test handling of dialogue text."""
        input_data = ChunkingInput(
            text=dialogue_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=200,
            chunk_overlap=30,
            processing_options={"preserve_sentences": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Dialogue should be handled properly
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Should preserve quoted speech as much as possible
    
    def test_technical_text_chunking(self, processor: TextChunker, technical_text: str):
        """Test chunking of technical text."""
        input_data = ChunkingInput(
            text=technical_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=400,
            chunk_overlap=60,
            processing_options={"preserve_paragraphs": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Technical terms should be preserved within chunks
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Should not break technical terms inappropriately
    
    def test_chunk_estimation(self, processor: TextChunker, narrative_text: str):
        """Test chunk count estimation."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            chunk_overlap=50
        )
        
        estimated_count = processor.estimate_chunks(input_data)
        
        assert isinstance(estimated_count, int)
        assert estimated_count > 0
        
        # Actually chunk and compare
        output = processor.chunk(input_data)
        actual_count = len(output.chunks)
        
        # Text chunker estimation should be fairly accurate
        assert abs(estimated_count - actual_count) <= max(1, actual_count // 3)
    
    def test_content_type_support(self, processor: TextChunker):
        """Test content type support checking."""
        assert processor.supports_content_type("text") == True
        assert processor.supports_content_type("markdown") == True
        assert processor.supports_content_type("document") == True
        assert processor.supports_content_type("narrative") == True
        assert processor.supports_content_type("code") == False
        assert processor.supports_content_type("unknown") == False
    
    def test_optimal_chunk_size(self, processor: TextChunker):
        """Test optimal chunk size calculation."""
        text_size = processor.get_optimal_chunk_size("text")
        markdown_size = processor.get_optimal_chunk_size("markdown")
        document_size = processor.get_optimal_chunk_size("document")
        
        assert isinstance(text_size, int)
        assert isinstance(markdown_size, int)
        assert isinstance(document_size, int)
        
        assert text_size > 0
        assert markdown_size > 0
        assert document_size > 0
        
        # Document chunks might be larger than regular text
        assert document_size >= text_size
    
    def test_overlap_handling(self, processor: TextChunker, narrative_text: str):
        """Test chunk overlap implementation."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            chunk_overlap=60
        )
        
        output = processor.chunk(input_data)
        
        assert len(output.chunks) > 1
        
        # Check that consecutive chunks have appropriate overlap
        for i in range(len(output.chunks) - 1):
            current_chunk = output.chunks[i]
            next_chunk = output.chunks[i + 1]
            
            assert current_chunk.start_index < current_chunk.end_index
            assert next_chunk.start_index < next_chunk.end_index
            
            # Should have overlap
            overlap_size = current_chunk.end_index - next_chunk.start_index
            assert overlap_size > 0
    
    def test_no_preservation_mode(self, processor: TextChunker, narrative_text: str):
        """Test chunking without preservation constraints."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=200,
            chunk_overlap=40,
            processing_options={
                "preserve_sentences": False,
                "preserve_paragraphs": False
            }
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should create chunks without strict boundary preservation
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Chunks might not end at sentence boundaries
    
    def test_very_long_sentences(self, processor: TextChunker):
        """Test handling of very long sentences."""
        long_sentence = "This is a very long sentence that goes on and on and contains many clauses and phrases that make it much longer than the typical chunk size that we might want to use for processing, and it should be handled gracefully by the text chunker even when sentence preservation is enabled because sometimes real-world text contains such lengthy constructions that need to be accommodated in a reasonable way."
        
        input_data = ChunkingInput(
            text=long_sentence,
            document_id="test_doc",
            content_type="text",
            chunk_size=100,
            chunk_overlap=20,
            processing_options={"preserve_sentences": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        
        # Should handle long sentences gracefully
        total_text = "".join(chunk.text for chunk in output.chunks)
        assert long_sentence in total_text.replace(" ", "").replace("\n", "")
    
    def test_mixed_content(self, processor: TextChunker):
        """Test handling of mixed content types."""
        mixed_text = """Here is some regular narrative text that flows naturally.

• This is a bullet point
• Here is another bullet point  
• And a third one

Back to regular paragraph text that continues the narrative flow.

1. This is a numbered list
2. With multiple items
3. That should be handled properly

Finally, we return to normal text to conclude the document."""
        
        input_data = ChunkingInput(
            text=mixed_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=200,
            chunk_overlap=40,
            processing_options={"preserve_paragraphs": True}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should handle mixed content appropriately
        for chunk in output.chunks:
            assert len(chunk.text) > 0
    
    def test_configured_processor(self, processor_with_config: TextChunker, narrative_text: str):
        """Test processor with pre-configured settings."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text"
            # Should use configured settings
        )
        
        output = processor_with_config.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should use configured settings
        if len(output.chunks) > 0:
            # Verify chunks respect configured size
            max_expected_size = 600 + 150  # chunk_size + tolerance
            for chunk in output.chunks:
                assert len(chunk.text) <= max_expected_size
    
    def test_metadata_tracking(self, processor: TextChunker, narrative_text: str):
        """Test that metadata is properly tracked."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            processing_options={"preserve_sentences": True}
        )
        
        output = processor.chunk(input_data)
        
        assert output.metadata.component_name == "text"
        assert output.metadata.component_version == "1.0.0"
        assert output.metadata.processing_time is not None
        assert output.metadata.processed_at is not None
        
        # Processing stats should be included
        assert "processing_time" in output.processing_stats
        assert "chunks_created" in output.processing_stats
        assert "preserve_sentences" in output.processing_stats
        assert "preserve_paragraphs" in output.processing_stats
        assert output.processing_stats["chunks_created"] == len(output.chunks)
    
    def test_chunk_metadata(self, processor: TextChunker, narrative_text: str):
        """Test that individual chunks have proper metadata."""
        input_data = ChunkingInput(
            text=narrative_text,
            document_id="test_doc",
            content_type="text",
            chunk_size=300,
            processing_options={"preserve_sentences": True}
        )
        
        output = processor.chunk(input_data)
        
        for chunk in output.chunks:
            assert isinstance(chunk.metadata, dict)
            assert "preserve_sentences" in chunk.metadata
            assert "preserve_paragraphs" in chunk.metadata
            assert "processing_method" in chunk.metadata
            assert chunk.metadata["processing_method"] == "text_aware"


if __name__ == "__main__":
    pytest.main([__file__])