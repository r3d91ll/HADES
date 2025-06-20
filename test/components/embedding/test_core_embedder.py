"""
Unit tests for Core Embedder Component
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any

from src.components.embedding.core.processor import CoreEmbedder
from src.types.components.contracts import (
    ComponentType,
    EmbeddingInput,
    EmbeddingOutput,
    DocumentChunk,
    ProcessingStatus
)


class TestCoreEmbedder:
    """Test suite for core embedder factory component."""
    
    @pytest.fixture
    def core_embedder(self):
        """Create core embedder instance for testing."""
        config = {
            'embedding_type': 'cpu',
            'model_name': 'test-model',
            'batch_size': 32
        }
        return CoreEmbedder(config)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample embedding input."""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="Test chunk for core embedder.",
                metadata={"source": "test"}
            )
        ]
        
        return EmbeddingInput(
            chunks=chunks,
            model_name="test-model",
            batch_size=32,
            metadata={"test": True}
        )
    
    def test_init(self, core_embedder):
        """Test core embedder initialization."""
        assert core_embedder.name == "core"
        assert core_embedder.version == "1.0.0"
        assert core_embedder.component_type == ComponentType.EMBEDDING
        assert core_embedder._embedding_type == "cpu"
    
    def test_configure(self, core_embedder):
        """Test configuration update."""
        new_config = {
            'embedding_type': 'gpu',
            'model_name': 'new-model',
            'batch_size': 64
        }
        
        core_embedder.configure(new_config)
        
        assert core_embedder._embedding_type == "gpu"
        assert core_embedder._current_embedder is None  # Reset on config change
    
    def test_validate_config(self, core_embedder):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'embedding_type': 'gpu',
            'model_name': 'test-model'
        }
        assert core_embedder.validate_config(valid_config) is True
        
        # Invalid embedding type
        invalid_config = {
            'embedding_type': 'invalid_type'
        }
        assert core_embedder.validate_config(invalid_config) is False
        
        # Not a dict
        assert core_embedder.validate_config("not a dict") is False
    
    def test_get_config_schema(self, core_embedder):
        """Test config schema generation."""
        schema = core_embedder.get_config_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'embedding_type' in schema['properties']
        assert schema['properties']['embedding_type']['enum'] == ['cpu', 'gpu', 'encoder']
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_get_embedder_cpu(self, mock_cpu_embedder, core_embedder):
        """Test getting CPU embedder."""
        mock_instance = Mock()
        mock_cpu_embedder.return_value = mock_instance
        
        embedder = core_embedder._get_embedder()
        
        assert embedder == mock_instance
        mock_cpu_embedder.assert_called_once_with(core_embedder._config)
    
    @patch('src.components.embedding.core.processor.GPUEmbedder')  
    def test_get_embedder_gpu(self, mock_gpu_embedder, core_embedder):
        """Test getting GPU embedder."""
        core_embedder._embedding_type = 'gpu'
        mock_instance = Mock()
        mock_gpu_embedder.return_value = mock_instance
        
        embedder = core_embedder._get_embedder()
        
        assert embedder == mock_instance
        mock_gpu_embedder.assert_called_once_with(core_embedder._config)
    
    @patch('src.components.embedding.core.processor.EncoderEmbedder')
    def test_get_embedder_encoder(self, mock_encoder_embedder, core_embedder):
        """Test getting encoder embedder."""
        core_embedder._embedding_type = 'encoder'
        mock_instance = Mock()
        mock_encoder_embedder.return_value = mock_instance
        
        embedder = core_embedder._get_embedder()
        
        assert embedder == mock_instance
        mock_encoder_embedder.assert_called_once_with(core_embedder._config)
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_health_check(self, mock_cpu_embedder, core_embedder):
        """Test health check delegation."""
        mock_instance = Mock()
        mock_instance.health_check.return_value = True
        mock_cpu_embedder.return_value = mock_instance
        
        assert core_embedder.health_check() is True
        mock_instance.health_check.assert_called_once()
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_get_metrics(self, mock_cpu_embedder, core_embedder):
        """Test metrics collection."""
        # Set up mock embedder metrics
        mock_instance = Mock()
        mock_instance.get_metrics.return_value = {
            'embedder_metric': 'value'
        }
        mock_cpu_embedder.return_value = mock_instance
        
        # Force embedder initialization
        core_embedder._get_embedder()
        
        metrics = core_embedder.get_metrics()
        
        assert metrics['component_name'] == 'core'
        assert metrics['component_version'] == '1.0.0'
        assert metrics['embedding_type'] == 'cpu'
        assert metrics['embedder_initialized'] is True
        assert 'embedder_metrics' in metrics
        assert metrics['embedder_metrics']['embedder_metric'] == 'value'
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_embed_delegation(self, mock_cpu_embedder, core_embedder, sample_input):
        """Test embedding delegation to specific embedder."""
        # Set up mock embedder
        mock_instance = Mock()
        mock_output = EmbeddingOutput(
            embeddings=[],
            metadata=ComponentMetadata(
                component_type=ComponentType.EMBEDDING,
                component_name="cpu",
                component_version="1.0.0",
                status=ProcessingStatus.SUCCESS
            ),
            embedding_stats={},
            model_info={},
            errors=[]
        )
        mock_instance.embed.return_value = mock_output
        mock_cpu_embedder.return_value = mock_instance
        
        # Perform embedding
        output = core_embedder.embed(sample_input)
        
        # Verify delegation
        assert output == mock_output
        mock_instance.embed.assert_called_once_with(sample_input)
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_embed_error_handling(self, mock_cpu_embedder, core_embedder, sample_input):
        """Test error handling during embedding."""
        # Set up mock to raise exception
        mock_instance = Mock()
        mock_instance.embed.side_effect = Exception("Embedding error")
        mock_cpu_embedder.return_value = mock_instance
        
        # Perform embedding
        output = core_embedder.embed(sample_input)
        
        # Verify error handling
        assert isinstance(output, EmbeddingOutput)
        assert output.metadata.status == ProcessingStatus.ERROR
        assert len(output.errors) > 0
        assert "Core embedding failed" in output.errors[0]
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_estimate_tokens_delegation(self, mock_cpu_embedder, core_embedder, sample_input):
        """Test token estimation delegation."""
        mock_instance = Mock()
        mock_instance.estimate_tokens.return_value = 100
        mock_cpu_embedder.return_value = mock_instance
        
        estimated = core_embedder.estimate_tokens(sample_input)
        
        assert estimated == 100
        mock_instance.estimate_tokens.assert_called_once_with(sample_input)
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_supports_model_delegation(self, mock_cpu_embedder, core_embedder):
        """Test model support checking delegation."""
        mock_instance = Mock()
        mock_instance.supports_model.return_value = True
        mock_cpu_embedder.return_value = mock_instance
        
        result = core_embedder.supports_model("test-model")
        
        assert result is True
        mock_instance.supports_model.assert_called_once_with("test-model")
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_get_embedding_dimension_delegation(self, mock_cpu_embedder, core_embedder):
        """Test embedding dimension retrieval delegation."""
        mock_instance = Mock()
        mock_instance.get_embedding_dimension.return_value = 768
        mock_cpu_embedder.return_value = mock_instance
        
        dimension = core_embedder.get_embedding_dimension("test-model")
        
        assert dimension == 768
        mock_instance.get_embedding_dimension.assert_called_once_with("test-model")
    
    @patch('src.components.embedding.core.processor.CPUEmbedder')
    def test_property_delegation(self, mock_cpu_embedder, core_embedder):
        """Test property delegation to current embedder."""
        mock_instance = Mock()
        mock_instance.embedding_dimension = 384
        mock_instance.max_sequence_length = 512
        mock_instance.supports_batch_processing.return_value = True
        mock_instance.get_optimal_batch_size.return_value = 32
        mock_cpu_embedder.return_value = mock_instance
        
        # Force embedder initialization
        core_embedder._get_embedder()
        
        assert core_embedder.embedding_dimension == 384
        assert core_embedder.max_sequence_length == 512
        assert core_embedder.supports_batch_processing() is True
        assert core_embedder.get_optimal_batch_size() == 32
    
    def test_embedder_switching(self, core_embedder):
        """Test switching between different embedder types."""
        # Start with CPU
        assert core_embedder._embedding_type == 'cpu'
        assert core_embedder._current_embedder is None
        
        # Switch to GPU
        core_embedder.configure({'embedding_type': 'gpu'})
        assert core_embedder._embedding_type == 'gpu'
        assert core_embedder._current_embedder is None  # Reset
        
        # Switch to encoder
        core_embedder.configure({'embedding_type': 'encoder'})
        assert core_embedder._embedding_type == 'encoder'
        assert core_embedder._current_embedder is None  # Reset