"""
Unit tests for GPU Embedder Component
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any

from src.components.embedding.gpu.processor import GPUEmbedder
from src.types.components.contracts import (
    ComponentType,
    EmbeddingInput,
    EmbeddingOutput,
    DocumentChunk,
    ProcessingStatus
)


class TestGPUEmbedder:
    """Test suite for GPU embedder component."""
    
    @pytest.fixture
    def gpu_embedder(self):
        """Create GPU embedder instance for testing."""
        config = {
            'model_name': 'test-gpu-model',
            'batch_size': 64,
            'max_length': 1024,
            'device': 'cuda',
            'use_fp16': True
        }
        return GPUEmbedder(config)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample embedding input."""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="This is the first GPU test chunk.",
                metadata={"source": "test"}
            ),
            DocumentChunk(
                id="chunk_2", 
                content="This is the second GPU test chunk.",
                metadata={"source": "test"}
            ),
            DocumentChunk(
                id="chunk_3",
                content="This is the third GPU test chunk.",
                metadata={"source": "test"}
            )
        ]
        
        return EmbeddingInput(
            chunks=chunks,
            model_name="test-gpu-model",
            batch_size=64,
            metadata={"test": True, "gpu": True}
        )
    
    def test_init(self, gpu_embedder):
        """Test GPU embedder initialization."""
        assert gpu_embedder.name == "gpu"
        assert gpu_embedder.version == "1.0.0"
        assert gpu_embedder.component_type == ComponentType.EMBEDDING
        assert gpu_embedder._model_name == "test-gpu-model"
        assert gpu_embedder._batch_size == 64
        assert gpu_embedder._device == "cuda"
        assert gpu_embedder._use_fp16 is True
    
    def test_configure(self, gpu_embedder):
        """Test configuration update."""
        new_config = {
            'model_name': 'new-gpu-model',
            'batch_size': 128,
            'max_length': 2048,
            'use_fp16': False,
            'gpu_memory_fraction': 0.8
        }
        
        gpu_embedder.configure(new_config)
        
        assert gpu_embedder._model_name == "new-gpu-model"
        assert gpu_embedder._batch_size == 128
        assert gpu_embedder._max_length == 2048
        assert gpu_embedder._use_fp16 is False
        assert gpu_embedder._gpu_memory_fraction == 0.8
    
    def test_validate_config(self, gpu_embedder):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'batch_size': 128,
            'max_length': 4096,
            'device': 'cuda',
            'gpu_memory_fraction': 0.9
        }
        assert gpu_embedder.validate_config(valid_config) is True
        
        # Invalid batch size
        invalid_config = {
            'batch_size': 0
        }
        assert gpu_embedder.validate_config(invalid_config) is False
        
        # Invalid device
        invalid_config = {
            'device': 'invalid_device'
        }
        assert gpu_embedder.validate_config(invalid_config) is False
        
        # Invalid GPU memory fraction
        invalid_config = {
            'gpu_memory_fraction': 1.5
        }
        assert gpu_embedder.validate_config(invalid_config) is False
    
    def test_get_config_schema(self, gpu_embedder):
        """Test config schema generation."""
        schema = gpu_embedder.get_config_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'model_name' in schema['properties']
        assert 'batch_size' in schema['properties']
        assert 'device' in schema['properties']
        assert 'use_fp16' in schema['properties']
        assert 'gpu_memory_fraction' in schema['properties']
    
    @patch('src.components.embedding.gpu.processor.torch.cuda.is_available')
    @patch('src.components.embedding.gpu.processor.ModelEngineFactory')
    def test_health_check_with_gpu(self, mock_factory, mock_cuda_available, gpu_embedder):
        """Test health check with GPU available."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        # Mock successful model initialization
        mock_model = Mock()
        mock_model.embed_texts.return_value = [[0.1] * 768]
        mock_factory.return_value.create_model_engine.return_value = mock_model
        
        # Force model initialization
        gpu_embedder._model_engine = mock_model
        gpu_embedder._model_loaded = True
        
        # Health check should succeed
        assert gpu_embedder.health_check() is True
    
    @patch('src.components.embedding.gpu.processor.torch.cuda.is_available')
    def test_health_check_without_gpu(self, mock_cuda_available, gpu_embedder):
        """Test health check without GPU available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        
        # Health check should fail for GPU embedder without GPU
        assert gpu_embedder.health_check() is False
    
    def test_get_metrics(self, gpu_embedder):
        """Test metrics collection."""
        # Set some test statistics
        gpu_embedder._total_embeddings_created = 1000
        gpu_embedder._total_processing_time = 25.5
        gpu_embedder._total_batches_processed = 50
        gpu_embedder._gpu_memory_peak = 2147483648  # 2GB
        
        metrics = gpu_embedder.get_metrics()
        
        assert metrics['component_name'] == "gpu"
        assert metrics['component_version'] == "1.0.0"
        assert metrics['total_embeddings_created'] == 1000
        assert metrics['total_processing_time'] == 25.5
        assert metrics['total_batches_processed'] == 50
        assert metrics['avg_processing_time'] == 0.0255
        assert metrics['gpu_memory_peak_mb'] == 2048.0
    
    @patch('src.components.embedding.gpu.processor.torch.cuda.is_available')
    @patch('src.components.embedding.gpu.processor.ModelEngineFactory')
    def test_embed_success(self, mock_factory, mock_cuda_available, gpu_embedder, sample_input):
        """Test successful GPU embedding generation."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        # Mock model engine
        mock_model = Mock()
        mock_model.embed_texts.return_value = [
            [0.1] * 768,  # Mock embedding for chunk 1
            [0.2] * 768,  # Mock embedding for chunk 2
            [0.3] * 768   # Mock embedding for chunk 3
        ]
        mock_factory.return_value.create_model_engine.return_value = mock_model
        
        # Force model initialization
        gpu_embedder._model_engine = mock_model
        gpu_embedder._model_loaded = True
        
        # Generate embeddings
        output = gpu_embedder.embed(sample_input)
        
        # Verify output
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 3
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert output.metadata.component_name == "gpu"
        assert len(output.errors) == 0
        
        # Verify embeddings
        assert output.embeddings[0].chunk_id == "chunk_1"
        assert output.embeddings[0].embedding_dimension == 768
        assert output.embeddings[2].chunk_id == "chunk_3"
        
        # Verify GPU-specific stats
        assert 'device' in output.model_info
        assert output.model_info['device'] == 'cuda'
        assert 'use_fp16' in output.model_info
    
    @patch('src.components.embedding.gpu.processor.torch.cuda.is_available')
    def test_embed_cpu_fallback(self, mock_cuda_available, gpu_embedder, sample_input):
        """Test CPU fallback when GPU not available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        
        # Force CPU fallback
        gpu_embedder._device = 'cpu'
        gpu_embedder._model_engine = "cpu_fallback"
        gpu_embedder._model_loaded = True
        
        # Generate embeddings
        output = gpu_embedder.embed(sample_input)
        
        # Verify output
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 3
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Verify CPU fallback
        assert output.model_info['device'] == 'cpu'
        assert 'gpu_available' in output.model_info
        assert output.model_info['gpu_available'] is False
    
    def test_embed_error(self, gpu_embedder, sample_input):
        """Test embedding error handling."""
        # Force error during embedding
        gpu_embedder._model_loaded = False
        gpu_embedder._initialize_model_engine = Mock(side_effect=Exception("GPU init error"))
        
        # Generate embeddings
        output = gpu_embedder.embed(sample_input)
        
        # Verify error handling
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 0
        assert output.metadata.status == ProcessingStatus.ERROR
        assert len(output.errors) > 0
        assert "GPU embedding failed" in output.errors[0]
    
    def test_estimate_tokens(self, gpu_embedder, sample_input):
        """Test GPU token estimation."""
        estimated = gpu_embedder.estimate_tokens(sample_input)
        
        # GPU should handle more tokens efficiently
        total_chars = sum(len(chunk.content) for chunk in sample_input.chunks)
        expected = max(1, total_chars // 3)  # More efficient than CPU
        
        assert estimated == expected
    
    def test_supports_model(self, gpu_embedder):
        """Test GPU model support checking."""
        # GPU-optimized models
        assert gpu_embedder.supports_model("sentence-transformers/all-mpnet-base-v2") is True
        assert gpu_embedder.supports_model("BAAI/bge-large-en-v1.5") is True
        assert gpu_embedder.supports_model("intfloat/e5-large-v2") is True
        
        # Should support general models too
        assert gpu_embedder.supports_model("all-MiniLM-L6-v2") is True
    
    def test_get_embedding_dimension(self, gpu_embedder):
        """Test GPU embedding dimension retrieval."""
        assert gpu_embedder.get_embedding_dimension("all-mpnet-base-v2") == 768
        assert gpu_embedder.get_embedding_dimension("BAAI/bge-large-en-v1.5") == 1024
        assert gpu_embedder.get_embedding_dimension("unknown-model") == 768  # Default
    
    def test_embedding_properties(self, gpu_embedder):
        """Test GPU embedder properties."""
        assert gpu_embedder.embedding_dimension == 768  # Based on test-gpu-model
        assert gpu_embedder.max_sequence_length == 1024
        assert gpu_embedder.supports_batch_processing() is True
        assert gpu_embedder.get_optimal_batch_size() == 64
    
    @patch('src.components.embedding.gpu.processor.torch.cuda.memory_allocated')
    @patch('src.components.embedding.gpu.processor.torch.cuda.empty_cache')
    def test_gpu_memory_management(self, mock_empty_cache, mock_memory_allocated, gpu_embedder):
        """Test GPU memory management."""
        # Mock memory usage
        mock_memory_allocated.return_value = 1073741824  # 1GB
        
        # Test memory cleanup
        gpu_embedder._cleanup_gpu_memory()
        
        # Verify cache was cleared
        mock_empty_cache.assert_called_once()
    
    def test_batch_processing(self, gpu_embedder):
        """Test GPU batch processing capabilities."""
        # Create large input
        chunks = [
            DocumentChunk(
                id=f"chunk_{i}",
                content=f"Test content number {i} for GPU batch processing.",
                metadata={"index": i}
            )
            for i in range(200)  # Large batch
        ]
        
        large_input = EmbeddingInput(
            chunks=chunks,
            model_name="test-gpu-model",
            batch_size=64,
            metadata={"test": True}
        )
        
        # Estimate should handle large batches efficiently
        estimated = gpu_embedder.estimate_tokens(large_input)
        assert estimated > 0
        
        # Verify batch size optimization
        optimal_batch = gpu_embedder.get_optimal_batch_size()
        assert optimal_batch == 64  # GPU can handle larger batches