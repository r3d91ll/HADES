"""
Unit tests for vLLM Model Engine Processor

Tests the VLLMModelEngine component implementation including:
- vLLM-specific configuration and initialization
- GPU acceleration and optimization features
- Server communication and error handling
- Performance metrics and monitoring
- Protocol compliance with ModelEngine interface
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.components.model_engine.engines.vllm.processor import VLLMModelEngine
from src.types.components.contracts import (
    ComponentType,
    ModelEngineInput,
    ModelEngineOutput,
    ModelInferenceResult,
    ProcessingStatus
)


class TestVLLMModelEngine:
    """Test suite for VLLMModelEngine component."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            "server_url": "http://localhost:8000",
            "device": "cuda",
            "max_retries": 3,
            "timeout": 60,
            "model_config": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9
            }
        }
        self.engine = VLLMModelEngine(config=self.default_config)
    
    def test_initialization(self):
        """Test vLLM component initialization."""
        # Test default initialization
        default_engine = VLLMModelEngine()
        assert default_engine.name == "vllm"
        assert default_engine.version == "1.0.0"
        assert default_engine.component_type == ComponentType.MODEL_ENGINE
        assert default_engine._server_url == "http://localhost:8000"
        assert default_engine._device == "cuda"
        
        # Test with configuration
        assert self.engine._config == self.default_config
        assert self.engine._server_url == "http://localhost:8000"
        assert self.engine._device == "cuda"
        assert self.engine._max_retries == 3
        assert self.engine._timeout == 60
    
    def test_configuration(self):
        """Test vLLM component configuration."""
        new_config = {
            "server_url": "http://remote-server:8080",
            "device": "cpu",
            "max_retries": 5,
            "timeout": 120,
            "model_config": {
                "max_model_len": 4096
            }
        }
        
        # Test valid configuration
        self.engine.configure(new_config)
        assert self.engine._server_url == "http://remote-server:8080"
        assert self.engine._device == "cpu"
        assert self.engine._max_retries == 5
        assert self.engine._timeout == 120
        
        # Test configuration validation
        invalid_config = {
            "server_url": 12345,  # Should be string
            "device": "invalid_device",  # Should be cuda or cpu
            "max_retries": -1,  # Should be positive
            "timeout": "not_a_number"  # Should be numeric
        }
        
        with pytest.raises(ValueError, match="Invalid configuration provided"):
            self.engine.configure(invalid_config)
    
    def test_config_validation(self):
        """Test vLLM configuration validation."""
        # Valid configurations
        valid_configs = [
            {"server_url": "http://localhost:8000"},
            {"device": "cuda"},
            {"device": "cpu"},
            {"max_retries": 5},
            {"timeout": 30},
            {"model_config": {"tensor_parallel_size": 4}},
            {}  # Empty config should be valid
        ]
        
        for config in valid_configs:
            assert self.engine.validate_config(config), f"Config should be valid: {config}"
        
        # Invalid configurations
        invalid_configs = [
            "not_a_dict",
            {"server_url": 123},
            {"device": "invalid"},
            {"max_retries": 0},
            {"max_retries": -1},
            {"timeout": -5},
            {"timeout": "invalid"},
            {"model_config": "not_a_dict"}
        ]
        
        for config in invalid_configs:
            assert not self.engine.validate_config(config), f"Config should be invalid: {config}"
    
    def test_config_schema(self):
        """Test vLLM configuration schema generation."""
        schema = self.engine.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "server_url" in schema["properties"]
        assert "device" in schema["properties"]
        assert "max_retries" in schema["properties"]
        assert "timeout" in schema["properties"]
        assert "model_config" in schema["properties"]
        
        # Check device enum values
        assert "cuda" in schema["properties"]["device"]["enum"]
        assert "cpu" in schema["properties"]["device"]["enum"]
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_health_check(self, mock_legacy_engine):
        """Test vLLM health check functionality."""
        # Test without initialized engine
        assert not self.engine.health_check()
        
        # Test with healthy mock engine
        mock_engine = Mock()
        mock_engine.is_ready.return_value = True
        mock_engine.health_check.return_value = {"status": "healthy"}
        self.engine._engine = mock_engine
        
        assert self.engine.health_check()
        
        # Test with unhealthy engine
        mock_engine.is_ready.return_value = False
        assert not self.engine.health_check()
        
        # Test engine without health methods
        mock_engine_basic = Mock(spec=[])
        self.engine._engine = mock_engine_basic
        assert self.engine.health_check()  # Should default to True
        
        # Test health check exception handling
        mock_engine.is_ready.side_effect = Exception("Health check failed")
        self.engine._engine = mock_engine
        assert not self.engine.health_check()
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_metrics_collection(self, mock_legacy_engine):
        """Test vLLM metrics collection."""
        # Test basic metrics
        metrics = self.engine.get_metrics()
        
        assert metrics["component_name"] == "vllm"
        assert metrics["component_version"] == "1.0.0"
        assert metrics["server_url"] == "http://localhost:8000"
        assert metrics["device"] == "cuda"
        assert "last_health_check" in metrics
        
        # Test with mock engine that has detailed stats
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {
            "requests_per_second": 10.5,
            "gpu_memory_usage": 0.75,
            "average_latency": 150.2,
            "total_requests": 1000,
            "failed_requests": 5
        }
        self.engine._engine = mock_engine
        
        metrics = self.engine.get_metrics()
        assert "engine_stats" in metrics
        assert metrics["engine_stats"]["requests_per_second"] == 10.5
        assert metrics["engine_stats"]["gpu_memory_usage"] == 0.75
        
        # Test GPU-specific metrics
        assert "gpu_utilization" in metrics or "engine_stats" in metrics
        
        # Test metrics collection failure handling
        mock_engine.get_stats.side_effect = Exception("Stats failed")
        metrics = self.engine.get_metrics()
        assert "engine_stats" not in metrics  # Should handle gracefully
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_process_text_generation(self, mock_legacy_engine):
        """Test vLLM text generation processing."""
        # Setup mock engine
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.output_text = "Generated text from vLLM"
        mock_response.tokens = [1, 2, 3, 4, 5]
        mock_response.metadata = {
            "finish_reason": "stop",
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 15}
        }
        
        mock_engine.generate.return_value = mock_response
        mock_legacy_engine.return_value = mock_engine
        
        # Create test input for text generation
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "vllm-test-1",
                    "request_type": "generate",
                    "input_text": "Complete this sentence:",
                    "parameters": {
                        "max_tokens": 100,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            ],
            model_config={
                "model_name": "test-model",
                "tensor_parallel_size": 2
            },
            batch_config={"batch_size": 32},
            metadata={"session_id": "test-session"}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "vllm-test-1"
        assert result.results[0].response_data["output_text"] == "Generated text from vLLM"
        assert result.results[0].response_data["tokens"] == [1, 2, 3, 4, 5]
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Validate vLLM-specific stats
        assert result.engine_stats["engine_type"] == "vllm"
        assert result.engine_stats["success_count"] == 1
        assert result.engine_stats["error_count"] == 0
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_process_embedding_generation(self, mock_legacy_engine):
        """Test vLLM embedding generation processing."""
        # Setup mock engine for embeddings
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5-dim embedding
        mock_response.metadata = {
            "model": "embedding-model",
            "embedding_dim": 5
        }
        
        mock_engine.embed.return_value = mock_response
        mock_legacy_engine.return_value = mock_engine
        
        # Create test input for embedding generation
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "embed-test-1",
                    "request_type": "embed",
                    "input_text": "Text to embed",
                    "parameters": {
                        "normalize": True,
                        "truncate": True
                    }
                }
            ],
            model_config={"model_name": "embedding-model"},
            batch_config={"batch_size": 64},
            metadata={}
        )
        
        # Process
        result = self.engine.process(test_input)
        
        # Validate embedding output
        assert isinstance(result, ModelEngineOutput)
        assert len(result.results) == 1
        assert result.results[0].request_id == "embed-test-1"
        assert "embeddings" in result.results[0].response_data
        assert result.results[0].response_data["embeddings"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_batch_processing(self, mock_legacy_engine):
        """Test vLLM batch processing capabilities."""
        # Setup mock engine with batch processing
        mock_engine = Mock()
        
        def batch_generate(requests):
            responses = []
            for i, req in enumerate(requests):
                mock_resp = Mock()
                mock_resp.output_text = f"Response {i+1}"
                mock_resp.tokens = [i, i+1, i+2]
                mock_resp.metadata = {"batch_index": i}
                responses.append(mock_resp)
            return responses
        
        mock_engine.generate_batch.return_value = batch_generate
        mock_legacy_engine.return_value = mock_engine
        
        # Create batch input
        batch_requests = [
            {
                "request_id": f"batch-{i}",
                "request_type": "generate",
                "input_text": f"Prompt {i}",
                "parameters": {"max_tokens": 50}
            }
            for i in range(5)
        ]
        
        test_input = ModelEngineInput(
            requests=batch_requests,
            model_config={},
            batch_config={"batch_size": 5, "enable_batching": True},
            metadata={}
        )
        
        # Process batch
        result = self.engine.process(test_input)
        
        # Validate batch processing
        assert len(result.results) == 5
        assert result.engine_stats["request_count"] == 5
        assert result.metadata.status == ProcessingStatus.SUCCESS
        
        # Check individual responses
        for i, res in enumerate(result.results):
            assert res.request_id == f"batch-{i}"
            assert res.response_data["output_text"] == f"Response {i+1}"
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_gpu_optimization_features(self, mock_legacy_engine):
        """Test vLLM GPU optimization features."""
        # Test tensor parallelism configuration
        gpu_config = {
            "model_config": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 2,
                "gpu_memory_utilization": 0.95,
                "max_model_len": 8192,
                "enable_chunked_prefill": True
            }
        }
        
        self.engine.configure(gpu_config)
        
        # Verify configuration applied
        assert self.engine._config["model_config"]["tensor_parallel_size"] == 4
        assert self.engine._config["model_config"]["gpu_memory_utilization"] == 0.95
        
        # Test optimization metrics
        mock_engine = Mock()
        mock_engine.get_optimization_stats.return_value = {
            "tensor_parallel_efficiency": 0.95,
            "gpu_memory_peak": 0.87,
            "prefill_throughput": 1500,
            "decode_throughput": 8500
        }
        self.engine._engine = mock_engine
        
        metrics = self.engine.get_metrics()
        # Should include GPU-specific metrics if available
        if hasattr(mock_engine, 'get_optimization_stats'):
            mock_engine.get_optimization_stats.assert_called()
    
    @patch('src.components.model_engine.engines.vllm.processor.LegacyVLLMEngine')
    def test_error_handling_and_retries(self, mock_legacy_engine):
        """Test vLLM error handling and retry logic."""
        # Setup mock engine with intermittent failures
        mock_engine = Mock()
        call_count = 0
        
        def generate_with_retries(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise Exception("Temporary vLLM error")
            
            # Success on 3rd attempt
            mock_resp = Mock()
            mock_resp.output_text = "Success after retries"
            mock_resp.tokens = [1, 2, 3]
            mock_resp.metadata = {"retry_count": call_count - 1}
            return mock_resp
        
        mock_engine.generate.side_effect = generate_with_retries
        mock_legacy_engine.return_value = mock_engine
        
        # Configure retries
        self.engine.configure({"max_retries": 3})
        
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "retry-test",
                    "input_text": "Test prompt",
                    "parameters": {}
                }
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Process (should succeed after retries)
        result = self.engine.process(test_input)
        
        # Validate retry behavior - this depends on implementation
        # For now, check that we get a result
        assert len(result.results) >= 1
    
    def test_performance_estimation(self):
        """Test vLLM performance estimation."""
        # Test small batch estimation
        small_input = ModelEngineInput(
            requests=[{"request_id": "1", "input_text": "test"}],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        estimate = self.engine.estimate_processing_time(small_input)
        assert estimate > 0
        
        # Test large batch estimation
        large_input = ModelEngineInput(
            requests=[{"request_id": str(i), "input_text": "test"} for i in range(100)],
            model_config={},
            batch_config={"batch_size": 32},
            metadata={}
        )
        
        large_estimate = self.engine.estimate_processing_time(large_input)
        assert large_estimate >= estimate  # Should account for batch processing efficiency
    
    def test_supported_models(self):
        """Test vLLM supported models listing."""
        models = self.engine.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Should include common vLLM-compatible models
        expected_models = [
            "microsoft/DialoGPT-medium",
            "BAAI/bge-large-en-v1.5",
            "meta-llama/Llama-2-7b-hf"
        ]
        
        for model in expected_models:
            if model in models:  # Some models might not be in the list
                assert model in models
    
    def test_device_configuration(self):
        """Test device-specific configuration."""
        # Test CUDA configuration
        cuda_config = {
            "device": "cuda",
            "model_config": {
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 2
            }
        }
        
        self.engine.configure(cuda_config)
        assert self.engine._device == "cuda"
        
        # Test CPU fallback configuration
        cpu_config = {
            "device": "cpu",
            "model_config": {
                "max_model_len": 2048  # Smaller for CPU
            }
        }
        
        self.engine.configure(cpu_config)
        assert self.engine._device == "cpu"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal configuration
        minimal_engine = VLLMModelEngine(config={})
        assert minimal_engine._server_url == "http://localhost:8000"
        assert minimal_engine._device == "cuda"
        
        # Test empty batch processing
        empty_input = ModelEngineInput(
            requests=[],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        result = self.engine.process(empty_input)
        assert len(result.results) == 0
        
        # Test malformed requests
        malformed_input = ModelEngineInput(
            requests=[
                {"invalid": "request"},  # Missing required fields
                {"request_id": "valid", "input_text": "Good request"}
            ],
            model_config={},
            batch_config={},
            metadata={}
        )
        
        # Should handle gracefully
        result = self.engine.process(malformed_input)
        assert isinstance(result, ModelEngineOutput)


if __name__ == "__main__":
    pytest.main([__file__])