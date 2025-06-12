"""
Test configuration and utilities for Model Engine tests

Provides shared test fixtures, mocks, and utilities for model engine testing.
"""

import pytest
from unittest.mock import Mock, MagicMock
import tempfile
import os
import yaml
from typing import Dict, Any, List


class MockModelEngine:
    """Mock model engine for testing purposes."""
    
    def __init__(self, name: str = "mock", **kwargs):
        self.name = name
        self.version = "1.0.0"
        self._config = kwargs
        self._initialized = False
        
    def is_ready(self) -> bool:
        return self._initialized
        
    def generate(self, request):
        """Mock generation method."""
        mock_response = Mock()
        mock_response.output_text = f"Generated text for {request.input_text}"
        mock_response.tokens = [1, 2, 3, 4, 5]
        mock_response.metadata = {"model": self.name}
        return mock_response
        
    def embed(self, texts: List[str]):
        """Mock embedding method."""
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3] for _ in texts]
        mock_response.metadata = {"model": self.name, "embedding_dim": 3}
        return mock_response
        
    def get_stats(self) -> Dict[str, Any]:
        """Mock stats method."""
        return {
            "requests_processed": 100,
            "average_latency": 0.5,
            "error_rate": 0.01,
            "uptime": 3600
        }


class MockPipeline:
    """Mock Haystack pipeline for testing."""
    
    def __init__(self, config_path: str = None, **kwargs):
        self.config_path = config_path
        self._config = kwargs
        self._nodes = ["retriever", "reader"]
        
    def run(self, query: str = None, **kwargs):
        """Mock pipeline run method."""
        if "question" in str(query).lower():
            return {
                "answers": [
                    {
                        "answer": "Mock answer to the question",
                        "score": 0.95,
                        "context": "Mock context for the answer",
                        "document_id": "mock_doc_1"
                    }
                ],
                "documents": [
                    {
                        "content": "Mock document content",
                        "id": "mock_doc_1",
                        "score": 0.9
                    }
                ]
            }
        else:
            return {
                "documents": [
                    {
                        "content": "Mock retrieved document",
                        "id": "mock_doc_1",
                        "score": 0.85
                    }
                ]
            }
    
    def get_nodes(self) -> List[str]:
        """Get pipeline nodes."""
        return self._nodes
        
    def get_stats(self) -> Dict[str, Any]:
        """Mock pipeline stats."""
        return {
            "queries_processed": 50,
            "avg_retrieval_time": 0.2,
            "avg_reading_time": 0.8,
            "document_count": 1000
        }


@pytest.fixture
def mock_model_engine():
    """Provide a mock model engine for testing."""
    return MockModelEngine()


@pytest.fixture
def mock_pipeline():
    """Provide a mock Haystack pipeline for testing."""
    return MockPipeline()


@pytest.fixture
def temp_config_dir():
    """Create temporary directory with test config files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test pipeline config
    pipeline_config = {
        "version": "1.0",
        "components": [
            {
                "name": "retriever",
                "type": "BM25Retriever",
                "params": {
                    "document_store": "document_store",
                    "top_k": 10
                }
            },
            {
                "name": "reader",
                "type": "FARMReader", 
                "params": {
                    "model_name_or_path": "deepset/roberta-base-squad2",
                    "top_k": 3
                }
            }
        ],
        "pipelines": [
            {
                "name": "query_pipeline",
                "nodes": [
                    {
                        "name": "retriever",
                        "inputs": ["Query"]
                    },
                    {
                        "name": "reader",
                        "inputs": ["retriever"]
                    }
                ]
            }
        ]
    }
    
    pipeline_config_path = os.path.join(temp_dir, "test_pipeline.yaml")
    with open(pipeline_config_path, 'w') as f:
        yaml.dump(pipeline_config, f)
    
    # Create test model config
    model_config = {
        "models": {
            "embedding": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "generation": {
                "name": "microsoft/DialoGPT-medium",
                "device": "cuda"
            }
        }
    }
    
    model_config_path = os.path.join(temp_dir, "test_models.yaml")
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)
    
    yield {
        "temp_dir": temp_dir,
        "pipeline_config": pipeline_config_path,
        "model_config": model_config_path
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_configs():
    """Provide sample configurations for different engines."""
    return {
        "core": {
            "engine_type": "vllm",
            "model_name": "BAAI/bge-large-en-v1.5",
            "server_config": {
                "host": "localhost",
                "port": 8000,
                "gpu_memory_utilization": 0.8,
                "max_model_len": 4096
            },
            "inference_config": {
                "max_tokens": 1024,
                "temperature": 0.7,
                "batch_size": 32
            }
        },
        "vllm": {
            "server_url": "http://localhost:8000",
            "device": "cuda",
            "max_retries": 3,
            "timeout": 60,
            "model_config": {
                "model_name": "microsoft/DialoGPT-medium",
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 2048,
                "enable_chunked_prefill": True
            }
        },
        "haystack": {
            "pipeline_config": "/tmp/test_pipeline.yaml",
            "server_url": "http://localhost:8080",
            "retriever_config": {
                "top_k": 10,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "reader_config": {
                "model": "deepset/roberta-base-squad2",
                "max_seq_len": 384,
                "top_k": 3
            },
            "document_store_config": {
                "type": "memory",
                "embedding_dim": 384
            }
        }
    }


@pytest.fixture
def sample_inputs():
    """Provide sample inputs for testing."""
    from src.types.components.contracts import ModelEngineInput
    
    return {
        "text_generation": ModelEngineInput(
            requests=[
                {
                    "request_id": "gen-1",
                    "request_type": "generate",
                    "input_text": "Complete this sentence: The future of AI is",
                    "parameters": {
                        "max_tokens": 100,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            ],
            model_config={"model_name": "microsoft/DialoGPT-medium"},
            batch_config={"batch_size": 32},
            metadata={"session_id": "test-session"}
        ),
        "embedding": ModelEngineInput(
            requests=[
                {
                    "request_id": "embed-1",
                    "request_type": "embed",
                    "input_text": "This is text to embed into a vector representation",
                    "parameters": {
                        "normalize": True,
                        "truncate": True
                    }
                }
            ],
            model_config={"model_name": "BAAI/bge-large-en-v1.5"},
            batch_config={"batch_size": 64},
            metadata={}
        ),
        "question_answering": ModelEngineInput(
            requests=[
                {
                    "request_id": "qa-1",
                    "request_type": "question_answering",
                    "input_text": "What is the capital of France?",
                    "parameters": {
                        "retriever": {"top_k": 10},
                        "reader": {"top_k": 3}
                    }
                }
            ],
            model_config={
                "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reader_model": "deepset/roberta-base-squad2"
            },
            batch_config={},
            metadata={}
        ),
        "batch": ModelEngineInput(
            requests=[
                {
                    "request_id": f"batch-{i}",
                    "request_type": "generate",
                    "input_text": f"Generate text for prompt {i}",
                    "parameters": {"max_tokens": 50}
                }
                for i in range(5)
            ],
            model_config={},
            batch_config={"batch_size": 5, "enable_batching": True},
            metadata={"experiment": "batch_test"}
        )
    }


def create_mock_response(request_id: str, output_text: str = None, error: str = None):
    """Create a mock ModelInferenceResult."""
    from src.types.components.contracts import ModelInferenceResult
    
    return ModelInferenceResult(
        request_id=request_id,
        response_data={
            "output_text": output_text or f"Mock response for {request_id}",
            "tokens": [1, 2, 3, 4, 5],
            "metadata": {"mock": True}
        },
        processing_time=0.5,
        error=error
    )


def assert_valid_output(output, expected_request_count: int = 1):
    """Assert that a ModelEngineOutput is valid."""
    from src.types.components.contracts import ModelEngineOutput, ProcessingStatus
    
    assert isinstance(output, ModelEngineOutput)
    assert len(output.results) == expected_request_count
    assert output.metadata is not None
    assert output.engine_stats is not None
    assert isinstance(output.errors, list)
    
    # Check engine stats
    assert "request_count" in output.engine_stats
    assert "success_count" in output.engine_stats
    assert "error_count" in output.engine_stats
    assert "processing_time" in output.engine_stats
    
    # Validate individual results
    for result in output.results:
        assert hasattr(result, 'request_id')
        assert hasattr(result, 'response_data')
        assert hasattr(result, 'processing_time')
        assert result.processing_time >= 0


# Test markers for categorizing tests
pytest_markers = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
    "slow": pytest.mark.slow,
    "gpu": pytest.mark.gpu,
    "mock": pytest.mark.mock,
    "real": pytest.mark.real,
    "core": pytest.mark.core,
    "vllm": pytest.mark.vllm,
    "haystack": pytest.mark.haystack
}