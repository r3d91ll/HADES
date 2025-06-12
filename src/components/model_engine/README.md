# Model Engine Component

The model engine component provides comprehensive model serving infrastructure for the HADES-PathRAG system. It supports multiple model serving backends through a hierarchical component architecture, enabling flexible deployment of language models, embedding models, and other AI services with unified interfaces and robust management.

## 📋 Overview

This module implements:

- **Multi-backend model serving** with support for vLLM, Haystack, and other engines
- **Adapter pattern** for seamless integration of different model serving platforms
- **Session management** for efficient model lifecycle and resource utilization
- **Factory pattern** for dynamic engine selection and initialization
- **Server management** for distributed model serving and load balancing
- **Performance optimization** through caching, batching, and resource pooling

## 🏗️ Architecture

### Model Serving Pipeline

```text
Model Requests → Engine Factory → Adapter Selection → Model Server → Response Processing
      ↓              ↓               ↓                ↓               ↓
API Calls → Engine Registry → Backend Adapter → Inference Service → Formatted Output
```

### Multi-Engine Design

```text
Client Request → Load Balancer → Engine Pool → Model Instance → Inference Result
      ↓             ↓             ↓            ↓                ↓
Query/Generate → Server Manager → vLLM/Haystack → GPU/CPU Compute → Processed Response
```

## 📁 Module Contents

### Core Components

- **`factory.py`** - Engine factory for dynamic engine creation and management
- **`server_manager.py`** - Server lifecycle management and health monitoring
- **`base.py`** - Base interfaces and abstract classes for engine implementations
- **`vllm_session.py`** - vLLM-specific session management and optimization
- **`__init__.py`** - Module exports and convenience imports

### Engine Implementations

**`core/`** - Core model engine implementation:

- **`processor.py`** - Core model engine processor implementing the ModelEngine protocol

**`engines/`** - Specific engine implementations:

**`engines/vllm/`** - vLLM integration:

- **`processor.py`** - vLLM model engine processor implementing the ModelEngine protocol
- **`vllm_engine.py`** - Main vLLM engine implementation for text generation and embeddings

**`engines/haystack/`** - Haystack integration:

- **`processor.py`** - Haystack model engine processor implementing the ModelEngine protocol
- **`haystack_engine.md`** - Haystack engine documentation
- **`runtime/`** - Haystack runtime components:
  - **`embedding.py`** - Haystack embedding service
  - **`server.py`** - Haystack server implementation
  - **`proto/`** - Protocol buffer definitions

### Adapter Layer

**`adapters/`** - Engine adapters for unified interfaces:

- **`vllm_adapter.py`** - vLLM adapter for text generation
- **`vllm_engine_adapter.py`** - Enhanced vLLM adapter with advanced features

### Documentation

- **`model_engine_readme.md`** - Legacy documentation (superseded by this README)

## 🚀 Key Features

### Multi-Backend Support

**Supported model engines**:

- **vLLM** - High-performance inference for large language models
- **Haystack** - Production-ready NLP pipeline framework
- **Custom Engines** - Extensible architecture for new backends

**Dynamic engine selection**:

```python
from src.model_engine.factory import ModelEngineFactory

# Automatic engine selection based on model type
factory = ModelEngineFactory()

# Text generation engine
llm_engine = factory.create_engine(
    engine_type="vllm",
    model_name="microsoft/DialoGPT-medium",
    task="text_generation"
)

# Embedding engine  
embedding_engine = factory.create_engine(
    engine_type="vllm",
    model_name="answerdotai/ModernBERT-base", 
    task="embedding"
)
```

### High-Performance Inference

**vLLM integration for optimal performance**:

```python
from src.model_engine.engines.vllm.vllm_engine import VLLMEngine

# Initialize high-performance vLLM engine
engine = VLLMEngine(
    model_name="microsoft/DialoGPT-medium",
    tensor_parallel_size=4,     # Multi-GPU parallelism
    max_model_len=2048,         # Context length
    gpu_memory_utilization=0.9, # GPU memory usage
    enable_chunked_prefill=True # Optimization for long sequences
)

# High-throughput text generation
responses = engine.generate(
    prompts=["Explain quantum computing", "What is machine learning?"],
    sampling_params={
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9
    }
)

# Efficient embedding generation
embeddings = engine.embed(
    texts=["Document chunk 1", "Document chunk 2"],
    batch_size=32
)
```

### Session Management

**Efficient resource management**:

```python
from src.model_engine.vllm_session import VLLMSessionManager

# Initialize session manager
session_manager = VLLMSessionManager(
    max_sessions=10,
    session_timeout=3600,  # 1 hour
    cleanup_interval=300   # 5 minutes
)

# Create managed session
with session_manager.create_session("text_generation") as session:
    # Session automatically manages resources
    results = session.generate(prompts, parameters)
    
    # Session cleanup handled automatically

# Batch processing with session pooling
async def process_large_batch(prompts):
    sessions = await session_manager.get_session_pool(size=4)
    
    # Distribute work across sessions
    tasks = []
    for i, session in enumerate(sessions):
        batch_prompts = prompts[i::len(sessions)]
        task = session.generate_async(batch_prompts)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return flatten_results(results)
```

### Server Management

**Distributed server orchestration**:

```python
from src.model_engine.server_manager import ModelServerManager

# Initialize server manager
server_manager = ModelServerManager(
    config_file="./config/model_servers.yaml"
)

# Start model servers
server_manager.start_servers([
    {
        "name": "llm_server_1",
        "engine": "vllm", 
        "model": "microsoft/DialoGPT-medium",
        "port": 8000,
        "gpu_ids": [0, 1]
    },
    {
        "name": "embedding_server_1", 
        "engine": "vllm",
        "model": "answerdotai/ModernBERT-base",
        "port": 8001,
        "gpu_ids": [2, 3]
    }
])

# Health monitoring
health_status = server_manager.check_server_health()
for server, status in health_status.items():
    print(f"{server}: {status['status']} (Load: {status['load']})")

# Load balancing
optimal_server = server_manager.select_optimal_server(
    task="text_generation",
    load_balancing="round_robin"
)
```

## 🔧 Configuration and Setup

### Model Engine Configuration

Configure model engines in `engine_config.yaml`:

```yaml
model_engines:
  # vLLM configuration
  vllm:
    # Server settings
    host: "localhost"
    port_range: [8000, 8010]
    max_servers: 4
    
    # Model settings
    default_models:
      text_generation: "microsoft/DialoGPT-medium"
      embedding: "answerdotai/ModernBERT-base"
    
    # Performance settings
    tensor_parallel_size: 2
    pipeline_parallel_size: 1
    max_model_len: 2048
    gpu_memory_utilization: 0.9
    
    # Optimization
    enable_chunked_prefill: true
    max_num_seqs: 256
    max_num_batched_tokens: 4096
    
  # Haystack configuration  
  haystack:
    # Pipeline settings
    pipeline_config: "./config/haystack_pipeline.yaml"
    
    # Server settings
    host: "localhost"
    port: 8080
    workers: 4
    
    # Model settings
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    qa_model: "deepset/roberta-base-squad2"

# Global settings
server_management:
  health_check_interval: 30
  max_retries: 3
  timeout_seconds: 60
  
  # Load balancing
  load_balancing_strategy: "least_connections"  # round_robin, least_connections, weighted
  
  # Resource monitoring
  memory_threshold: 0.85
  gpu_memory_threshold: 0.9
  cpu_threshold: 0.8
```

### Environment Variables

```bash
# vLLM configuration
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8000"
export VLLM_TENSOR_PARALLEL_SIZE="4"
export VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Model paths
export HADES_MODEL_CACHE="/path/to/model/cache"
export HUGGINGFACE_HUB_CACHE="/path/to/hf/cache"

# Server management
export HADES_MAX_MODEL_SERVERS="8"
export HADES_SERVER_TIMEOUT="120"
export HADES_HEALTH_CHECK_INTERVAL="30"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG="INFO"
```

## 🔌 Engine Adapters

### vLLM Adapter for Text Generation

```python
from src.model_engine.adapters.vllm_adapter import VLLMAdapter

# Initialize vLLM adapter
adapter = VLLMAdapter(
    model_name="microsoft/DialoGPT-medium",
    server_url="http://localhost:8000",
    api_key="your-api-key"
)

# Text generation
responses = adapter.generate(
    prompts=[
        "Explain the concept of retrieval-augmented generation",
        "What are the benefits of graph neural networks?"
    ],
    generation_config={
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["\n\n"]
    }
)

# Streaming generation
async for token in adapter.generate_stream("Tell me about AI"):
    print(token, end="", flush=True)
```

### vLLM Engine Adapter (Enhanced)

```python
from src.model_engine.adapters.vllm_engine_adapter import VLLMEngineAdapter

# Enhanced adapter with advanced features
adapter = VLLMEngineAdapter(
    engine_config={
        "model": "microsoft/DialoGPT-medium",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9
    },
    optimization_config={
        "enable_prefix_caching": True,
        "chunked_prefill": True,
        "max_num_seqs": 256
    }
)

# Batch processing with optimization
batch_results = adapter.process_batch(
    requests=[
        {"prompt": "Question 1", "max_tokens": 100},
        {"prompt": "Question 2", "max_tokens": 150}, 
        {"prompt": "Question 3", "max_tokens": 200}
    ],
    batch_size=32,
    enable_optimization=True
)

# Advanced features
adapter.enable_speculative_decoding("small_model_name")
adapter.configure_quantization(method="awq", bits=4)
```

### Haystack Integration

```python
from src.model_engine.engines.haystack.runtime.server import HaystackServer

# Initialize Haystack server
server = HaystackServer(
    pipeline_config="./config/haystack_pipeline.yaml",
    host="localhost",
    port=8080
)

# Start server
server.start()

# Query processing
query_result = server.query(
    query="What is the main contribution of the PathRAG paper?",
    params={
        "retriever": {"top_k": 10},
        "reader": {"top_k": 3}
    }
)

print(f"Answer: {query_result['answer']}")
print(f"Sources: {[doc['title'] for doc in query_result['documents']]}")
```

## 🚀 Advanced Features

### Multi-GPU Model Serving

```python
from src.model_engine.engines.vllm.vllm_engine import VLLMEngine

# Multi-GPU configuration
engine = VLLMEngine(
    model_name="microsoft/DialoGPT-large",
    tensor_parallel_size=8,      # Use 8 GPUs
    pipeline_parallel_size=2,    # 2-stage pipeline
    distributed_executor_backend="ray",  # Ray for distributed computing
    
    # Advanced GPU optimization
    gpu_memory_utilization=0.95,
    swap_space=16,               # 16GB swap space
    max_num_seqs=512,           # High throughput
    max_num_batched_tokens=8192 # Large batch processing
)

# Efficient batch processing
large_batch_prompts = ["Prompt " + str(i) for i in range(1000)]
results = engine.generate_batch(
    prompts=large_batch_prompts,
    batch_size=128,
    use_tqdm=True
)
```

### Model Quantization and Optimization

```python
# AWQ quantization for memory efficiency
quantized_engine = VLLMEngine(
    model_name="microsoft/DialoGPT-medium",
    quantization="awq",
    max_model_len=4096,
    gpu_memory_utilization=0.8  # Reduced due to quantization
)

# GPTQ quantization alternative
gptq_engine = VLLMEngine(
    model_name="microsoft/DialoGPT-medium", 
    quantization="gptq",
    quantization_param_path="./gptq_params.json"
)

# Speculative decoding for faster generation
engine.enable_speculative_decoding(
    speculative_model="microsoft/DialoGPT-small",
    num_speculative_tokens=5
)
```

### Custom Engine Development

```python
from src.model_engine.base import BaseModelEngine
from typing import List, Dict, Any

class CustomModelEngine(BaseModelEngine):
    """Custom model engine implementation."""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        """Load custom model implementation."""
        # Implement custom model loading
        pass
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using custom model."""
        results = []
        for prompt in prompts:
            # Custom generation logic
            result = self.model.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using custom model."""
        embeddings = []
        for text in texts:
            # Custom embedding logic
            embedding = self.model.encode(text, **kwargs)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def health_check(self) -> Dict[str, Any]:
        """Check engine health."""
        return {
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "memory_usage": self.get_memory_usage(),
            "gpu_utilization": self.get_gpu_utilization()
        }

# Register custom engine
from src.model_engine.factory import register_engine
register_engine("custom", CustomModelEngine)
```

## 🔍 Monitoring and Performance

### Server Health Monitoring

```python
from src.model_engine.server_manager import ModelServerManager

class AdvancedServerMonitor:
    def __init__(self, server_manager: ModelServerManager):
        self.server_manager = server_manager
        self.metrics_history = []
    
    def collect_metrics(self):
        """Collect comprehensive server metrics."""
        metrics = {
            "timestamp": datetime.now(),
            "servers": {}
        }
        
        for server_name in self.server_manager.list_servers():
            server_metrics = self.server_manager.get_server_metrics(server_name)
            metrics["servers"][server_name] = {
                "status": server_metrics["status"],
                "requests_per_second": server_metrics["rps"],
                "latency_p95": server_metrics["latency_p95"],
                "gpu_memory_usage": server_metrics["gpu_memory"],
                "queue_size": server_metrics["queue_size"],
                "error_rate": server_metrics["error_rate"]
            }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def detect_performance_issues(self):
        """Detect and alert on performance issues."""
        if len(self.metrics_history) < 2:
            return []
        
        issues = []
        latest = self.metrics_history[-1]
        
        for server_name, metrics in latest["servers"].items():
            # High latency check
            if metrics["latency_p95"] > 5000:  # 5 seconds
                issues.append(f"{server_name}: High latency ({metrics['latency_p95']}ms)")
            
            # High error rate check
            if metrics["error_rate"] > 0.05:  # 5%
                issues.append(f"{server_name}: High error rate ({metrics['error_rate']:.2%})")
            
            # GPU memory check
            if metrics["gpu_memory_usage"] > 0.95:  # 95%
                issues.append(f"{server_name}: High GPU memory usage ({metrics['gpu_memory_usage']:.1%})")
        
        return issues
```

### Performance Optimization

```python
from src.model_engine.adapters.vllm_adapter import VLLMAdapter

class OptimizedVLLMAdapter(VLLMAdapter):
    """Performance-optimized vLLM adapter."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_optimization()
    
    def setup_optimization(self):
        """Configure performance optimizations."""
        # Enable all available optimizations
        self.enable_prefix_caching()
        self.enable_chunked_prefill()
        self.configure_kv_cache_optimization()
        
        # Dynamic batching
        self.dynamic_batching = True
        self.max_waiting_time = 0.1  # 100ms
        
        # Memory optimization
        self.enable_memory_pool()
        self.configure_attention_optimization()
    
    async def optimized_batch_generate(self, prompts: List[str], **kwargs):
        """Optimized batch generation with dynamic batching."""
        # Group prompts by similar length for better batching
        length_groups = self.group_by_length(prompts)
        
        tasks = []
        for group in length_groups:
            task = self.generate_batch_async(group, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return self.merge_results(results, prompts)
    
    def group_by_length(self, prompts: List[str]) -> List[List[str]]:
        """Group prompts by similar token length."""
        # Implement length-based grouping for optimal batching
        pass
```

## 🛠️ Development and Testing

### Engine Testing Framework

```python
import pytest
from src.model_engine.factory import ModelEngineFactory

def test_vllm_engine_creation():
    """Test vLLM engine creation and basic functionality."""
    factory = ModelEngineFactory()
    
    engine = factory.create_engine(
        engine_type="vllm",
        model_name="microsoft/DialoGPT-small",  # Small model for testing
        task="text_generation"
    )
    
    assert engine is not None
    assert engine.health_check()["status"] == "healthy"

def test_text_generation():
    """Test text generation functionality."""
    engine = create_test_engine()
    
    prompts = ["Hello, how are you?", "Explain AI briefly."]
    responses = engine.generate(prompts)
    
    assert len(responses) == len(prompts)
    assert all(isinstance(response, str) for response in responses)
    assert all(len(response) > 0 for response in responses)

def test_embedding_generation():
    """Test embedding generation."""
    engine = create_test_embedding_engine()
    
    texts = ["Document chunk 1", "Document chunk 2"]
    embeddings = engine.embed(texts)
    
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) > 0 for emb in embeddings)

def test_server_manager():
    """Test server management functionality."""
    from src.model_engine.server_manager import ModelServerManager
    
    manager = ModelServerManager()
    
    # Test server lifecycle
    server_config = {
        "name": "test_server",
        "engine": "vllm",
        "model": "microsoft/DialoGPT-small",
        "port": 8888
    }
    
    manager.start_server(server_config)
    assert manager.is_server_running("test_server")
    
    health = manager.check_server_health("test_server")
    assert health["status"] in ["healthy", "starting"]
    
    manager.stop_server("test_server")
    assert not manager.is_server_running("test_server")
```

## 📚 Related Documentation

- **Configuration**: See `config/README.md` for model engine configuration
- **vLLM Documentation**: Official vLLM documentation for advanced features
- **Haystack Documentation**: Haystack framework documentation
- **Types**: See `types/model_engine/` for type definitions
- **API Integration**: See `api/README.md` for API integration patterns

## 🎯 Best Practices

1. **Choose appropriate engines** - Select engines based on your performance and scalability needs
2. **Monitor resource usage** - Track GPU memory, CPU usage, and server health
3. **Use batch processing** - Optimize throughput with appropriate batch sizes
4. **Configure properly** - Tune tensor parallelism and memory settings for your hardware
5. **Handle failures gracefully** - Implement proper error handling and retry logic
6. **Scale horizontally** - Use multiple servers for high-availability deployments
7. **Cache strategically** - Use prefix caching and other optimizations when appropriate
8. **Test thoroughly** - Validate performance and reliability under load
