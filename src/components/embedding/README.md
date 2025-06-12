# Embedding Component

The embedding component provides text vectorization functionality for the HADES-PathRAG system, converting text chunks into dense vector representations using various neural embedding models and strategies.

## Architecture Overview

The embedding system follows a hierarchical component architecture with specialized embedders for different deployment scenarios and performance requirements:

```
src/components/embedding/
├── core/                    # Core orchestration and factory
│   └── processor.py        # CoreEmbedder - Main coordinator
├── cpu/                    # CPU-optimized sentence transformers
│   └── processor.py        # CPUEmbedder - Local inference
├── gpu/                    # GPU-accelerated high-throughput
│   └── processor.py        # GPUEmbedder - Model engine integration
└── encoder/                # Custom transformer encoders
    └── processor.py        # EncoderEmbedder - Fine-tuned models
```

## Component Types

### Core Embedder (`core/processor.py`)
- **Purpose**: Factory and coordinator for different embedding strategies
- **Key Features**:
  - Dynamic embedder selection based on model requirements
  - Caching of embedder instances for performance
  - Configuration delegation to specialized embedders
  - Health monitoring and metrics collection
  - Support for embedding dimension discovery
- **Model Support**: Delegates to specialized embedders

### CPU Embedder (`cpu/processor.py`)
- **Purpose**: CPU-based inference using sentence transformers
- **Key Features**:
  - Sentence-transformers integration
  - Model engine abstraction (Haystack)
  - Fallback to basic text features
  - Memory-efficient processing
- **Model Support**: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase models
- **Best For**: Local inference, resource-constrained environments

### GPU Embedder (`gpu/processor.py`)
- **Purpose**: High-throughput GPU inference via model engines
- **Key Features**:
  - VLLM/Haystack model engine integration
  - GPU memory monitoring and optimization
  - Large batch processing (64+ samples)
  - Tensor parallelism support
- **Model Support**: Sentence-transformers, E5, BGE, GTE models
- **Best For**: Production workloads, high-volume processing

### Encoder Embedder (`encoder/processor.py`)
- **Purpose**: Custom transformer encoder models
- **Key Features**:
  - Direct transformer model integration
  - Multiple pooling strategies (mean, max, CLS, first, last)
  - Fine-tuned model support
  - Custom architecture compatibility
- **Model Support**: BERT, RoBERTa, DistilBERT, ELECTRA, custom encoders
- **Best For**: Research, custom models, fine-tuned embeddings

## Configuration

### Core Configuration
```yaml
embedding:
  default_embedder: "cpu"  # cpu | gpu | encoder
  
  # Model settings shared across embedders
  model_settings:
    model_name: "all-MiniLM-L6-v2"
    embedding_dimension: 384
    batch_size: 32
  
  # Processing options
  processing_options:
    normalize: true
    pooling: "mean"  # mean | max | cls
  
  # CPU-specific configuration
  cpu_config:
    model_name: "all-MiniLM-L6-v2"
    batch_size: 32
    max_length: 512
    normalize_embeddings: true
    model_engine_type: "haystack"
  
  # GPU-specific configuration
  gpu_config:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: 64
    max_length: 512
    gpu_memory_utilization: 0.8
    tensor_parallel_size: 1
    model_engine_type: "vllm"
  
  # Encoder-specific configuration
  encoder_config:
    model_name: "microsoft/DialoGPT-medium"
    batch_size: 16
    max_length: 512
    pooling_strategy: "mean"
    hidden_size: 768
    device: "auto"
```

### Model Engine Integration
The embedding components integrate with the model engine abstraction layer:

```yaml
model_engine:
  # VLLM for GPU embedders
  vllm:
    host: "localhost"
    port: 8000
    api_key: null
    timeout: 30
  
  # Haystack for CPU embedders
  haystack:
    cache_dir: "./cache/models"
    device: "cpu"
    use_fast_tokenizers: true
```

## Usage Examples

### Basic Embedding Generation
```python
from src.components.embedding.core.processor import CoreEmbedder
from src.types.components.contracts import EmbeddingInput, EmbeddingChunk

# Initialize embedder
embedder = CoreEmbedder({
    'default_embedder': 'cpu',
    'model_settings': {
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 32
    }
})

# Prepare input chunks
chunks = [
    EmbeddingChunk(
        id="chunk_001",
        content="This is the first text chunk for embedding.",
        metadata={}
    ),
    EmbeddingChunk(
        id="chunk_002", 
        content="This is the second text chunk for embedding.",
        metadata={}
    )
]

input_data = EmbeddingInput(
    chunks=chunks,
    model_name="all-MiniLM-L6-v2",
    embedding_options={'embedder_type': 'cpu'}
)

# Generate embeddings
result = embedder.embed(input_data)
print(f"Generated {len(result.embeddings)} embeddings")
for emb in result.embeddings:
    print(f"Chunk {emb.chunk_id}: {emb.embedding_dimension}D vector")
```

### GPU-Accelerated Processing
```python
from src.components.embedding.gpu.processor import GPUEmbedder

# Initialize GPU embedder
gpu_embedder = GPUEmbedder({
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'batch_size': 64,
    'gpu_memory_utilization': 0.8,
    'model_engine_type': 'vllm'
})

# Check health and GPU availability
if gpu_embedder.health_check():
    result = gpu_embedder.embed(input_data)
    metrics = gpu_embedder.get_metrics()
    
    print(f"GPU Memory Used: {metrics.get('gpu_memory_allocated_mb', 0):.2f} MB")
    print(f"Processing Speed: {result.embedding_stats['throughput_embeddings_per_second']:.2f} emb/sec")
```

### Custom Encoder Models
```python
from src.components.embedding.encoder.processor import EncoderEmbedder

# Initialize custom encoder
encoder = EncoderEmbedder({
    'model_name': 'microsoft/DialoGPT-medium',
    'pooling_strategy': 'mean',
    'hidden_size': 1024,
    'batch_size': 16,
    'normalize_embeddings': True
})

# Process with custom pooling
result = encoder.embed(input_data)
print(f"Encoder embedding dimension: {result.model_info['embedding_dimension']}")
```

### Embedding Estimation and Planning
```python
# Estimate processing requirements
token_count = embedder.estimate_tokens(input_data)
processing_time = embedder.estimate_embedding_time(input_data)
optimal_batch_size = embedder.get_optimal_batch_size()

print(f"Estimated tokens: {token_count}")
print(f"Estimated time: {processing_time:.2f} seconds")
print(f"Optimal batch size: {optimal_batch_size}")

# Check model support
model_name = "all-MiniLM-L6-v2"
if embedder.supports_model(model_name):
    dimension = embedder.get_embedding_dimension(model_name)
    print(f"Model {model_name} supported with {dimension}D embeddings")
```

## Model Support Matrix

### CPU Embedder Models
| Model | Dimension | Speed | Memory | Best Use Case |
|-------|-----------|-------|---------|---------------|
| all-MiniLM-L6-v2 | 384 | Fast | Low | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Medium | High quality |
| paraphrase-MiniLM-L6-v2 | 384 | Fast | Low | Paraphrase detection |
| all-distilroberta-v1 | 768 | Medium | Medium | Balanced performance |

### GPU Embedder Models  
| Model | Dimension | Throughput | GPU Memory | Best Use Case |
|-------|-----------|------------|------------|---------------|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Very High | Low | Production inference |
| BAAI/bge-large-en | 1024 | High | High | State-of-the-art quality |
| thenlper/gte-large | 1024 | High | High | Multilingual support |
| intfloat/e5-large | 1024 | High | High | Instruction following |

### Encoder Models
| Model | Dimension | Customization | Training | Best Use Case |
|-------|-----------|---------------|----------|---------------|
| bert-base-uncased | 768 | High | Easy | Fine-tuning base |
| roberta-large | 1024 | High | Medium | Advanced NLP |
| microsoft/DialoGPT-medium | 1024 | High | Hard | Conversational |
| distilbert-base-uncased | 768 | Medium | Easy | Lightweight custom |

## Performance Characteristics

### CPU Embedder
- **Throughput**: ~100-500 embeddings/sec
- **Memory**: 1-4GB RAM (model dependent)
- **Latency**: 10-50ms per batch
- **Scalability**: Single machine, multi-process
- **Dependencies**: sentence-transformers, torch

### GPU Embedder  
- **Throughput**: ~1000-5000 embeddings/sec
- **Memory**: 2-16GB VRAM (model dependent)
- **Latency**: 1-10ms per batch
- **Scalability**: Multi-GPU, tensor parallel
- **Dependencies**: VLLM/Haystack, CUDA

### Encoder Embedder
- **Throughput**: ~50-200 embeddings/sec
- **Memory**: 2-8GB (model dependent)
- **Latency**: 20-100ms per batch
- **Scalability**: Custom optimization
- **Dependencies**: transformers, torch

## Integration Patterns

### Pipeline Integration
```python
from src.orchestration.pipelines.data_ingestion.pipeline import DataIngestionPipeline

# Configure embedding stage in pipeline
pipeline_config = {
    'stages': {
        'embedding': {
            'component_type': 'embedding',
            'component_name': 'core',
            'config': {
                'default_embedder': 'gpu',
                'gpu_config': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'batch_size': 64
                }
            }
        }
    }
}
```

### Dynamic Model Selection
```python
def select_embedder_by_requirements(
    chunk_count: int, 
    latency_requirement: str,
    quality_requirement: str
) -> str:
    """Select optimal embedder based on requirements."""
    
    if chunk_count > 10000 and latency_requirement == "low":
        return "gpu"  # High throughput
    elif quality_requirement == "highest":
        return "encoder"  # Custom fine-tuned models
    else:
        return "cpu"  # General purpose

# Use in core embedder
embedder_type = select_embedder_by_requirements(
    chunk_count=len(chunks),
    latency_requirement="medium", 
    quality_requirement="standard"
)

core_embedder = CoreEmbedder({
    'default_embedder': embedder_type
})
```

### Batch Processing Optimization
```python
def optimize_batch_processing(
    embedder, 
    chunks: List[EmbeddingChunk], 
    max_memory_mb: int = 4000
) -> List[EmbeddingOutput]:
    """Optimize batch processing based on memory constraints."""
    
    optimal_batch_size = embedder.get_optimal_batch_size()
    
    # Adjust batch size based on memory constraints
    estimated_memory_per_item = max_memory_mb // optimal_batch_size
    if estimated_memory_per_item > 50:  # 50MB per item threshold
        batch_size = max_memory_mb // 50
    else:
        batch_size = optimal_batch_size
    
    results = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_input = EmbeddingInput(chunks=batch_chunks)
        result = embedder.embed(batch_input)
        results.append(result)
    
    return results
```

## Error Handling and Fallbacks

### Health Monitoring
```python
def monitor_embedder_health(embedder):
    """Monitor embedder health and performance."""
    
    if not embedder.health_check():
        logger.warning("Embedder health check failed")
        return False
    
    metrics = embedder.get_metrics()
    
    # Check performance thresholds
    if metrics['avg_processing_time'] > 5.0:  # 5 second threshold
        logger.warning("Embedder processing time degraded")
    
    # Check memory usage for GPU embedders
    if 'gpu_memory_allocated_mb' in metrics:
        if metrics['gpu_memory_allocated_mb'] > 8000:  # 8GB threshold
            logger.warning("High GPU memory usage")
    
    return True
```

### Fallback Strategies
```python
class RobustEmbedder:
    """Embedder with automatic fallback strategies."""
    
    def __init__(self):
        self.primary = CoreEmbedder({'default_embedder': 'gpu'})
        self.fallback = CoreEmbedder({'default_embedder': 'cpu'})
        self.basic = CoreEmbedder({'default_embedder': 'encoder'})
    
    def embed_with_fallback(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """Attempt embedding with automatic fallback."""
        
        # Try primary embedder (GPU)
        try:
            if self.primary.health_check():
                return self.primary.embed(input_data)
        except Exception as e:
            logger.warning(f"Primary embedder failed: {e}")
        
        # Try fallback embedder (CPU)
        try:
            if self.fallback.health_check():
                return self.fallback.embed(input_data)
        except Exception as e:
            logger.warning(f"Fallback embedder failed: {e}")
        
        # Use basic embedder as last resort
        return self.basic.embed(input_data)
```

## Testing and Validation

### Embedding Quality Validation
```python
def validate_embedding_quality(embeddings: List[ChunkEmbedding]) -> Dict[str, Any]:
    """Validate embedding quality metrics."""
    
    if not embeddings:
        return {"valid": False, "reason": "No embeddings generated"}
    
    # Check dimension consistency
    dimensions = [emb.embedding_dimension for emb in embeddings]
    if len(set(dimensions)) > 1:
        return {"valid": False, "reason": "Inconsistent embedding dimensions"}
    
    # Check for zero vectors
    zero_vectors = 0
    for emb in embeddings:
        if all(abs(x) < 1e-6 for x in emb.embedding):
            zero_vectors += 1
    
    if zero_vectors > len(embeddings) * 0.1:  # More than 10% zero vectors
        return {"valid": False, "reason": f"Too many zero vectors: {zero_vectors}"}
    
    # Calculate basic statistics
    avg_norm = np.mean([np.linalg.norm(emb.embedding) for emb in embeddings])
    
    return {
        "valid": True,
        "dimension": dimensions[0],
        "count": len(embeddings),
        "zero_vectors": zero_vectors,
        "avg_norm": avg_norm
    }
```

### Performance Benchmarking
```python
def benchmark_embedder(embedder, test_chunks: List[EmbeddingChunk]) -> Dict[str, float]:
    """Benchmark embedder performance."""
    
    import time
    
    input_data = EmbeddingInput(chunks=test_chunks)
    
    # Warm up
    embedder.embed(EmbeddingInput(chunks=test_chunks[:5]))
    
    # Benchmark
    start_time = time.time()
    result = embedder.embed(input_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    return {
        "processing_time": processing_time,
        "embeddings_per_second": len(result.embeddings) / processing_time,
        "tokens_per_second": embedder.estimate_tokens(input_data) / processing_time,
        "memory_efficiency": len(result.embeddings) / processing_time / 1024  # emb/sec/MB
    }
```

## Best Practices

### Model Selection Guidelines
1. **Production Workloads**: Use GPU embedder with VLLM for high throughput
2. **Development/Testing**: Use CPU embedder for simplicity and cost
3. **Research/Custom**: Use encoder embedder for fine-tuned models
4. **Resource Constrained**: Use CPU embedder with lightweight models

### Configuration Optimization
1. **Batch Size**: Balance memory usage and throughput
2. **Model Choice**: Consider embedding dimension vs quality tradeoffs
3. **Pooling Strategy**: Mean pooling for general purpose, CLS for classification
4. **Normalization**: Enable for similarity search applications

### Error Handling
1. Always implement health checks before processing
2. Use fallback embedders for robustness
3. Monitor memory usage, especially for GPU embedders
4. Validate embedding quality in production

## Related Components

- **Chunking**: Provides text chunks for embedding generation
- **Model Engine**: Abstracts model serving infrastructure
- **Storage**: Stores embeddings and vector indices
- **Pipeline**: Orchestrates embedding within processing workflows
- **Graph Enhancement**: Consumes embeddings for ISNE processing