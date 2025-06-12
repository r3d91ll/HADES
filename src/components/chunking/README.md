# Chunking Component

The chunking component provides text segmentation functionality for the HADES-PathRAG system, implementing various strategies to split documents into manageable chunks while preserving semantic and structural boundaries.

## Architecture Overview

The chunking system follows a hierarchical component architecture with specialized chunkers for different content types and performance requirements:

```
src/components/chunking/
├── core/                    # Core orchestration and factory
│   └── processor.py        # CoreChunker - Main coordinator
├── chunkers/               # Specialized chunker implementations
│   ├── cpu/               # CPU-optimized algorithms
│   ├── chonky/            # GPU-accelerated high-throughput
│   ├── code/              # Code-optimized chunking
│   └── text/              # Text-optimized chunking
└── factory.py             # Component factory
```

## Component Types

### Core Chunker (`core/processor.py`)
- **Purpose**: Factory and coordinator for different chunking strategies
- **Key Features**:
  - Dynamic chunker selection based on content type
  - Caching of chunker instances for performance
  - Configuration delegation to specialized chunkers
  - Health monitoring and metrics collection

### CPU Chunker (`chunkers/cpu/processor.py`)
- **Purpose**: CPU-optimized chunking without GPU dependencies
- **Key Features**:
  - Multiple chunking methods: fixed, sentence-aware, paragraph-aware, adaptive
  - NLTK integration for advanced sentence boundary detection
  - Configurable overlap and boundary preservation
  - Language-aware processing
- **Best For**: General text processing, resource-constrained environments

### Chonky Chunker (`chunkers/chonky/processor.py`)
- **Purpose**: GPU-accelerated high-throughput chunking
- **Key Features**:
  - Transformer tokenizer integration (HuggingFace)
  - CUDA GPU acceleration with memory monitoring
  - Batch processing for large document sets
  - Fallback to character-based chunking
- **Best For**: High-volume processing, GPU-enabled environments

### Code Chunker (`chunkers/code/processor.py`)
- **Purpose**: Code-specific chunking with larger context windows
- **Key Features**:
  - Code-optimized chunk sizes (1024+ characters)
  - Language-specific size adjustments
  - Structure-preserving chunking strategies
  - Support for multiple programming languages
- **Best For**: Source code, configuration files, structured data

### Text Chunker (`chunkers/text/processor.py`)
- **Purpose**: Text-optimized chunking with sentence awareness
- **Key Features**:
  - Sentence boundary preservation
  - Smaller, more granular chunks (512 characters)
  - Document structure awareness
  - Natural language processing optimizations
- **Best For**: Natural language text, documents, articles

## Configuration

### Core Configuration
```yaml
chunking:
  default_chunker: "cpu"  # cpu | chonky | code | text
  
  # CPU-specific configuration
  cpu_config:
    chunking_method: "sentence_aware"
    chunk_size: 512
    chunk_overlap: 50
    preserve_sentence_boundaries: true
    language: "en"
  
  # Chonky (GPU) configuration
  chonky_config:
    device: "cuda"
    batch_size: 32
    tokenizer_model: "bert-base-uncased"
    max_token_length: 512
    chunk_overlap_tokens: 50
  
  # Code-specific configuration
  code_config:
    chunk_size: 1024
    chunk_overlap: 100
    min_chunk_size: 200
  
  # Text-specific configuration
  text_config:
    chunking_method: "sentence_aware"
    chunk_size: 512
    preserve_sentence_boundaries: true
```

### Chunking Methods

#### CPU Chunker Methods
1. **Fixed**: Simple character-based chunking with fixed boundaries
2. **Sentence Aware**: Respects sentence boundaries using regex or NLTK
3. **Paragraph Aware**: Chunks at paragraph boundaries
4. **Adaptive**: Combines sentence-aware with small chunk merging

#### Content Type Optimization
- **Text**: 512 chars, sentence-aware, boundary preservation
- **Code**: 1024 chars, fixed chunking, larger overlap
- **Markdown**: 768 chars, structure-aware
- **JSON/YAML**: 512 chars, compact processing
- **Documents**: 1024 chars, paragraph-aware

## Usage Examples

### Basic Chunking
```python
from src.components.chunking.core.processor import CoreChunker
from src.types.components.contracts import ChunkingInput

# Initialize chunker
chunker = CoreChunker({
    'default_chunker': 'cpu',
    'cpu_config': {
        'chunking_method': 'sentence_aware',
        'chunk_size': 512,
        'chunk_overlap': 50
    }
})

# Prepare input
input_data = ChunkingInput(
    text="Your document text here...",
    document_id="doc_001",
    chunk_size=512,
    chunk_overlap=50,
    processing_options={'chunker_type': 'cpu'}
)

# Perform chunking
result = chunker.chunk(input_data)
print(f"Created {len(result.chunks)} chunks")
```

### GPU-Accelerated Chunking
```python
from src.components.chunking.chunkers.chonky.processor import ChonkyChunker

# Initialize GPU chunker
gpu_chunker = ChonkyChunker({
    'device': 'cuda',
    'batch_size': 32,
    'tokenizer_model': 'bert-base-uncased',
    'max_token_length': 512
})

# Check health before processing
if gpu_chunker.health_check():
    result = gpu_chunker.chunk(input_data)
    metrics = gpu_chunker.get_metrics()
    print(f"GPU Memory Used: {metrics['gpu_memory_allocated_mb']:.2f} MB")
```

### Code-Specific Chunking
```python
from src.components.chunking.chunkers.code.processor import CodeChunker

# Initialize code chunker
code_chunker = CodeChunker({
    'chunk_size': 1024,
    'chunk_overlap': 100
})

# Process source code
code_input = ChunkingInput(
    text=source_code,
    document_id="main.py",
    chunk_size=1024,
    chunk_overlap=100
)

result = code_chunker.chunk(code_input)
```

### Content Type Selection
```python
def select_chunker_by_content(content_type: str) -> str:
    """Select optimal chunker based on content type."""
    if content_type in ['python', 'javascript', 'java', 'code']:
        return 'code'
    elif content_type in ['text', 'markdown', 'document']:
        return 'text'
    elif torch.cuda.is_available():
        return 'chonky'  # GPU acceleration when available
    else:
        return 'cpu'     # Fallback
```

## Component Integration

### Factory Pattern Usage
```python
from src.components.chunking.factory import ChunkingComponentFactory

# Create chunker via factory
factory = ChunkingComponentFactory()
chunker = factory.create_component('chunking', {
    'component_name': 'core',
    'default_chunker': 'cpu'
})
```

### Pipeline Integration
```python
from src.orchestration.pipelines.data_ingestion.pipeline import DataIngestionPipeline

# Configure pipeline with chunking component
pipeline_config = {
    'stages': {
        'chunking': {
            'component_type': 'chunking',
            'component_name': 'core',
            'config': {
                'default_chunker': 'cpu',
                'cpu_config': {
                    'chunking_method': 'sentence_aware',
                    'chunk_size': 512
                }
            }
        }
    }
}
```

## Performance Characteristics

### CPU Chunker
- **Throughput**: ~1000 chars/sec
- **Memory**: Low memory footprint
- **Latency**: Medium (sentence detection overhead)
- **Dependencies**: Minimal (optional NLTK)

### Chonky Chunker
- **Throughput**: ~10,000+ chars/sec (GPU dependent)
- **Memory**: High GPU memory usage
- **Latency**: Low (batch processing)
- **Dependencies**: PyTorch, transformers, CUDA

### Code Chunker
- **Throughput**: ~800 chars/sec
- **Memory**: Low memory footprint
- **Latency**: Low (simple fixed chunking)
- **Dependencies**: Minimal

### Text Chunker
- **Throughput**: ~1200 chars/sec
- **Memory**: Low memory footprint
- **Latency**: Medium (sentence awareness)
- **Dependencies**: Minimal

## Error Handling and Fallbacks

### GPU Fallback Strategy
```python
# Chonky chunker automatically falls back to character-based chunking
# when GPU processing fails

def safe_gpu_chunking(input_data):
    chunker = ChonkyChunker({'device': 'cuda'})
    if not chunker.health_check():
        # Fallback to CPU chunker
        chunker = CPUChunker()
    return chunker.chunk(input_data)
```

### Robustness Features
- **Automatic Fallbacks**: GPU → CPU, advanced → simple methods
- **Error Recovery**: Continue processing with degraded functionality
- **Health Monitoring**: Regular health checks and metrics collection
- **Configuration Validation**: Schema-based config validation

## Testing and Validation

### Health Checks
```python
# All chunkers implement health_check()
healthy = chunker.health_check()
if not healthy:
    logger.warning("Chunker health check failed")
```

### Metrics Collection
```python
# Get performance metrics
metrics = chunker.get_metrics()
print(f"Chunks created: {metrics['total_chunks_created']}")
print(f"Avg processing time: {metrics['avg_processing_time']:.3f}s")
```

### Chunk Estimation
```python
# Estimate chunks before processing
estimated_count = chunker.estimate_chunks(input_data)
print(f"Estimated {estimated_count} chunks")
```

## Best Practices

### Content Type Mapping
1. **Text Documents**: Use text chunker with sentence awareness
2. **Source Code**: Use code chunker with larger chunks and overlap
3. **Mixed Content**: Use core chunker with dynamic selection
4. **High Volume**: Use chonky chunker when GPU is available

### Configuration Guidelines
1. **Chunk Size**: Balance between context and processing efficiency
2. **Overlap**: 10-20% of chunk size for good context preservation
3. **Boundaries**: Enable sentence/paragraph preservation for text
4. **GPU Memory**: Monitor memory usage for large batch processing

### Error Handling
1. Always implement fallback strategies
2. Monitor health status before processing
3. Validate configuration schemas
4. Log performance metrics for optimization

## Related Components

- **Document Processing**: Provides input documents for chunking
- **Embedding**: Consumes chunks for vector generation
- **Storage**: Stores chunk metadata and relationships
- **Pipeline**: Orchestrates chunking within processing workflows