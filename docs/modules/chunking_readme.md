# HADES-PathRAG Chunking Module

This module provides document chunking capabilities for the HADES-PathRAG system, supporting both code and text documents.

## Overview

The chunking module contains two main chunking systems:

1. **AST-based chunking** for code files (`code_chunkers/`)
   - Preserves code structure using Abstract Syntax Tree analysis
   - Language-aware chunking for Python, JavaScript, Java, and C++

2. **Chonky semantic chunking** for text files (`text_chunkers/`)
   - Neural-based semantic chunking that respects content boundaries
   - Preserves original text formatting and casing
   - Supports overlap context for better retrieval
   - **CPU-optimized** implementation for large documents with parallel processing

## Key Features

- **Semantic Boundaries**: Identifies natural paragraph and section boundaries
- **Original Text Preservation**: Maintains original casing, formatting, and special characters
- **Overlap Context**: Stores context before and after each chunk for better retrieval
- **Content Hashing**: Efficiently stores and retrieves chunks using content hashing
- **CPU and GPU Support**: Optimized implementations for both CPU and GPU processing
- **Parallel Processing**: Multi-threaded CPU implementation for large documents
- **Configurable**: Highly customizable through configuration system

## Usage

### Basic Usage

```python
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.chunking.code_chunkers.ast_chunker import chunk_python_code

# Chunk a text document
text_document = {
    "content": "This is a sample text document...",
    "path": "/path/to/document.md",
    "type": "markdown"
}
text_chunks = chunk_text(text_document)

# Chunk a code document
code_document = {
    "content": "def example():\n    pass",
    "path": "/path/to/code.py",
    "type": "python"
}
code_chunks = chunk_python_code(code_document)
```

### CPU-Optimized Usage

For large documents, the CPU-optimized implementation can be used with parallel processing:

```python
from src.chunking.text_chunkers.chonky_chunker import ParagraphSplitter
from multiprocessing.pool import ThreadPool

# Initialize the paragraph splitter
splitter = ParagraphSplitter(
    model_id="mirth/chonky_modernbert_large_1",
    device="cpu",
    use_model_engine=False
)

# Process a large document in segments
content = "Large document content..."
segment_size = 10000  # characters
segments = []

# Split into segments with overlap
for i in range(0, len(content), segment_size):
    start = max(0, i - 200 if i > 0 else 0)
    end = min(len(content), i + segment_size + 200)
    segments.append(content[start:end])

# Process segments in parallel
with ThreadPool(4) as pool:
    segment_results = pool.map(splitter.split_text, segments)

# Combine results
paragraphs = []
for result in segment_results:
    paragraphs.extend(result)
```

## Configuration

### Chunking Configuration

The chunking behavior can be customized through configuration files:

#### Direct Chunker Configuration

See `src/config/chunker_config.yaml` for direct chunker configuration.

Key options include:

- **Model settings**: Model ID, engine, and device
- **Chunking parameters**: Token limits, overlap, and batch size
- **Overlap context settings**: Context storage options
- **Cache settings**: Device-specific caching options
- **Model engine settings**: Availability checks and startup options

#### Ingestion Pipeline Configuration

For the ingestion pipeline, the chunking configuration is part of `src/pipelines/ingest/orchestrator/config.py`:

```python
# CPU-only configuration
config = IngestionConfig.create_cpu_only()

# GPU-enabled configuration
config = IngestionConfig.create_gpu_enabled(device="cuda:0")

# Custom configuration
config = IngestionConfig(
    chunking=ChunkingConfig(
        use_gpu=False,
        device="cpu",
        max_tokens=2048,
        model_id="mirth/chonky_modernbert_large_1",
        use_semantic_chunking=True,
        num_cpu_workers=8
    ),
    embedding=EmbeddingConfig(
        use_gpu=False,
        device="cpu"
    )
)

# From environment variables
config = IngestionConfig.from_env()
```

For detailed configuration options, see the [Chunker Configuration Guide](../../docs/integration/chunker_configuration.md) and the [Ingestion Configuration Guide](../../docs/pipelines/ingestion_config.md).

## Performance Considerations

### CPU vs. GPU Performance

Benchmarking shows that our CPU implementation provides excellent semantic chunking quality while being practical for most ingestion workloads. Key findings:

- **Semantic Quality**: Both CPU and GPU implementations produce high-quality semantic paragraphs that respect natural content boundaries.
- **Processing Speed**: The GPU implementation is faster for large documents but requires a compatible GPU setup.
- **Parallel Processing**: Our CPU implementation uses multi-threading to process document segments concurrently, improving throughput.
- **Stability**: The CPU implementation has fewer dependencies and avoids potential GPU compatibility issues.
- **Resource Allocation**: Using CPU for chunking frees GPU resources for more intensive tasks like inference.

### Recommendations

- For ingestion pipelines: Use the CPU implementation to free GPU resources for inference.
- For real-time applications: Use the GPU implementation if available, with CPU fallback.
- For very large documents: Increase CPU worker count in both the chunking and embedding configurations.

## Testing

The chunking module includes comprehensive tests in the `tests/chunking/` directory:

```bash
# Run all chunking tests
python -m pytest tests/chunking/

# Run specific test file
python -m pytest tests/chunking/test_chonky_original_text.py

# Run benchmark comparison
python benchmark/benchmark_cpu_vs_gpu_chunking.py
```

## Type Safety

The module is fully type-checked with mypy:

```bash
# Run type checking
python -m mypy src/chunking/
```
