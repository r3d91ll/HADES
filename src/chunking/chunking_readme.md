# Chunking Module Documentation

## Overview
The chunking module provides semantic document chunking capabilities for HADES-PathRAG, with support for both CPU and GPU processing paths. It uses the Chonky neural paragraph splitting model to ensure semantically coherent paragraphs.

## Components

### 1. Text Chunkers
- **Chonky Chunker**: Uses the `mirth/chonky_modernbert_large_1` model for semantic paragraph chunking
- **Support for both CPU and GPU**: Configurable via environment variables

### 2. Environment Variables
- `CUDA_VISIBLE_DEVICES`: Set to empty string (`""`) to force CPU-only mode
- `HADES_DEFAULT_DEVICE`: Set to `"cpu"` for CPU mode or `"cuda:0"` (or other GPU index) for GPU mode
- `HADES_MODEL_MGR_SOCKET`: Path for the Haystack model manager socket

## Performance Considerations

### CPU Path
- **Advantages**:
  - Lower resource overhead
  - No need for GPU infrastructure
  - Suitable for smaller documents or batches
  - More accessible to deployment on standard servers
- **Disadvantages**:
  - Slower for large documents
  - Less suitable for high-throughput processing

### GPU Path
- **Advantages**:
  - Significantly faster for large documents and batches
  - Better suited for production-scale processing
- **Disadvantages**:
  - Requires GPU hardware
  - Higher resource consumption
  - Initialization overhead for loading models

## Recommendations

We recommend a **CPU-first approach** with optional GPU acceleration:

1. Start with CPU-based processing for development and smaller workloads
2. Enable GPU acceleration for:
   - Production deployments
   - Large document batches
   - When throughput becomes a bottleneck

This allows for maximum deployment flexibility while providing a performance upgrade path when needed.

## Usage Example

```python
# CPU-only mode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HADES_DEFAULT_DEVICE"] = "cpu"

from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Process document
chunks = chunk_text(
    content=document_text,
    doc_id="doc123",
    model_id="mirth/chonky_modernbert_large_1"
)

# GPU mode (if available)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["HADES_DEFAULT_DEVICE"] = "cuda:0"

from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Process document with GPU acceleration
chunks = chunk_text(
    content=document_text,
    doc_id="doc123",
    model_id="mirth/chonky_modernbert_large_1"
)
```

## Best Practices

1. **Use environment variables** to control CPU/GPU usage rather than hardcoding
2. **Pre-process documents** to remove unnecessary formatting before chunking
3. **Batch process documents** when possible for better throughput
4. **Monitor memory usage** especially when processing large documents on CPU
5. **Use the benchmark script** to evaluate performance on your specific hardware

## Benchmark Results

The `benchmark_cpu_vs_gpu_chunking.py` script provides detailed performance comparisons between CPU and GPU chunking. Key metrics include:
- Processing time
- Number of chunks generated
- Average chunk size (characters and tokens)
- Speedup factor

Run the benchmark to determine the best approach for your specific workload and hardware.
