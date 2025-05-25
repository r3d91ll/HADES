# vLLM Integration for HADES-PathRAG

This document outlines the integration of vLLM (Very Large Language Model) for accelerated model inference and embedding generation in HADES-PathRAG.

## Overview

vLLM is a high-performance, production-ready library for LLM inference and serving. It offers:

1. Significantly faster inference compared to Ollama due to:
   - PagedAttention for optimized KV cache management
   - Continuous batching for higher throughput
   - CUDA graph optimization
   - Multi-GPU pipeline parallelism

2. OpenAI-compatible API endpoints
3. Support for a wide range of models

## Configuration

vLLM settings are managed through the `src/config/vllm_config.py` module, which supports both code-based configuration and environment variables.

### Environment Variables

Create a `.env` file in your project root with the following variables:

```ini
# vLLM Server Settings
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.85

# Model Selection
VLLM_CODE_MODEL=Qwen2.5-coder
VLLM_GENERAL_MODEL=Llama-3-8b
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Default Models
VLLM_DEFAULT_MODEL=general
VLLM_DEFAULT_EMBEDDING_MODEL=embedding
```

### Model Configuration

The default configuration includes three model types:

1. **Code Model**: Specialized for code understanding and generation (default: Qwen2.5-coder)
2. **General Model**: For general text tasks (default: Llama-3-8b)
3. **Embedding Model**: For text embedding generation (default: BAAI/bge-large-en-v1.5)

## Usage in Code

### Starting the vLLM Server

The server is automatically started when needed using the vLLM adapter:

```python
from src.pathrag.vllm_adapter import vllm_model_complete

# The server will start automatically when this is called
response = await vllm_model_complete(
    prompt="What is PathRAG?",
    model_alias="general"  # Uses the model defined as "general" in config
)
```

### Generating Embeddings

```python
from src.pathrag.vllm_adapter import vllm_embed

embeddings = await vllm_embed(
    texts=["Document 1", "Document 2"],
    model_alias="embedding"  # Uses the model defined as "embedding" in config
)
```

## Integration with ISNE and Chonky

The vLLM embedding support is particularly important for:

1. **ISNE Embeddings**: Accelerates the embedding computations for graph relationships
2. **Chonky Semantic Chunking**: Provides fast embeddings for semantic text chunking

## Installation

vLLM requires CUDA-compatible GPUs. Install it with:

```bash
pip install vllm
```

For more advanced installation options (like building from source), see the [vLLM documentation](https://github.com/vllm-project/vllm).

## Performance Considerations

For optimal performance:

1. Use the highest `VLLM_GPU_MEMORY_UTILIZATION` your system can support (typically 0.85-0.9)
2. Increase `VLLM_TENSOR_PARALLEL_SIZE` if you have multiple GPUs
3. Batch embedding requests when possible

## Troubleshooting

Common issues:

1. **CUDA out of memory**: Reduce `VLLM_GPU_MEMORY_UTILIZATION` or use a smaller model
2. **Model not found**: Check model path/ID is correct and accessible
3. **Server not starting**: Ensure CUDA drivers are properly installed

For more information, check logs at `/tmp/vllm.log`.
