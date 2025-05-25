# ModernBERT Embedding Adapter

## Overview

The ModernBERT embedding adapter provides high-quality vector embeddings for document chunks using the [ModernBERT model](https://huggingface.co/answerdotai/ModernBERT-base) from AnswerDotAI. This adapter is particularly well-suited for academic and technical documents due to ModernBERT's 8K token context window, which allows it to effectively capture semantics in large chunks of text.

## Features

- **Long Context Support**: Handles up to 8,192 tokens per chunk, ideal for academic papers with large semantic sections
- **Haystack Integration**: Uses the Haystack server for efficient model loading and management
- **Pooling Strategies**: Supports different embedding pooling methods (CLS token, mean, max)
- **Batch Processing**: Efficiently processes multiple chunks in batches to optimize memory usage
- **L2 Normalization**: Normalizes embeddings for improved similarity comparisons

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "answerdotai/ModernBERT-base" | Model identifier or path |
| `max_length` | 8192 | Maximum sequence length in tokens |
| `pooling_strategy` | "cls" | Embedding pooling method (cls, mean, max) |
| `batch_size` | 8 | Number of chunks to process at once |
| `normalize_embeddings` | True | Whether to L2-normalize vectors |

## Integration with HADES-PathRAG

The ModernBERT adapter is designed to work with the document transformation pipeline in HADES-PathRAG:

1. Documents are first converted to a normalized markdown format
2. Chonky splits documents into semantically meaningful chunks
3. ModernBERT generates high-quality embeddings for these chunks
4. Embeddings are stored for retrieval by the PathRAG system

## Usage Example

```python
from src.embedding.base import get_adapter

# Initialize the adapter
adapter = get_adapter("modernbert", max_length=8192, pooling_strategy="cls")

# Generate embeddings for chunks
embeddings = await adapter.embed(chunks)
```

## Prerequisites

- Haystack server must be running (starts automatically)
- PyTorch with appropriate CUDA support for GPU acceleration
- Transformer libraries (automatically installed with project dependencies)

## Benchmarks

| Document Type | Avg. Chunk Size | Embedding Time (CPU) | Embedding Time (GPU) |
|---------------|-----------------|----------------------|----------------------|
| Academic PDF  | 1000 tokens     | TBD                  | TBD                  |
| Code Files    | 500 tokens      | TBD                  | TBD                  |
| HTML Content  | 800 tokens      | TBD                  | TBD                  |

## Implementation Details

The ModernBERT adapter leverages Haystack's model management capabilities to efficiently load and use the model. It communicates with the Haystack server via a JSON-RPC interface over a Unix domain socket. Embeddings are generated through the following process:

1. The model is loaded by Haystack if not already in memory
2. Text is tokenized with appropriate sequence length handling
3. Forward pass through the model generates hidden states
4. Embeddings are extracted based on the pooling strategy
5. Vectors are normalized and returned in a serializable format

## Future Improvements

- Implement parallel batch processing for faster embedding generation
- Add support for custom ModernBERT variants
- Optimize for specific document types
- Enhance caching mechanisms for repeated text

## Limitations

- Large chunk sizes can lead to memory pressure on GPUs with limited VRAM
- Embedding quality for very specialized domains may require fine-tuning
