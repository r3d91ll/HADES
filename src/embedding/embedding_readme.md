# Embedding Module

## Overview

The embedding module is responsible for generating vector embeddings for document chunks in the HADES-PathRAG pipeline. It takes the output from the chunking module and produces documents with embeddings that can be stored in the database.

## Components

### `base.py`
- Defines the core `EmbeddingAdapter` protocol and adapter registry
- Implements the adapter factory pattern for flexibility
- Provides adapter registration and retrieval functionality

### `adapters/cpu_adapter.py`
- Implements a CPU-optimized embedding adapter
- Uses lightweight models suitable for CPU processing
- Based on the sentence-transformers library
- Designed for environments where GPU resources may be limited

### `processors.py`
- Provides document processing functions that add embeddings to chunks
- Supports both single document and batch processing
- Includes both async and sync workflows
- Tracks embedding statistics for monitoring

## Usage

### Basic Usage

```python
from src.embedding.processors import process_chunked_document_file
import asyncio

# Process a single chunked document file
result, output_file = asyncio.run(process_chunked_document_file(
    file_path="path/to/chunked_document.json",
    output_dir="path/to/output",
    adapter_name="cpu"
))
```

### Batch Processing

```python
from src.embedding.processors import process_chunked_documents_batch
import asyncio

# Process multiple chunked documents
stats = asyncio.run(process_chunked_documents_batch(
    file_paths=["doc1.json", "doc2.json"],
    output_dir="path/to/output",
    adapter_name="cpu",
    adapter_options={"model_name": "all-MiniLM-L6-v2"}
))
```

## Configuration

The embedding module accepts the following configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| adapter_name | Name of the embedding adapter to use | "cpu" |
| model_name | Name of the model to use for embeddings | "all-MiniLM-L6-v2" |
| max_length | Maximum sequence length for tokenization | 512 |
| batch_size | Batch size for processing | 32 |

## Input/Output Format

### Input
The embedding module expects chunked documents in the following format:
```json
{
  "id": "document_id",
  "path": "path/to/source",
  "content": "document content",
  "type": "document type",
  "metadata": { ... },
  "chunks": [
    {
      "id": "chunk_id",
      "parent": "document_id",
      "content": "chunk content",
      "chunk_index": 0,
      "start_offset": 0,
      "end_offset": 100,
      ...
    }
  ]
}
```

### Output
The module adds embeddings to each chunk:
```json
{
  "id": "document_id",
  ...
  "chunks": [
    {
      "id": "chunk_id",
      "content": "chunk content",
      ...
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "_embedding_info": {
    "adapter": "cpu",
    "chunk_count": 10,
    "embedded_chunk_count": 10,
    "embedding_dimensions": 384,
    "timestamp": "..."
  }
}
```

## Dependencies

- `sentence-transformers`: For CPU-based embedding generation
- `numpy`: For vector operations
- `asyncio`: For asynchronous processing

## Extension Points

To add a new embedding adapter:
1. Create a new adapter class in `adapters/`
2. Implement the `EmbeddingAdapter` protocol
3. Register the adapter in `__init__.py`

## Testing

Run the unit tests with:
```bash
pytest tests/unit/embedding/
```

## Integration with Pipeline

The embedding module fits into the HADES-PathRAG pipeline between the chunking and storage modules:

1. Document Processing (`src/docproc`)
2. Chunking (`src/chunking`)
3. → Embedding (`src/embedding`) ←
4. Storage (`src/storage`)
