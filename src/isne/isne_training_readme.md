# ISNE Training Module Documentation

## Overview

The ISNE (Inductive Shallow Node Embedding) Training Module provides specialized functionality for training the ISNE model on text documents. This module processes PDF files in a specified directory (including nested subdirectories), creates document embeddings, builds a document graph incorporating directory structure relationships, and trains the ISNE model to enhance these embeddings with graph structural information.

## Core Components

### TextTrainingPipeline Class

The primary class that orchestrates the complete training workflow:

- **Document Processing**: Transforms source documents into normalized text
- **Chunking**: Splits documents into semantic chunks optimized for ModernBERT token limits
- **Embedding**: Computes vector embeddings for chunks using ModernBERT with memory-efficient batching
- **Graph Construction**: Builds a graph structure linking chunks through multiple relationship types:
  - Sequential relationships within documents
  - Directory structure relationships (documents in same folders)
  - Semantic similarity between chunks
- **ISNE Training**: Trains the ISNE model on the constructed graph using the specified device (CPU/GPU)
- **Model Saving**: Persists the trained model for later use
- **Performance Reporting**: Provides comprehensive statistics for both document processing and ISNE training

## Usage

### Command-line Interface

```bash
python -m src.pipelines.ingest.orchestrator.isne_training_text \
    --input_dir /path/to/pdf/directory \
    --output_dir ./test-output/isne-training \
    --model_dir ./models/isne \
    --file_pattern "*.pdf" \
    --epochs 50 \
    --learning_rate 0.001 \
    --hidden_dim 256 \
    --device cpu
```

### Programmatic Usage

```python
from src.pipelines.ingest.orchestrator.isne_training_text import run_training_pipeline
import asyncio

asyncio.run(run_training_pipeline(
    input_dir="/path/to/pdf/directory",
    output_dir="./test-output/isne-training",
    model_output_dir="./models/isne",
    training_options={
        "epochs": 50,
        "learning_rate": 0.001,
        "hidden_dim": 256
    },
    file_pattern="*.pdf",
    save_intermediate_results=True
))
```

## Training Process

1. **Document Collection**:
   - All PDF files in the input directory and nested subdirectories are identified
   - Directory structure is recorded for use in graph construction

2. **Document Processing**: Each document is processed through:
   - Text extraction and normalization
   - Semantic chunking optimized for ModernBERT token limits
   - Embedding computation with memory-efficient batching

3. **Graph Construction**:
   - Chunks within documents are connected sequentially
   - Documents in the same directory receive additional edge connections
   - Similarity-based connections are created between related chunks

4. **ISNE Training**:
   - The model learns to enhance embeddings based on graph structure
   - Training uses a combination of reconstruction loss and structural preservation
   - GPU optimization techniques prevent out-of-memory errors
   - The model is periodically checkpointed and evaluated

5. **Performance Reporting**:
   - Detailed statistics for document processing stages
   - ISNE training metrics including loss values and convergence data
   - Total processing times and resource utilization

6. **Model Persistence**:
   - The best model is saved for later use
   - Intermediate results are saved at each stage for analysis

## Configuration Options

### Chunking Options

- `max_tokens_per_chunk`: Maximum token count per chunk (default: 4096, optimized for ModernBERT)
- `doc_type`: Document type for chunking strategy (default: "academic_pdf")

### Embedding Options

- `adapter_name`: Embedding adapter to use (default: "modernbert")
- `model_name`: Specific model for the adapter (default: "answerdotai/ModernBERT-base")
- `device`: Computation device (default: "cpu", configurable to "cuda:0", "cuda:1", etc.)
- `batch_size`: Batch size for embedding generation (default: 4, reduced to prevent CUDA OOM errors)
- `normalize_embeddings`: Whether to L2-normalize embeddings (default: True)

### Training Options

- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of ISNE layers (default: 2)
- `num_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout probability (default: 0.1)
- `early_stopping_patience`: Epochs without improvement before stopping (default: 10)
- `checkpoint_interval`: Save frequency in epochs (default: 5)

## Testing and Performance

For testing module performance, use the following command to train on a small set of documents:

```bash
python -m src.pipelines.ingest.orchestrator.isne_training_text \
    --input_dir ./data/test-papers \
    --output_dir ./test-output/isne-test \
    --epochs 10 \
    --device cpu
```

## Implementation Notes

### Directory Traversal

The system now supports recursive directory traversal to locate documents:

- All supported document formats are discovered in nested subdirectories
- The directory structure is incorporated into the ISNE graph construction
- Documents in the same directory receive additional edge connections
- Configurable directory relationship weight (default: 0.5)

### GPU Optimization

The system includes several optimizations for GPU processing:

- Dynamic device selection based on configuration
- Reduced embedding batch size (default: 4) to prevent CUDA out-of-memory errors
- Optimized token limits for ModernBERT (max_tokens_per_chunk: 4096)
- Proper memory management during embedding generation

### Performance Reporting

Comprehensive performance metrics are now available:

- Document processing time breakdowns (extraction, chunking, embedding)
- ISNE training statistics (time, loss, epochs)
- Memory usage and GPU utilization metrics
- Total documents, chunks, and tokens processed

## Future Improvements

- Integration with GPU accelerated embedding extraction via vLLM
- Support for distributed training on larger document collections
- Implementation of more sophisticated document relation extraction
- Dynamic adjustment of graph construction parameters
- Further GPU memory optimization strategies
- Automated performance tuning based on document collection size
