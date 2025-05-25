# Orchestration Pipelines

## Overview

The pipelines module provides parallel processing implementations for different document processing workflows. These pipelines leverage the core orchestration components for queue management, worker pools, and monitoring to efficiently process large volumes of documents.

## Pipeline Types

### Parallel Pipeline (Base Class)

The `ParallelPipeline` is the foundation for all specialized pipelines, providing common functionality:

- Worker pool management
- Queue configuration with backpressure
- Performance monitoring
- Batch processing capabilities

### Text Parallel Pipeline

The `TextParallelPipeline` specializes in processing text documents through these stages:

1. Document Processing - Parse and extract metadata from text documents
2. Chunking - Split documents into semantic chunks
3. Embedding - Generate vector embeddings for chunks
4. Enhancement - Apply ISNE enhancement to improve embeddings

### ISNE Training Pipeline

The `ISNETrainingPipeline` focuses on training ISNE models:

1. Document Ingestion - Process training corpus documents
2. Graph Construction - Build document relationship graph
3. Model Training - Train ISNE model with configured parameters
4. Evaluation - Measure model performance on validation data

## Configuration

Pipeline behavior is controlled through YAML configuration in `src/config/pipeline_config.yaml`, which defines:

- Worker thread allocations
- Queue memory limits
- Backpressure thresholds
- Performance profiles (high-throughput, balanced, low-memory)

## Usage Examples

### Text Processing

```python
from src.config.pipeline_config import load_pipeline_config
from src.orchestration.pipelines.text_pipeline import TextParallelPipeline

# Load configuration with balanced profile
config = load_pipeline_config(profile="balanced")

# Initialize pipeline
pipeline = TextParallelPipeline(config)

# Process batch of documents
results = pipeline.process_batch([
    "/path/to/document1.txt",
    "/path/to/document2.txt"
])
```

### ISNE Training

```python
from src.config.pipeline_config import load_pipeline_config
from src.orchestration.pipelines.training_pipeline import ISNETrainingPipeline

# Load configuration with high-throughput profile for training
config = load_pipeline_config(profile="high_throughput")

# Initialize training pipeline
pipeline = ISNETrainingPipeline(config)

# Train model on corpus
results = pipeline.train(
    corpus_path="/path/to/training/corpus",
    epochs=10,
    batch_size=32
)
```

## Integration Points

The pipelines integrate with other HADES-PathRAG components:

- **Document Processing**: Uses adapters from `src/docproc`
- **Chunking**: Leverages chunkers from `src/chunking`
- **Embedding**: Integrates with models from `src/embedding`
- **ISNE**: Connects to model layers in `src/isne`
