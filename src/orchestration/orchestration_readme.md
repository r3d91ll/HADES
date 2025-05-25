# Orchestration Module

## Overview

The orchestration module provides a framework for building and managing parallel processing pipelines for document ingestion, embedding, chunking, and ISNE enhancement. It is designed to efficiently process large volumes of documents with automatic flow control and memory management.

## Directory Structure

```
src/orchestration/
  ├── __init__.py
  ├── orchestration_readme.md  # This file
  ├── core/                    # Core components
  │   ├── __init__.py
  │   ├── queue_manager.py     # Memory-aware queue with backpressure
  │   ├── parallel_worker.py   # Worker pool management
  │   └── monitoring.py        # Performance monitoring
  └── pipelines/               # Specific pipeline implementations
      ├── __init__.py
      ├── parallel_pipeline.py # Base parallel pipeline
      ├── text_pipeline.py     # Text processing pipeline
      └── training_pipeline.py # ISNE training pipeline
```

## Key Features

- **Parallel Processing**: Process multiple documents simultaneously for improved throughput
- **Memory Management**: Configurable memory limits with automatic backpressure
- **Flow Control**: Automatic throttling when downstream stages become overloaded
- **Configurable**: YAML-based configuration with profiles for different hardware capabilities
- **Modality Support**: Dedicated pipelines for different content types (text, code)
- **Monitoring**: Real-time metrics for pipeline performance and resource usage

## Configuration

Pipeline behavior is controlled via configuration files in `src/config/pipeline_config.yaml`. This allows for adjustment of:

- Queue sizes and memory limits
- Worker thread counts
- Backpressure thresholds
- Performance profiles (high-throughput, balanced, low-memory)

## Usage

Example of using the parallel text pipeline:

```python
from src.config.pipeline_config import load_pipeline_config
from src.orchestration.pipelines.text_pipeline import TextParallelPipeline

# Load configuration
config = load_pipeline_config(profile="balanced")

# Initialize pipeline
pipeline = TextParallelPipeline(config)

# Process a batch of documents
result = pipeline.process_batch([
    "/path/to/document1.txt",
    "/path/to/document2.txt",
    "/path/to/document3.txt"
])
```

## Integration with ISNE

The orchestration module is designed to integrate seamlessly with the ISNE module for document enhancement:

```python
from src.orchestration.pipelines.training_pipeline import ISNETrainingPipeline

# Initialize ISNE training pipeline
training_pipeline = ISNETrainingPipeline(config)

# Train ISNE model on a document corpus
training_pipeline.train(corpus_path="/path/to/corpus", 
                       epochs=10,
                       batch_size=32)
```
