# HADES-PathRAG Engine Modules

## Overview

The HADES-PathRAG codebase contains one primary engine-related module:

1. **Model Engine** (`src/model_engine/`): Provides interfaces for language model inference

> **Note**: The former Batch Processing Engine (`src/engine/`) has been moved to the `dead-code` directory as it was superseded by more specialized pipeline implementations.

## Module Relationships

These modules have different purposes but work together in the system:

- `model_engine` is responsible for the actual language model interfaces and adapters
- `engine` provides the batch processing pipeline that uses models for specific tasks
- `pipelines/ingest` builds on both to implement a repository ingestion workflow

## Module Purposes

### Model Engine

The Model Engine module (`src/model_engine/`) provides:

- Unified interface for different model backends (vLLM, Haystack)
- Server management for model inference services
- Adapters for various model frameworks
- Embedding generation, completion, and chat functionality

### Batch Processing Engine

The Batch Processing Engine module (`src/engine/`) provides:

- Pipeline stages for document processing (document processing, chunking)
- Batch types for moving data between stages
- Orchestration of the entire pipeline
- GPU-acceleration with efficient resource management

### Ingest Pipeline

The Ingest Pipeline module (`src/pipelines/ingest/`) builds on both to provide:

- Specialized repository ingestion workflow
- Document processing and entity extraction
- Storage interfaces for ArangoDB

## When to Use Which Module

- Use `model_engine` when you need direct access to language model capabilities
- Use `engine` when you need to process batches of documents through a pipeline
- Use `pipelines/ingest` for the complete repository ingestion workflow

## Future Directions

As the codebase evolves, these modules may be further consolidated or specialized based on emerging requirements.
