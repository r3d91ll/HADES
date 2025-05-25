# Pipeline Module

## Overview

The Pipeline module provides a modular, extensible architecture for document processing, embedding generation, ISNE enhancement, and storage operations in the HADES-PathRAG system. It's designed to support both batch processing for creating new datastores and incremental updates to existing datastores.

## Key Components

### 1. Pipeline Stages

The pipeline is built around the concept of stages, with each stage responsible for a specific aspect of document processing:

- **Document Processing Stage**: Extracts and preprocesses text from various document formats
- **Chunking Stage**: Divides documents into semantically meaningful chunks
- **Embedding Stage**: Generates vector embeddings for document chunks
- **ISNE Stage**: Enhances embeddings using graph-based relationships
- **Storage Stage**: Persists processed documents to the datastore

Each stage follows a consistent interface defined by the `PipelineStage` abstract base class, ensuring interoperability and modularity.

### 2. Data Schemas

Pydantic models define the structure of data flowing through the pipeline:

- **Document Schema**: Structure for entire documents
- **Chunk Schema**: Structure for individual document chunks
- **Validation Schema**: Structure for validation results

These schemas enforce data integrity and provide automatic validation at each stage.

### 3. Pipeline Orchestration

The pipeline orchestrator coordinates the execution of stages, handling:

- Stage sequencing and data flow
- Error handling and recovery
- Resource allocation
- Monitoring and reporting

## Usage

The pipeline can be used in two primary modes:

### Batch Processing (New Datastore)

This mode processes all documents in a batch, building a complete graph of relationships before storing to a new datastore:

```python
from src.pipeline import Pipeline
from src.pipeline.stages import DocProcStage, ChunkingStage, EmbeddingStage, ISNEStage, StorageStage

# Create pipeline stages
doc_proc = DocProcStage(config)
chunking = ChunkingStage(config)
embedding = EmbeddingStage(config)
isne = ISNEStage(config)
storage = StorageStage(config, mode="create")

# Create and run pipeline
pipeline = Pipeline([doc_proc, chunking, embedding, isne, storage])
results = pipeline.run(document_paths)
```

### Incremental Processing (Existing Datastore)

This mode processes documents incrementally, leveraging existing database content for relationship building:

```python
from src.pipeline import Pipeline
from src.pipeline.stages import DocProcStage, ChunkingStage, EmbeddingStage, ISNEStage, StorageStage

# Create pipeline stages
doc_proc = DocProcStage(config)
chunking = ChunkingStage(config)
embedding = EmbeddingStage(config)
isne = ISNEStage(config)
storage = StorageStage(config, mode="append")

# Create and run pipeline
pipeline = Pipeline([doc_proc, chunking, embedding, isne, storage])
results = pipeline.run(document_paths)
```

## Development Guidelines

When extending the pipeline:

1. New stages must inherit from `PipelineStage` and implement its interface
2. Data models should extend the appropriate Pydantic schemas
3. All stages should include validation and error handling
4. New functionality should be covered by unit and integration tests

## Testing

Each component includes comprehensive tests:

- Unit tests for individual stages
- Integration tests for stage interactions
- End-to-end tests for complete pipeline execution

## Future Improvements

- Support for distributed processing
- Dynamic stage configuration based on document types
- Advanced monitoring and visualization tools
- Integration with workflow orchestration systems
