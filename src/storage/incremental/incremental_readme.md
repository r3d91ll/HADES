# Incremental Storage Module

This module implements ArangoDB-based incremental storage for HADES, enabling efficient document ingestion and ISNE model updates without full reprocessing.

## Overview

The incremental storage system provides:

1. **Incremental Document Processing**: Only process new or changed documents
2. **Graph Preservation**: Maintain existing graph structure while adding new nodes
3. **Model Continuity**: Expand ISNE model capacity and fine-tune with new data
4. **Conflict Resolution**: Handle document overlaps with configurable strategies
5. **Version Control**: Track all changes with comprehensive versioning

## Architecture

### Database Schema

The system uses 9 specialized ArangoDB collections:

#### Document Collections
- **`documents`**: Source documents with content hashing
- **`chunks`**: Text chunks with embeddings and metadata
- **`embeddings`**: Vector embeddings with model versioning

#### Graph Collections  
- **`nodes`**: Graph nodes linking chunks to embeddings
- **`edges`**: Graph edges with weights and temporal bounds
- **`models`**: ISNE model versions and metadata

#### Change Tracking Collections
- **`ingestion_logs`**: Document processing logs
- **`model_versions`**: Model training history
- **`conflicts`**: Document conflict resolution records

### Key Components

#### 1. Schema Manager (`schema.py`)
- Defines collection schemas and indexes
- Manages database initialization and migrations
- Provides validation for all data structures

#### 2. Incremental Manager (`manager.py`) 
- Orchestrates incremental ingestion pipeline
- Handles change detection and conflict resolution
- Manages model capacity expansion

#### 3. Graph Builder (`graph_builder.py`)
- Constructs graph incrementally
- Connects new chunks to existing structure
- Optimizes edge creation for performance

#### 4. Model Updater (`model_updater.py`)
- Expands ISNE model capacity for new nodes
- Performs incremental training with new data
- Manages model versioning and rollback

#### 5. Conflict Resolver (`conflict_resolver.py`)
- Detects document content changes
- Resolves conflicts using configurable strategies
- Maintains audit trail for all resolutions

## Usage

### Basic Incremental Ingestion

```python
from src.storage.incremental import IncrementalManager

manager = IncrementalManager(
    db_name="hades_incremental",
    model_path="models/current_isne_model.pth"
)

# Process new documents
results = await manager.ingest_documents(
    input_dir="/path/to/new/docs",
    conflict_strategy="update"
)

print(f"Processed {results.new_documents} new documents")
print(f"Updated {results.updated_documents} existing documents")
print(f"Model expanded to {results.new_model_size} nodes")
```

### Advanced Configuration

```python
# Configure incremental processing
config = IncrementalConfig(
    conflict_strategy="merge",
    similarity_threshold=0.85,
    batch_size=100,
    enable_caching=True,
    model_expansion_strategy="gradual"
)

manager = IncrementalManager(config=config)
```

## Testing

### Unit Tests
- All functions tested with ≥85% coverage
- Mock ArangoDB connections for isolated testing
- Comprehensive edge case coverage

### Integration Tests
- End-to-end incremental ingestion pipeline
- Performance benchmarks for large-scale updates
- Model continuity validation across updates

### Performance Benchmarks
- Document processing throughput
- Graph construction efficiency
- Model training convergence metrics

## Performance Optimizations

1. **Content Hashing**: SHA256-based change detection
2. **Batch Processing**: Configurable batch sizes for all operations
3. **Intelligent Caching**: Redis-backed caching for similarity results
4. **Index Optimization**: Advanced ArangoDB indexing strategies
5. **Parallel Processing**: Multi-worker document processing

## Configuration

See `src/config/components/storage/incremental/config.yaml` for full configuration options.

## Dependencies

- ArangoDB Python driver
- PyTorch (for ISNE model operations)
- Redis (for caching)
- Pydantic (for schema validation)
- asyncio (for async operations)

## API Reference

### IncrementalManager

Main orchestrator for incremental operations.

#### Methods

- `ingest_documents(input_dir, **kwargs)` - Process new documents
- `update_model(training_config)` - Update ISNE model with new data
- `rollback_to_version(version_id)` - Rollback to previous state
- `get_ingestion_stats()` - Get processing statistics

### SchemaManager

Manages database schema and validation.

#### Methods

- `initialize_database()` - Create collections and indexes
- `validate_schema()` - Validate current schema state
- `migrate_schema(target_version)` - Migrate to new schema version

### GraphBuilder

Constructs graph incrementally.

#### Methods

- `add_chunks(chunks)` - Add new chunks to graph
- `build_edges(similarity_threshold)` - Create edges between chunks
- `optimize_graph()` - Optimize graph structure

### ModelUpdater

Manages ISNE model updates.

#### Methods

- `expand_model(new_node_count)` - Expand model capacity
- `train_incremental(new_data)` - Train model with new data
- `save_version(version_info)` - Save model version

## Error Handling

The module provides comprehensive error handling:

- **ValidationError**: Schema validation failures
- **ConflictError**: Document conflict resolution failures
- **ModelError**: ISNE model operation failures
- **StorageError**: Database operation failures

## Monitoring and Alerts

Integrated with HADES alert system:

- Document processing metrics
- Model training progress
- Error notifications
- Performance warnings

## Security Considerations

- Input validation for all operations
- Secure handling of document content
- Access control for model operations
- Audit logging for all changes

## Future Enhancements

1. **Distributed Processing**: Multi-node processing support
2. **Advanced Conflict Resolution**: ML-based conflict detection
3. **Streaming Updates**: Real-time document processing
4. **Cross-Model Updates**: Support for multiple model architectures