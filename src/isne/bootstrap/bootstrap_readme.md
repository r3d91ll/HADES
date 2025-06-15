# ISNE Bootstrap Pipeline

## Overview

The ISNE Bootstrap Pipeline is a production-ready system for creating Inductive Shallow Node Embedding (ISNE) models from large document collections. This pipeline handles the complete end-to-end process from raw documents to trained ISNE models that can enhance graph-aware embeddings for retrieval-augmented generation (RAG) systems.

## Purpose

The bootstrap pipeline serves as a **one-time model creation system** that:

1. **Processes diverse document collections** (PDFs, Python code, text files)
2. **Creates high-quality embeddings** using state-of-the-art models
3. **Constructs knowledge graphs** from document relationships
4. **Trains ISNE models** to capture graph structure in embeddings
5. **Outputs production-ready models** for integration with RAG systems

This pipeline is distinct from the API-accessible training pipeline and is designed for comprehensive model bootstrapping operations.

## Architecture

### Pipeline Stages

The bootstrap pipeline consists of five sequential stages:

```
Input Files → Document Processing → Chunking → Embedding → Graph Construction → ISNE Training → Model Output
```

#### 1. Document Processing (`stages/document_processing.py`)
- **Purpose**: Convert various file formats into structured documents
- **Supported Formats**: PDF, Python, Markdown, JSON, YAML, text files
- **Output**: Standardized document objects with metadata
- **Components**: Uses `src.components.docproc.factory` for adapter selection

#### 2. Chunking (`stages/chunking.py`)
- **Purpose**: Split documents into semantic chunks suitable for embedding
- **Strategies**: Semantic, sliding window, paragraph, sentence-based
- **Features**: Preserves document structure and metadata
- **Components**: Uses `src.components.chunking.factory` for chunker selection

#### 3. Embedding (`stages/embedding.py`)
- **Purpose**: Generate vector embeddings for all document chunks
- **Models**: Supports sentence-transformers, ModernBERT, custom models
- **Features**: Batch processing, GPU acceleration, quality validation
- **Components**: Uses `src.components.embedding.factory` for embedder selection

#### 4. Graph Construction (`stages/graph_construction.py`)
- **Purpose**: Build knowledge graph from embeddings and relationships
- **Features**: Multi-hop relationships, semantic similarity edges
- **Storage**: Compatible with ArangoDB, NetworkX, nano-vectordb
- **Output**: Graph structure ready for ISNE training

#### 5. ISNE Training (`stages/isne_training.py`)
- **Purpose**: Train graph neural network to enhance embeddings
- **Model**: Multi-layer GNN with attention mechanisms
- **Losses**: Feature, structural, and contrastive loss functions
- **Output**: Trained ISNE model (.pth file) with metadata

### Configuration System

The pipeline uses a comprehensive configuration system located in `config.py`:

```python
from src.isne.bootstrap.config import (
    BootstrapConfig,
    DocumentProcessingConfig,
    ChunkingConfig,
    EmbeddingConfig,
    GraphConstructionConfig,
    ISNETrainingConfig
)

# Load configuration
config = BootstrapConfig.from_yaml("bootstrap_config.yaml")
```

### Monitoring Integration

Real-time monitoring and alerting via `monitoring.py`:

- **Progress Tracking**: Stage-by-stage progress reporting
- **Performance Metrics**: Processing speeds, memory usage, error rates
- **Alert System**: Integration with `src.alerts.AlertManager`
- **Resource Monitoring**: GPU usage, memory consumption, disk I/O

## Usage

### Basic Usage

```python
from src.isne.bootstrap import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig

# Load configuration
config = BootstrapConfig.from_yaml("config/bootstrap_config.yaml")

# Initialize pipeline
pipeline = ISNEBootstrapPipeline(config)

# Run complete bootstrap process
result = pipeline.run(
    input_files=[Path("/path/to/docs"), Path("/path/to/code")],
    output_dir=Path("/path/to/output")
)

# Access trained model
model_path = result.model_path
stats = result.training_stats
```

### Advanced Configuration

```yaml
# bootstrap_config.yaml
document_processing:
  processor_type: "docling"  # or "core"
  supported_formats: ["pdf", "py", "md", "txt", "json", "yaml"]
  extract_metadata: true
  extract_sections: true
  batch_size: 10

chunking:
  chunker_type: "chonky"  # GPU-accelerated
  strategy: "semantic"
  chunk_size: 512
  chunk_overlap: 50
  preserve_structure: true

embedding:
  embedder_type: "gpu"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  device: "cuda"
  normalize: true

graph_construction:
  similarity_threshold: 0.7
  max_edges_per_node: 10
  include_metadata_edges: true
  
isne_training:
  hidden_dim: 128
  num_layers: 3
  num_heads: 8
  learning_rate: 0.001
  epochs: 50
  batch_size: 1024
```

### Command Line Usage

```bash
# Run bootstrap pipeline with custom config
python -m src.isne.bootstrap.cli \
  --config config/bootstrap_config.yaml \
  --input-dir /path/to/documents \
  --output-dir /path/to/output \
  --model-name "comprehensive_isne_model"

# Monitor progress
python -m src.isne.bootstrap.monitor \
  --run-id "bootstrap_20250614_081411"
```

## Integration with HADES

### Model Loading in PathRAG

```python
from src.pathrag import PathRAG
from src.isne import ISNEModel

# Load trained ISNE model
isne_model = ISNEModel.load("/path/to/isne_model.pth")

# Initialize PathRAG with ISNE enhancement
pathrag = PathRAG(
    graph_storage=arango_storage,
    vector_storage=vector_storage,
    isne_model=isne_model,
    enhance_embeddings=True
)

# Enhanced retrieval
results = pathrag.query(
    question="What is graph neural network attention?",
    mode="hybrid"
)
```

### API Integration

The bootstrap pipeline integrates with the HADES FastAPI service for model management:

```python
# Register trained model
POST /api/v1/models/isne/register
{
    "name": "comprehensive_isne_model",
    "path": "/path/to/model.pth",
    "metadata": {...}
}

# Use model for enhancement
POST /api/v1/embeddings/enhance
{
    "embeddings": [...],
    "model_name": "comprehensive_isne_model"
}
```

## Performance Characteristics

### Resource Requirements

- **Memory**: 8-16 GB RAM (depends on dataset size)
- **GPU**: 4-8 GB VRAM (optional but recommended)
- **Storage**: 2-5x input data size for intermediate files
- **Time**: 10-60 minutes for 100K-1M documents

### Scalability

- **Document Limit**: Tested up to 500K documents
- **Batch Processing**: Configurable batch sizes for memory management
- **Distributed Training**: Support for multi-GPU ISNE training
- **Incremental Updates**: Planned feature for model updates

## Output Structure

```
output_directory/
├── models/
│   ├── isne_model_final.pth          # Trained ISNE model
│   ├── model_metadata.json           # Model configuration and stats
│   └── training_history.json         # Training loss curves
├── data/
│   ├── processed_documents.json      # Document processing results
│   ├── chunks.json                   # Chunking results
│   ├── embeddings.npz                # Base embeddings
│   └── graph_data.json              # Graph construction results
├── logs/
│   ├── pipeline_log.txt              # Detailed execution log
│   ├── stage_timings.json            # Performance metrics
│   └── error_log.txt                 # Error details if any
└── config/
    └── used_config.yaml              # Configuration snapshot
```

## Quality Assurance

### Validation Checks

- **Document Processing**: Content extraction validation, format support checks
- **Chunking**: Chunk size distribution, overlap validation
- **Embedding**: Dimension consistency, quality metrics (cosine similarity)
- **Graph Construction**: Node/edge count validation, connectivity checks
- **ISNE Training**: Loss convergence, gradient monitoring, dtype compatibility

### Error Handling

- **Stage Isolation**: Failures in one stage don't affect others
- **Graceful Degradation**: Partial processing continues when possible
- **Detailed Logging**: Comprehensive error traces for debugging
- **Alert Integration**: Real-time notifications for critical failures

## Testing

### Unit Tests

```bash
# Test individual stages
poetry run pytest tests/unit/isne/bootstrap/test_document_processing.py
poetry run pytest tests/unit/isne/bootstrap/test_chunking.py
poetry run pytest tests/unit/isne/bootstrap/test_embedding.py

# Test configuration system
poetry run pytest tests/unit/isne/bootstrap/test_config.py
```

### Integration Tests

```bash
# End-to-end pipeline test
poetry run pytest tests/integration/isne/test_bootstrap_pipeline.py

# Test with real data
python scripts/test_bootstrap_pipeline.py \
  --input-dir test_data/small_collection \
  --output-dir test_output
```

### Performance Benchmarks

```bash
# Benchmark pipeline stages
python benchmarks/bootstrap_pipeline_benchmark.py

# Memory profiling
python scripts/profile_bootstrap_memory.py
```

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch sizes in embedding and ISNE training configs
   - Use CPU-based embedders if GPU memory is insufficient

2. **Document Processing Failures**
   - Check supported formats configuration
   - Verify file permissions and accessibility

3. **ISNE Training Instability**
   - Ensure dtype compatibility (Fixed in current version)
   - Adjust learning rate and batch size
   - Monitor gradient norms

4. **Graph Construction Memory Issues**
   - Increase similarity threshold to reduce edge count
   - Limit max_edges_per_node parameter

### Debug Mode

```python
# Enable detailed logging
config.debug_mode = True
config.log_level = "DEBUG"

# Save intermediate results
config.save_intermediate = True
```

## Migration and Updates

### Model Version Compatibility

The bootstrap pipeline maintains backward compatibility for model loading:

```python
# Load legacy models
isne_model = ISNEModel.load("legacy_model.pth", version="v1.0")

# Convert to current format
isne_model.save("updated_model.pth", version="v2.0")
```

### Configuration Migration

```bash
# Migrate old configurations
python scripts/migrate_bootstrap_config.py \
  --old-config old_config.yaml \
  --new-config new_config.yaml
```

## Contributing

### Development Workflow

1. **Feature Branches**: Create branch for new features
2. **Stage Implementation**: Follow `BaseBootstrapStage` protocol
3. **Configuration**: Add config parameters to appropriate config classes
4. **Testing**: Include unit and integration tests
5. **Documentation**: Update this README and add docstrings
6. **Review**: Complete all development protocol reviews

### Adding New Stages

```python
from src.isne.bootstrap.stages.base import BaseBootstrapStage

class NewStage(BaseBootstrapStage):
    def __init__(self):
        super().__init__("new_stage")
    
    def execute(self, inputs, config):
        # Stage implementation
        pass
    
    def validate_inputs(self, inputs, config):
        # Input validation
        return []
```

## Related Components

- **PathRAG Engine**: `src/pathrag/` - Uses ISNE models for enhanced retrieval
- **Graph Enhancement**: `src/components/graph_enhancement/` - ISNE integration
- **Document Processing**: `src/components/docproc/` - Document adapter system
- **Embedding System**: `src/components/embedding/` - Base embedding generation
- **Alert System**: `src/alerts/` - Monitoring and notification integration

## References

- **ISNE Paper**: [Inductive Shallow Node Embedding](research/isne_paper.pdf)
- **PathRAG Documentation**: `src/pathrag/pathrag_readme.md`
- **Component Architecture**: `src/orchestration/README.md`
- **Configuration Guide**: `src/config/README.md`

---

**Last Updated**: 2025-06-14  
**Version**: 2.0  
**Maintainer**: HADES Development Team