# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Set up development environment
./setup_dev_env.sh

# Full setup with tests and type checking
./setup_dev_env.sh --full
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src --cov-report=term

# Run specific test modules
poetry run pytest test/sampler/
poetry run pytest tests/unit/embedding/
poetry run pytest tests/docproc/adapters/

# Run individual test files
python -m pytest tests/unit/isne/training/test_neighbor_sampler.py -v
```

### Type Checking and Code Quality
```bash
# Run mypy type checking
poetry run mypy src/

# Check syntax in specific files
python scripts/check_syntax.py <file_path>

# Format code with black
poetry run black src/

# Sort imports with isort
poetry run isort src/

# Run flake8 linting
poetry run flake8 src/
```

### Pipeline Execution
```bash
# End-to-end pipeline test
python scripts/run_end_to_end_test.py -i <input_dir> -o <output_dir>

# ISNE pipeline with alerts
python scripts/run_isne_pipeline_with_alerts.py

# Validate ISNE embeddings
python scripts/validate_isne_embeddings.py
```

### API and Services
```bash
# Start the FastAPI server
python -m src.api.server

# Run CLI interface
python -m src.api.cli
```

## Architecture Overview

HADES-PathRAG is an enhanced implementation of PathRAG (Path-based Retrieval Augmented Generation) that combines graph-based knowledge representation with advanced language models.

### Core System Components

#### 1. PathRAG Engine (`src/pathrag/`)
- **PathRAG Class**: Central orchestrator managing storage, embeddings, and retrieval
- **Storage Abstractions**: `BaseGraphStorage`, `BaseKVStorage`, `BaseVectorStorage`
- **Query Interface**: `QueryParam` with hybrid retrieval modes (naive, local, global, hybrid)
- **Multi-backend Support**: NetworkX, ArangoDB, nano-vectordb

#### 2. ISNE (Inductive Shallow Node Embedding) (`src/isne/`)
- **Purpose**: Graph structure-aware embedding enhancement
- **ISNEModel**: Multi-layer graph neural network with attention mechanisms
- **Training**: Combines feature, structural, and contrastive losses
- **Pipeline Integration**: `ISNEPipeline` with validation and real-time alerting

#### 3. Document Processing (`src/docproc/`)
- **Adapter Pattern**: Format-specific processors (Python, Markdown, JSON, YAML, Docling)
- **Manager**: `DocumentProcessorManager` for batch processing and caching
- **Validation**: Pydantic-based schema validation throughout
- **Format Detection**: Automatic content type detection and category classification

#### 4. Chunking System (`src/chunking/`)
- **Text Chunkers**: CPU-based and "Chonky" (GPU-accelerated) chunkers
- **Code Chunkers**: AST-based chunkers preserving code structure
- **Registry Pattern**: Dynamic chunker selection based on content type

#### 5. Embedding System (`src/embedding/`)
- **Multiple Backends**: CPU, vLLM, ModernBERT adapters
- **Batch Processing**: Efficient batch embedding generation
- **Registry System**: Dynamic adapter registration and selection

### Data Flow

```
Document Input → Document Processing (docproc) → Chunking → 
Base Embedding (embedding) → ISNE Enhancement → 
Graph Construction & Storage (pathrag) → Query Processing & Retrieval
```

### Key Design Patterns

1. **Adapter Pattern**: Used throughout for document processing, embedding, and storage
2. **Registry Pattern**: Dynamic component selection (chunkers, adapters, embeddings)
3. **Pipeline Pattern**: Modular stages with standardized interfaces and error isolation
4. **Protocol-Based Design**: Type-safe interfaces using Python protocols

## Configuration Management

### Configuration Files Location: `src/config/`
- **chunker_config.yaml**: Chunking strategy configuration
- **embedding_config.yaml**: Embedding model and adapter settings
- **database_config.yaml**: ArangoDB and storage configuration  
- **isne_config.yaml**: ISNE training and model parameters
- **vllm_config.yaml**: vLLM server and inference settings

### Loading Configuration
```python
from src.config.config_loader import load_config

# Load specific configuration
chunker_config = load_config("chunker_config")
embedding_config = load_config("embedding_config")
```

## Database Integration

### ArangoDB Setup
- **Connection**: `src/database/arango_client.py` handles connection pooling and retry logic
- **Repository Pattern**: Abstract interfaces in `src/storage/interfaces/`
- **Graph Operations**: Full CRUD operations for documents, chunks, and relationships

### Storage Namespaces
```python
# Different storage backends are available:
from src.pathrag.storage import NetworkXStorage, ArangoStorage
from src.pathrag.base import QueryParam

# Query modes: "naive", "local", "global", "hybrid"
result = rag.query(question, param=QueryParam(mode="hybrid"))
```

## Monitoring and Alerts

### Alert System (`src/alerts/`)
```python
from src.alerts import AlertManager, AlertLevel

alert_manager = AlertManager(alert_dir="./alerts")
alert_manager.alert(
    message="Processing error occurred",
    level=AlertLevel.HIGH,
    source="pipeline_stage"
)
```

### Validation (`src/validation/`)
- **Embedding Validator**: Validates embedding consistency and quality
- **Pipeline Validation**: Multi-stage validation with error recovery

## Testing Guidelines

### Test Structure
- **Unit Tests**: `test/` directory with marker-based organization
- **Integration Tests**: `scripts/run_end_to_end_test.py`
- **Test Markers**: `unit`, `integration`, `slow`, `models`, `parsers`, `adapters`, `storage`

### Test Execution
```bash
# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m "adapters and not slow"
```

## Type System

This codebase uses comprehensive type annotations with strict mypy configuration:

### Type Checking Configuration
- **mypy.ini**: Strict type checking with specific module overrides
- **Protocols**: Extensive use of protocols for interface definitions
- **Generic Types**: Complex generic types for pipeline stages and adapters

### Important Type Locations
- **Common Types**: `src/types/common.py`
- **DocProc Types**: `src/types/docproc/`
- **ISNE Types**: `src/types/isne/`
- **Pipeline Types**: `src/types/pipeline/`

## Performance Considerations

### GPU Support
- **ISNE Training**: GPU acceleration available with proper PyTorch setup
- **vLLM Integration**: GPU inference for high-throughput embedding and completion
- **Chonky Chunker**: GPU-accelerated chunking for large documents

### Batch Processing
- **Embedding Batching**: `src/embedding/batch.py` for efficient batch processing
- **Document Batching**: `src/tools/batching/file_batcher.py`

### Caching
- **Adapter Caching**: Document processing adapters cache results
- **Embedding Caching**: Embedding adapters support caching layers

## Development Workflow

1. **Environment**: Use Poetry for dependency management (`poetry install`)
2. **Type Checking**: Run `poetry run mypy src/` before commits
3. **Testing**: Use `poetry run pytest` with appropriate markers
4. **Code Quality**: Black formatting and isort import sorting required
5. **Branch Strategy**: Feature branches with descriptive names