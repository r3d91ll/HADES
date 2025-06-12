# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Protocols

### Standard Protocol for Creating New Modules

When creating a new module:

1. **Create a moduleName_readme.md file** in the root of the module directory
2. **Track all documentation** in this file for that module

### Standard Protocol for Code and Documentation Review

Before declaring a module complete, perform the following reviews:

#### Testing Review (Testing Standard)

- Unit tests all functions to a minimum **85% coverage**
- Unit tests for all new functions
- Integration tests for module interaction
- Performance benchmarks for critical operations

#### Code Review (Code Quality Checklist)

- Run `mypy` to check for type errors
- Ensure unit test coverage ≥ **85%**
- Verify all functions have docstrings
- Remove debugging print statements
- Check for TODOs and resolve or document them

#### Documentation Review (Documentation Requirements)

- Module-level docstring explaining purpose and usage
- Function/class docstrings with parameters, return types, and examples
- Update README or relevant documentation files
- Add usage examples for new functionality

#### Organization Review (Organization Process)

- Move unused/deprecated code to `dead-code/` directory
- Consolidate duplicated functionality
- Ensure proper imports (no circular dependencies)

#### Benchmark Review (Performance Standard)

- Performance benchmarks for critical operations
- Ensure benchmarks are updated for new functionality

### Standard Protocol for Directory Administration

- **Test files** belong in the `test/` directory
- **Benchmark files** belong in the `benchmark/` directory
- **Utility scripts** not directly related to a module or code within the `src/` directory belong in the `scripts/` directory

### Development Workflow Requirements

#### USE GIT

- Use git to track changes
- Use git to commit changes
- Use git to push changes
- Use git to pull changes
- Use git to merge changes
- **Use feature branches**

#### USE Tree-sitter MCP Server

- Use tree-sitter MCP server to parse code
- Use tree-sitter MCP server to generate documentation
- Use tree-sitter MCP server to analyze code while troubleshooting
- Use tree-sitter MCP server to generate code
- **Ensure you clear the tree-sitter MCP server cache** after making changes to the code

#### USE Test and Utility Directories

- Use the `test/` directory to store test files
- Use the `temp-utils/` directory to store temporary utility scripts

#### TaskManager MCP Server

- **Make no change without a task**
- All work MUST be documented as a legitimate task in TaskManager
- Making any change in this workspace REQUIRES that the work be documented and prioritized in the TaskManager MCP server

### Development Philosophy

**This project is in Development:**

- NO backwards compatibility required
- NO migration scripts
- **Break fast, fix fast**

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

HADES is an enhanced implementation of PathRAG (Path-based Retrieval Augmented Generation) that combines graph-based knowledge representation with advanced language models. The system is transitioning to a **config-driven modular architecture** to enable A/B testing, component swapping, and experimental workflows.

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

### Modular Architecture Design

The system is being refactored into a config-driven modular architecture located in `src/orchestration/`:

```text
src/orchestration/
├── components/                      # Shared, reusable components
│   ├── base/                       # Base protocols and utilities
│   ├── document_processing/        # Document processing components
│   ├── chunking/                   # Chunking components
│   ├── embedding/                  # Embedding components
│   ├── graph_enhancement/          # Graph enhancement components
│   └── storage/                    # Storage components
├── pipelines/                      # Pipeline implementations
├── config/                         # Configuration system
└── utils/                          # Orchestration utilities
```

#### Key Features of Modular Architecture

- **Component Registry System**: Dynamic component loading and registration
- **Protocol-Based Design**: Type-safe interfaces using Python protocols
- **Configuration-Driven**: YAML-based configuration for component selection
- **A/B Testing Support**: Built-in support for experimental workflows
- **Pipeline Abstraction**: Standardized pipeline execution framework

### Data Flow

```text
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

### Pipeline Configuration Example

```yaml
# Example: data_ingestion_vllm_isne.yaml
pipeline:
  name: "data_ingestion_vllm_isne"
  description: "Data ingestion with VLLM embeddings and ISNE enhancement"
  type: "data_ingestion"
  version: "1.0"
  
components:
  document_processing:
    type: "docling"
    config_file: "components/docling_config.yaml"
    params:
      batch_size: 10
      timeout: 300
      
  chunking:
    type: "chonky"
    config_file: "components/chonky_config.yaml"
    params:
      chunk_size: 512
      overlap: 50
      
  embedding:
    type: "vllm"
    config_file: "components/vllm_config.yaml"
    params:
      model_name: "BAAI/bge-large-en-v1.5"
      batch_size: 32

# A/B Testing Configuration
ab_testing:
  enabled: true
  experiment_id: "embedding_comparison_001"
  variants:
    - name: "vllm_variant"
      weight: 50
      description: "VLLM embeddings"
      overrides:
        embedding:
          type: "vllm"
    - name: "cpu_variant"
      weight: 50
      description: "CPU embeddings"
      overrides:
        embedding:
          type: "cpu"
```

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

## Pipeline Execution Examples

### Running a Pipeline with Configuration

```bash
# Run data ingestion with VLLM + ISNE
python -m src.orchestration.pipelines.data_ingestion \
  --config config/pipelines/data_ingestion_vllm_isne.yaml \
  --input-dir /path/to/docs \
  --output-dir /path/to/output

# Run A/B test
python -m src.orchestration.utils.ab_testing \
  --config config/pipelines/ab_test_embedding_comparison.yaml \
  --input-dir /path/to/docs \
  --variants vllm_variant,cpu_variant
```

### Swapping Components via Configuration

```yaml
# Switch from ISNE to SageGraph
graph_enhancement:
  type: "sagegraph"  # Changed from "isne"
  config_file: "components/sagegraph_config.yaml"
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

### Coverage Requirements

- **Minimum 85% unit test coverage** for all modules
- All new functions must have unit tests
- Integration tests for module interactions
- Performance benchmarks for critical operations

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
2. **Task Management**: Document all work in TaskManager MCP server
3. **Feature Branches**: Use git feature branches for all development
4. **Type Checking**: Run `poetry run mypy src/` before commits
5. **Testing**: Use `poetry run pytest` with appropriate markers
6. **Coverage**: Ensure ≥85% test coverage for all new code
7. **Code Quality**: Black formatting and isort import sorting required
8. **Documentation**: Update module README and docstrings
9. **Tree-sitter**: Clear MCP server cache after code changes
10. **Review Process**: Complete all review protocols before declaring modules complete

## Component Protocol Interfaces

The modular architecture uses protocol-based interfaces for type safety:

```python
# Base component protocols
class ComponentProtocol(Protocol):
    def configure(self, config: Dict[str, Any]) -> None: ...
    def validate_config(self, config: Dict[str, Any]) -> bool: ...

# Specific component protocols
class DocumentProcessorProtocol(ComponentProtocol):
    def process_documents(self, file_paths: List[str], **kwargs) -> List[Dict[str, Any]]: ...

class ChunkerProtocol(ComponentProtocol):
    def chunk_documents(self, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]: ...

class EmbedderProtocol(ComponentProtocol):
    def generate_embeddings(self, chunks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]: ...
```

This modular architecture enables flexible experimentation, A/B testing, and component swapping through configuration while maintaining type safety and performance standards.
