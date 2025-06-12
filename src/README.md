# HADES Source Code

Welcome to the HADES (Hierarchical Adaptive Data Extrapolation System with Path-based Retrieval Augmented Generation) source code. This is an advanced RAG system that combines graph-based knowledge representation with ISNE (Inductive Shallow Node Embedding) within a multi-modal ArangoDB for enhanced document understanding and retrieval.

## 🎯 System Overview

HADES is designed to process, understand, and enable intelligent querying of large document corpora, with special emphasis on:

- **Python code repositories** with AST-based analysis
- **Research papers and technical documentation**
- **Multi-format document processing** (PDF, Markdown, JSON, YAML, etc.)
- **Graph-based relationship modeling** between document chunks
- **Inductive embeddings** that generalize to new content

## 🏗️ Architecture

The system follows a modular, pipeline-based architecture with clear separation of concerns:

```text
Document Input → Processing → Chunking → Embedding → Graph Construction → Storage → Retrieval
```

### Core Components

1. **Document Processing** (`docproc/`) - Multi-format document parsing and extraction
2. **Chunking** (`chunking/`) - Semantic segmentation of documents
3. **Embedding** (`embedding/`) - Vector representation generation
4. **ISNE** (`isne/`) - Graph-aware embedding enhancement
5. **Storage** (`storage/`) - Graph and vector database management
6. **Orchestration** (`orchestration/`) - Pipeline coordination and execution
7. **PathRAG** (`pathrag/`) - Query processing and retrieval

## 📁 Module Structure

### Core Processing Modules

- **[`alerts/`](alerts/README.md)** - System monitoring and alerting infrastructure
- **[`api/`](api/README.md)** - REST API and CLI interfaces for system interaction
- **[`chunking/`](chunking/README.md)** - Document segmentation with text and code-specific strategies
- **[`config/`](config/README.md)** - Centralized configuration management
- **[`database/`](database/README.md)** - ArangoDB integration and graph database operations
- **[`docproc/`](docproc/README.md)** - Multi-format document processing and content extraction
- **[`embedding/`](embedding/README.md)** - Vector embedding generation with multiple backend support

### Advanced Processing

- **[`isne/`](isne/README.md)** - Inductive Shallow Node Embedding system for graph-aware embeddings
- **[`model_engine/`](model_engine/README.md)** - Model serving infrastructure (vLLM, Haystack)
- **[`orchestration/`](orchestration/README.md)** - Pipeline orchestration and parallel processing
- **[`pathrag/`](pathrag/README.md)** - Path-based retrieval augmented generation engine

### Infrastructure & Support

- **[`cli/`](cli/README.md)** - Command-line interfaces for administration and operations
- **[`schemas/`](schemas/README.md)** - Pydantic schemas and data validation
- **[`storage/`](storage/README.md)** - Database abstraction and storage interfaces
- **[`types/`](types/README.md)** - Type definitions and data structures
- **[`utils/`](utils/README.md)** - Shared utilities and helper functions
- **[`validation/`](validation/README.md)** - Data quality validation and testing

## 🚀 Key Features

### Document Processing

- **Multi-format support**: PDF, Markdown, Python, JSON, YAML, and more
- **AST-based code analysis**: Preserves code structure and semantics
- **Metadata extraction**: Rich document and chunk metadata
- **Format detection**: Automatic content type identification

### Graph-Based Understanding

- **Semantic relationships**: Automatic discovery of document interconnections
- **Cross-document linking**: Relationships spanning multiple documents
- **Graph storage**: Efficient ArangoDB-based graph persistence
- **Path-based retrieval**: Query paths through document relationships

### ISNE Embeddings

- **Graph-aware embeddings**: Leverage document relationships for better representations
- **Inductive capabilities**: Handle new documents without retraining
- **Hierarchical training**: Daily incremental + weekly/monthly full retraining
- **Quality monitoring**: Automatic embedding quality assessment

### Production Ready

- **Scalable architecture**: Handle large document corpora
- **Monitoring & alerts**: Comprehensive system health monitoring
- **Configuration management**: YAML-based configuration system
- **API interfaces**: REST API and CLI for all operations

## 🛠️ Development Workflow

### Environment Setup

```bash
# Install dependencies
poetry install

# Set up development environment
./setup_dev_env.sh --full
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src --cov-report=term

# Run specific modules
poetry run pytest src/chunking/ -v
```

### Code Quality

```bash
# Type checking
poetry run mypy src/

# Code formatting
poetry run black src/
poetry run isort src/

# Linting
poetry run flake8 src/
```

## 📈 Usage Patterns

### 1. Bootstrap New System

```bash
# Initialize ISNE with document corpus
python -m src.isne.bootstrap --corpus-dir ./docs --output-dir ./models
```

### 2. Process Documents

```bash
# Run end-to-end pipeline
python scripts/run_end_to_end_test.py -i ./input -o ./output

# Process with specific configuration
python scripts/run_data_ingestion_with_config.py
```

### 3. Query System

```bash
# Start API server
python -m src.api.server

# Query via CLI
python -m src.api.cli query "How does chunking work?"
```

### 4. Monitor System

```bash
# Check training status
python scripts/check_training_status.py

# Set up automated training
./scripts/setup_training_cron.sh
```

## 🔧 Configuration

The system uses YAML-based configuration files in `src/config/`:

- `database_config.yaml` - ArangoDB connection settings
- `embedding_config.yaml` - Embedding model configuration
- `chunker_config.yaml` - Document chunking parameters
- `isne_config.yaml` - ISNE training settings
- `scheduled_training_config.yaml` - Automated training schedules

## 📊 Data Flow

```text
1. Documents → DocProc → Structured Content
2. Structured Content → Chunking → Semantic Segments  
3. Segments → Embedding → Vector Representations
4. Vectors + Graph → ISNE → Enhanced Embeddings
5. Enhanced Embeddings → Storage → ArangoDB
6. Queries → PathRAG → Intelligent Retrieval
```

## 🎛️ Extension Points

The system is designed for extensibility:

- **Document Adapters**: Add new document format support in `docproc/adapters/`
- **Chunking Strategies**: Implement new chunking algorithms in `chunking/`
- **Embedding Backends**: Add new embedding models in `embedding/adapters/`
- **Storage Backends**: Implement new storage systems in `storage/`

## 🔍 Research Context

This system implements and extends several research concepts:

- **PathRAG**: Path-based retrieval for enhanced context understanding
- **ISNE**: Inductive embeddings that capture graph structure
- **Hierarchical Processing**: Multi-level document understanding
- **Adaptive Training**: Intelligent model updates for evolving corpora

## 📚 Documentation

Each module contains detailed documentation:

- Technical specifications in module README files
- API documentation in docstrings
- Configuration examples in YAML files
- Usage examples in script files

For comprehensive bootstrap and training documentation, see [`isne/bootstrap/isne_bootstrap.md`](isne/bootstrap/isne_bootstrap.md).

## 🤝 Contributing

This codebase uses itself recursively - HADES processes its own source code to improve understanding of Python codebases and related research documentation. This creates a powerful feedback loop for development and improvement.

When contributing:

1. Follow existing code patterns and conventions
2. Add comprehensive tests for new functionality
3. Update documentation for any API changes
4. Run full test suite and type checking before committing
