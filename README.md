# HADES - Heuristic Adaptive Data Extrapolation System

Enhanced implementation of **PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths** with ArangoDB integration, **ISNE enhancement**, and **production-ready API architecture**.

## 🎯 Overview

HADES is a production-ready graph-based RAG system with advanced ISNE (Inductive Shallow Node Embedding) capabilities. The project has been professionally organized with clean API architecture, comprehensive testing, and automated production pipelines.

## ✨ Features

- 🔍 **PathRAG Implementation**: Efficient graph-based RAG with path pruning
- 🧠 **ISNE Enhancement**: Graph structure-aware embedding learning
- 📊 **ArangoDB Support**: Scalable multi-model database backend
- 🚀 **FastAPI Interface**: Production-ready API with MCP tools
- 🔄 **Automated Pipelines**: End-to-end automation via API endpoints
- 📝 **Semantic Collections**: Organized code and documentation analysis
- 🔧 **Type-Safe Implementation**: Fully type-annotated codebase
- 🧪 **Comprehensive Testing**: 85%+ test coverage with organized test suite

## 🏗️ Architecture

### Core Components

- **Document Processing**: Advanced parsing with Docling and custom adapters
- **Chunking**: Smart semantic boundary-aware chunking
- **Embedding**: ModernBERT with ISNE graph enhancement
- **Storage**: Multi-model ArangoDB with graph, document, and vector storage
- **Retrieval**: Graph-traversal semantic search with enhanced embeddings
- **API**: FastAPI with background processing and status tracking

### Project Organization

```
HADES/
├── src/                    # Core source code
│   ├── api/               # FastAPI server and endpoints
│   ├── components/        # Modular component system
│   ├── pathrag/          # PathRAG implementation
│   └── types/            # Type definitions
├── scripts/              # Organized script categories
│   ├── production/       # Production-ready scripts
│   ├── experiments/      # Testing and development
│   ├── bootstrap/        # Data ingestion scripts
│   └── archive/          # Deprecated scripts
├── test/                 # Comprehensive test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── system/          # System tests
│   └── components/      # Component-specific tests
└── docs/                # Documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/r3d91ll/HADES.git
cd HADES

# Install dependencies with Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

### Start the API Server

```bash
# Using the entry point script
python src/api/run_hades.py --host 0.0.0.0 --port 8595 --reload

# Or directly
python -m src.api.server
```

### Run Complete ISNE Pipeline

```bash
# Via API endpoint (recommended)
curl -X POST "http://localhost:8595/api/v1/production/run-complete-pipeline" \
  -d "input_dir=/path/to/isne-testdata&database_prefix=production"

# Check status
curl "http://localhost:8595/api/v1/production/status/complete_20241218_223000"
```

### Manual Pipeline Steps

```bash
# 1. Bootstrap data
python src/pipelines/production/bootstrap_full_isne_testdata.py

# 2. Train ISNE model
python src/pipelines/production/train_isne_memory_efficient.py

# 3. Apply model and create production database
python src/pipelines/production/apply_efficient_isne_model.py \
  --model-path output/isne_training_efficient/isne_model_*.pth \
  --target-db isne_production_database --create-new-db

# 4. Build semantic collections
python src/pipelines/production/build_semantic_collections.py \
  --db-name isne_production_database
```

## 📊 API Endpoints

### Production Pipeline (`/api/v1/production/`)

- `POST /bootstrap` - Bootstrap ISNE dataset into ArangoDB
- `POST /train` - Train memory-efficient ISNE model
- `POST /apply-model` - Apply trained model to create production database
- `POST /build-collections` - Build semantic collections
- `POST /run-complete-pipeline` - **Run entire pipeline end-to-end**
- `GET /status/{operation_id}` - Monitor operation progress
- `GET /status` - List all pipeline operations

### Core PathRAG (`/api/v1/`)

- `POST /write` - Add documents to knowledge graph
- `POST /query` - Query the PathRAG system
- `GET /status` - System health check
- `GET /metrics` - Prometheus metrics

### Documentation

- `/docs` - Interactive API documentation (Swagger)
- `/redoc` - Alternative API documentation

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run by category
poetry run pytest test/unit/          # Unit tests
poetry run pytest test/integration/   # Integration tests
poetry run pytest test/system/        # System tests

# Run specific components
poetry run pytest test/components/docproc/     # Document processing
poetry run pytest test/components/embedding/  # Embedding tests
```

## 🔧 MCP Tools

HADES provides MCP (Model Context Protocol) tools for external integration:

```python
from src.api.tools.production_tools import production_tools

# Run complete pipeline
result = production_tools.run_complete_pipeline("/path/to/data")

# Monitor progress
status = production_tools.wait_for_completion(result["operation_id"])
```

## 📈 Production Results

Recent successful production run:

- **91,165 enhanced embeddings** created from ISNE test dataset
- **ISNE-discovered edges** for improved graph connectivity
- **Semantic collections** built: 258 classes, 592 modules, 383 tests, 179 concepts
- **Processing time**: ~2.3 minutes for complete pipeline
- **API integration**: All operations available via REST and MCP

## 🗂️ Workspace Organization

### Before Cleanup

- 70+ scattered scripts in root directory
- Mixed production and experimental code
- No API integration
- Difficult maintenance

### After Organization

- **4 logical categories**: production, experiments, bootstrap, archive
- **FastAPI endpoints** with background processing
- **MCP tools** for external integration
- **Comprehensive testing** with 85%+ coverage
- **Clean architecture** with clear separation of concerns

## 💡 Usage Patterns

### Development

```bash
# Test experimental features
python scripts/experiments/quick_test_isne_training.py

# Validate bootstrap processes  
python scripts/bootstrap/bootstrap_micro_validation.py
```

### Production

```bash
# API-driven operations (recommended)
curl -X POST "http://localhost:8595/api/v1/production/train" \
  -H "Content-Type: application/json" \
  -d '{"database_name": "my_training_db", "epochs": 100}'

# Direct pipeline execution
python src/pipelines/production/train_isne_memory_efficient.py --epochs 100
```

## 📚 Documentation

- [**Complete Documentation Index**](./docs/README.md) - Comprehensive documentation guide
- [Scripts Organization](./scripts/README.md) - Organized script structure
- [Test Suite](./test/README.md) - Comprehensive testing documentation
- [API Integration](./src/api/README.md) - FastAPI and MCP tools
- [HADES Architecture](./docs/HADES_SERVICE_ARCHITECTURE.md) - System design
- [ISNE Production Success](./docs/analysis/ISNE_PRODUCTION_SUCCESS_REPORT.md) - **Current achievements**
- [Database Schema](./docs/schemas/hades_unified_database_schema.md) - Production schema
- [Integration Guides](./docs/integration/) - Setup and deployment guides

## 🔄 Migration Benefits

1. **Professional Organization**: Clear separation between production and experimental code
2. **API-First Architecture**: All operations available via REST endpoints
3. **Background Processing**: Long-running operations don't block API
4. **Status Tracking**: Real-time monitoring of pipeline operations
5. **MCP Integration**: External systems can programmatically interact
6. **Automated Testing**: Comprehensive test suite with coverage requirements
7. **Scalable Structure**: Easy to add new components and operations

## 🗺️ Development Roadmap

### Phase 1: Core Infrastructure ✅ COMPLETE

- ✅ **Database Creation & Management** - Core storage functionality
- ✅ **ISNE Model Training & Application** - Model creation and enhancement capabilities  
- ✅ **Critical Infrastructure Fixes** - Bootstrap imports, database connectivity, error handling
- ✅ **MCP Endpoint Functionality** - All endpoints operational and tested

### Phase 2: Authentication & Basic PathRAG (3-4 weeks)

- 🔄 **Authentication Framework** - Token-based auth for MCP endpoints
- 🔄 **Configuration Management** - Environment-aware, hierarchical configs
- 🔄 **Database Factory Pattern** - Centralized database management
- 🔄 **Basic PathRAG Search** - Directory structure-based graph with organic growth
- 🔄 **Classification Framework** - Syntactically correct placeholders for future research

### Phase 3: Enhanced Storage & Complete PathRAG (3-4 weeks)

- 🔮 **Enhanced Storage Manager** - ISNE-enhanced data storage and semantic collections
- 🔮 **Complete PathRAG Implementation** - Multi-mode search (naive, local, global, hybrid)
- 🔮 **Graph Enhancement System** - Organic edge discovery and graph evolution
- 🔮 **Performance Optimization** - Search and storage performance tuning

### Phase 4: Production Deployment (2-3 weeks)

- 🔮 **Agent Management Foundation** - Basic agent lifecycle and monitoring
- 🔮 **Full Automation Pipelines** - End-to-end workflow automation
- 🔮 **Production Hardening** - Security, monitoring, and deployment optimization
- 🔮 **Evaluation Framework** - Comprehensive RAG system assessment

### Phase 4.5: Evaluation & Decision Point

- 🔍 **RAG System Evaluation** - Assess directory graph effectiveness using `benchmark/rag/`
- 🔍 **User Satisfaction Analysis** - Measure cross-domain discovery and retrieval quality
- 🔍 **Phase 5 Decision** - Data-driven decision on advanced classification research need

### Phase 5: Advanced Classification Research (8-10 weeks, CONDITIONAL)

- 🔬 **Annif-Bootstrapped Training Data** - Generate 100k+ high-quality classifications
- 🔬 **Custom 7B Model Training** - Fine-tune for interdisciplinary classification
- 🔬 **Bridge Keyword System** - Abstract concepts for cross-domain connections  
- 🔬 **Integration & Evaluation** - Compare against simple approaches with rigorous metrics

### Priority Items

1. **Basic Authentication Infrastructure** - Simple token-based auth for MCP endpoints to prepare for future agent/RBAC implementation without major architectural changes
2. **Pipeline Naming Consistency** - Rename `run_pipeline` to `bootstrap_isne_pipeline` for clarity
3. **Storage Module Integration** - Complete the database → ISNE → storage workflow

## 🎯 Production Ready

HADES is now production-ready with:

- ✅ **Organized codebase** with clear architecture
- ✅ **API endpoints** for all major operations
- ✅ **Background processing** with status tracking
- ✅ **MCP tools** for external integration
- ✅ **Comprehensive testing** with 85%+ coverage
- ✅ **ISNE enhancement** for graph-aware embeddings
- ✅ **Semantic collections** for advanced RAG queries
- ✅ **Documentation** and migration guides

## 📄 Citation

```bibtex
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```
