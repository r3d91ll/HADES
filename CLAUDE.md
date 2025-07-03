# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Overview

HADES is a unified RAG (Retrieval-Augmented Generation) system that uses Jina v4's multimodal embeddings as the core processing engine. The system reduces complexity from 6+ components to just 2 main components while preserving all core innovations.

## Architecture

The system follows a streamlined pipeline:
```
Input Documents → Jina v4 Processor → ISNE Enhancement → ArangoDB Storage → PathRAG Retrieval
```

### Core Components

1. **Jina v4 Processor** (`src/jina_v4/`)
   - Unified multimodal document processing
   - Late chunking after embedding
   - Attention-based keyword extraction
   
2. **ISNE (Inductive Shallow Node Embedding)** (`src/isne/`)
   - Graph-aware embedding enhancement
   - Directory-aware training
   - Supra-weight bootstrap initialization

3. **PathRAG** (`src/pathrag/`)
   - Path-based retrieval with multi-dimensional weights
   - Query-as-graph implementation
   
4. **Storage** (`src/storage/`)
   - ArangoDB for graph storage
   - Hierarchical structure preservation

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Jina v4 embeddings (required separately)
pip install git+https://github.com/jinaai/jina-embeddings-v4
```

### Running the Server
```bash
python run_server.py                    # Start FastAPI server (port 8000)
```

### Bootstrap Documents
```bash
# Process a directory of documents
python scripts/bootstrap_jinav4.py /path/to/documents --output-dir output

# With custom config
python scripts/bootstrap_jinav4.py /path/to/documents --config config/bootstrap.yaml

# Skip ISNE training
python scripts/bootstrap_jinav4.py /path/to/documents --no-train
```

### Development Tools
```bash
# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Run tests (when implemented)
pytest tests/
pytest tests/unit -v           # Unit tests only
pytest tests/integration -v    # Integration tests only
```

### API Endpoints
- `http://localhost:8000` - API root
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc documentation

Key endpoints:
- `POST /process/directory` - Process directory of documents
- `POST /process/file` - Process single file
- `POST /process/text` - Process raw text
- `POST /query` - Query the knowledge graph

## Key Design Patterns

### Protocol-Based Components
All components implement protocols defined in `src/types/components/protocols.py`. When adding new components:
1. Define the protocol interface
2. Implement the component following the protocol
3. Register with appropriate factory

### Configuration System
YAML-based configuration with environment overrides:
- `config/jina_v4/config.yaml` - Jina v4 settings
- `config/isne/` - ISNE configurations
- `config/storage/` - Storage backend settings
- `config/pathrag/` - PathRAG parameters

### Metadata System
Every directory contains `.hades/` with:
- `metadata.json` - Directory purpose and contents
- `relationships.json` - Internal/external relationships
- `keywords.json` - Technical and domain keywords

## Implementation Notes

### Placeholder Methods
Several methods in `src/jina_v4/jina_processor.py` are marked with ⚠️ as placeholders:
- Document parsing with Docling
- Local vLLM embedding extraction
- Attention-based keyword extraction
- Semantic operations
- Image processing

These need implementation to complete the system.

### Key Principles
1. **Unified Processing**: Single Jina v4 model handles all document types
2. **Graph-First Design**: Everything is a node in the knowledge graph
3. **Filesystem Awareness**: Directory structure used as semantic information
4. **Multimodal Native**: Images and text in same embedding space
5. **Late Chunking**: Chunk after embedding to preserve context

### Storage Considerations
- ArangoDB is the primary storage backend
- Document keys are derived from file paths (slashes replaced with underscores)
- Embeddings stored alongside graph relationships
- Supra-weights persist multi-dimensional edge information

## Testing Approach

Currently no tests implemented. When adding tests:
1. Unit tests in `tests/unit/` for individual components
2. Integration tests in `tests/integration/` for pipeline testing
3. Mock external services (ArangoDB, vLLM) in unit tests
4. Use real services for integration tests

## Common Tasks

### Adding a New Document Type
1. Update `JinaV4Processor.parse_document()` method
2. Add appropriate parser (e.g., using Docling)
3. Update file extension lists in bootstrap script
4. Add tests for the new format

### Modifying PathRAG Behavior
1. Update `PathRAG` class in `src/pathrag/PathRAG.py`
2. Adjust weight calculations in `calculate_path_weights()`
3. Update configuration schema if needed
4. Test with various query patterns

### Enhancing ISNE
1. Modify training pipeline in `src/isne/training/pipeline.py`
2. Update bootstrap initialization if changing dimensions
3. Ensure compatibility with Jina v4 embeddings
4. Validate enhancement improves retrieval quality