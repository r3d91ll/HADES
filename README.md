# HADES

Heuristic Adaptive Data Extrapolation System - Unified Architecture with Jina v4

## Overview

HADES is a complete reimplementation of the HADES system using Jina v4's multimodal embeddings as the core processing engine. This dramatically simplifies the architecture from 6+ components to just 2 main components while preserving all core innovations.

## Key Features

- **Unified Processing**: Single pipeline for all document types using Jina v4
- **Multimodal Support**: Text, code, images in one embedding space
- **Code Intelligence**: AST analysis for Python files with symbol extraction and dependency tracking
- **Theory-Practice Bridges**: Intelligent keyword extraction connecting research to implementation
- **Hierarchical Organization**: Filesystem structure as semantic information
- **Supra-Weights**: Multi-dimensional relationship modeling
- **Query-as-Graph**: Treats queries as temporary nodes in the knowledge graph

## Architecture

```
Input Documents → Jina v4 Processor → ISNE Enhancement → ArangoDB Storage
                        ↓
                  PathRAG Retrieval ← Query Processing
```

### Components

1. **Jina v4 Processor** (`/src/jina_v4/`)
   - Multimodal document processing
   - Late chunking after embedding
   - Attention-based keyword extraction
   - Hierarchical JSON structure generation

2. **ISNE Enhancement** (`/src/isne/`)
   - Graph-aware embedding enhancement
   - Directory-aware training
   - Supra-weight bootstrap initialization

3. **PathRAG** (`/src/pathrag/`)
   - Path-based retrieval
   - Multi-dimensional weight calculations
   - Query-as-graph implementation

4. **Storage** (`/src/storage/`)
   - ArangoDB for graph storage
   - Supra-weight persistence
   - Hierarchical structure preservation

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd HADES

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Jina v4 embeddings
pip install git+https://github.com/jinaai/jina-embeddings-v4
```

## Configuration

Edit configurations in `/config/`:
- `jina_v4/config.yaml` - Jina v4 settings
- `isne/bootstrap_config.yaml` - ISNE bootstrap settings
- `storage/arango_config.yaml` - ArangoDB connection

## Usage

### Start the API Server

```bash
python run_server.py
```

The API will be available at:
- http://localhost:8000 - API root
- http://localhost:8000/docs - Swagger UI documentation
- http://localhost:8000/redoc - ReDoc documentation

### Process Documents

```python
import requests

# Process a directory
response = requests.post("http://localhost:8000/process/directory", json={
    "directory_path": "/path/to/documents",
    "recursive": True,
    "file_patterns": ["*.md", "*.py", "*.pdf"]
})

# Process a single file
response = requests.post("http://localhost:8000/process/file", json={
    "file_path": "/path/to/document.pdf"
})

# Process raw text
response = requests.post("http://localhost:8000/process/text", json={
    "text": "Your text content here",
    "source": "manual_input"
})
```

### Query the Knowledge Graph

```python
# Query
response = requests.post("http://localhost:8000/query", json={
    "query": "How does ISNE enhance embeddings?",
    "max_results": 10,
    "include_embeddings": False
})
```

## Implementation Status

### Completed ✓
- Core architecture and structure
- ISNE implementation
- PathRAG components
- Storage layer
- API framework
- Configuration system
- Documentation
- Metadata system (.hades directories)

### TODO
- Implement Jina v4 placeholder methods:
  - Document parsing with Docling
  - Local vLLM embedding extraction
  - Attention-based keyword extraction
  - Semantic operations
  - Image processing
- Complete PathRAG integration in API
- Add bootstrap scripts
- Create validation suite

## Development

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy src/
```

### Code Formatting

```bash
black src/
isort src/
```

## Metadata System

Every directory contains a `.hades/` folder with:
- `metadata.json` - Directory purpose and contents
- `relationships.json` - Internal and external relationships
- `keywords.json` - Technical and domain keywords

This metadata helps Jina v4 understand the codebase structure and create theory-practice bridges.

## Key Innovations

1. **Unified Processing**: Replace complex pipelines with single Jina v4 processor
2. **Late Chunking**: Chunk after embedding to preserve context
3. **Theory-Practice Bridges**: Keywords connect research papers to code
4. **Filesystem as Taxonomy**: Directory structure provides semantic information
5. **Supra-Weights**: Multi-dimensional edge weights for nuanced relationships

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

Apache License 2.0 - See LICENSE file for details