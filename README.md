# HADES - Heterogeneous Adaptive Dimensional Embedding System

Production infrastructure for processing, storing, and serving embedded knowledge from multiple data sources.

## Overview

HADES is a scalable data infrastructure system designed to:

- Process and embed large-scale document collections (ArXiv, documentation, codebases)
- Store embeddings and metadata in ArangoDB
- Serve data through an MCP (Model Context Protocol) interface
- Support semantic search and retrieval across heterogeneous data sources

## Architecture

```
HADES/
├── processors/           # Data ingestion pipelines
│   ├── arxiv/           # ArXiv paper processing
│   ├── documentation/   # Documentation site processing
│   ├── codebase/        # Source code processing
│   └── pdf/             # Generic PDF processing
├── storage/             # Database layer
│   ├── schemas/         # ArangoDB collection schemas
│   └── managers/        # Database connection managers
├── embeddings/          # Embedding generation
│   └── models/          # Model configurations
├── mcp_server/          # MCP server interface
└── utils/               # Shared utilities
```

## Features

### Current

- **ArXiv Processing**: Full pipeline for processing ArXiv papers with GPU-accelerated embeddings
- **ArangoDB Storage**: Optimized document and graph database storage
- **Jina Embeddings**: State-of-the-art embedding generation with Jina v3

### Planned

- **PDF Management**: Download, store, and embed full PDF content
- **Documentation Ingestion**: Scrape and embed technical documentation
- **Codebase Processing**: AST-aware code embedding with dependency tracking
- **MCP Server**: Serve processed data through standardized MCP interface
- **Semantic Search**: Cross-collection semantic search capabilities

## Installation

```bash
# Clone the repository
git clone git@github.com:r3d91ll/HADES.git
cd HADES

# Install dependencies
pip install -r requirements.txt

# Configure ArangoDB connection
export ARANGO_HOST="your-host"
export ARANGO_PASSWORD="your-password"
```

## Usage

### Process ArXiv Papers

```python
from processors.arxiv.process_arxiv_production import ProductionProcessor, ProcessingConfig

config = ProcessingConfig(
    db_name="academy_store",
    collection_name="base_arxiv",
    gpu_batch_size=1024
)

processor = ProductionProcessor(config)
processor.run()
```

### MCP Server (Coming Soon)

```bash
# Start the MCP server
python -m mcp_server.server

# Connect from Claude or other MCP clients
mcp connect hades://localhost:8080
```

## Development

This repository contains production infrastructure code. Experimental features and research code should be developed in the [reconstructionism](https://github.com/r3d91ll/reconstructionism) repository and merged here when stable.

## License

MIT License - See LICENSE file for details
