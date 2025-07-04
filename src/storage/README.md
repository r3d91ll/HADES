# Storage Layer

## Overview
This directory contains the storage implementations for HADES, providing unified interfaces for graph, vector, and document storage.

## Contents
- `arango_client.py` - ArangoDB connection and operations
- `arango_supra_storage.py` - Supra-weight storage implementation
- `arangodb/` - ArangoDB-specific implementations

## Related Resources
- **Configuration**: `/src/config/storage/`
- **Database Schema**: `/docs/architecture/database_schema.md`
- **PathRAG Integration**: `/src/pathrag/` (uses storage for graph operations)
- **ISNE Storage**: `/src/isne/` (stores enhanced embeddings)
- **API Layer**: `/src/api/` (exposes storage operations)

## Storage Types
1. **Document Storage**: Full documents with metadata
2. **Chunk Storage**: Document chunks with embeddings
3. **Vector Storage**: Embedding vectors for similarity search
4. **Graph Storage**: Nodes and edges with supra-weights

## Key Features
- **Unified Schema**: Single database for all data types
- **Supra-weights**: Multi-dimensional edge weights
- **Filesystem Awareness**: Preserves directory relationships
- **Query Optimization**: Indexes for fast retrieval

## ArangoDB Collections
- `documents` - Full documents with hierarchy
- `chunks` - Document chunks with embeddings
- `nodes` - Graph nodes (documents, chunks, queries)
- `supra_edges` - Multi-dimensional weighted edges

## Dependencies
- python-arango for database operations
- NumPy for vector operations
- NetworkX for graph utilities