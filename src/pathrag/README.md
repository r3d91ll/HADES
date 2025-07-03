# PathRAG Implementation

## Overview
This directory contains the PathRAG (Path-based Retrieval Augmented Generation) implementation, which combines vector search with graph traversal for enhanced retrieval.

## Contents
- `PathRAG_paper.pdf` - Original research paper describing the PathRAG algorithm
- `pathrag_rag_strategy.py` - Core PathRAG implementation
- `pathrag_graph_engine.py` - Graph traversal engine
- `supra_weight_calculator.py` - Multi-dimensional relationship weight calculator

## Related Resources
- **User Documentation**: `/docs/concepts/CORE_CONCEPTS.md#queries-as-graph-objects`
- **Architecture Docs**: `/docs/architecture/ARCHITECTURE.md#pathrag-query-processor`
- **API Reference**: `/docs/api/pathrag_api.md`
- **Configuration**: `/config/pathrag_config.yaml`
- **Tests**: `/tests/unit/pathrag/`
- **Examples**: `/scripts/examples/pathrag_query_example.py`

## Key Concepts
- **Supra-weights**: Multi-dimensional edge weights capturing different relationship types
- **Query-as-graph**: Queries become temporary nodes in the knowledge graph
- **Path scoring**: Combines vector similarity with graph distance

## Dependencies
- ArangoDB for graph storage
- ISNE for embedding enhancement
- NumPy for numerical operations

## Research Context
The PathRAG paper introduces a novel approach to retrieval that leverages both semantic similarity and structural relationships in knowledge graphs. This implementation extends the original work with supra-weights and filesystem-aware enhancements.