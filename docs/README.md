# HADES Documentation

## Overview
This directory contains all documentation for HADES, organized by type and purpose.

## Structure

### `/concepts/`
Core concepts and theoretical foundations:
- `CORE_CONCEPTS.md` - Theory-practice bridges, query-as-graph, filesystem taxonomy
- `JSON_STRUCTURE_MOCKUP.md` - Hierarchical data structure design
- `KEYWORD_IMPLEMENTATION_PLAN.md` - Dual-level keyword strategy
- `FILESYSTEM_METADATA_DESIGN.md` - Metadata-driven organization

### `/architecture/`
System design and technical architecture:
- `ARCHITECTURE.md` - Overall system architecture
- `ISNE_BOOTSTRAP_ARCHITECTURE.md` - Supra-weight bootstrap design

### `/api/`
API reference and usage guides (to be created)

## Key Concepts Summary

### 1. Theory-Practice Bridges
Automatic connection between research papers and code implementations through intelligent keyword extraction and co-location.

### 2. Query-as-Graph
Queries become temporary nodes in the knowledge graph, enabling contextual understanding and path-based retrieval.

### 3. Filesystem Taxonomy
Directory structure provides semantic relationships, with metadata files strengthening connections across the graph.

### 4. Supra-Weights
Multi-dimensional edge weights capturing various relationship types (semantic, structural, co-location, etc.).

## Documentation Philosophy

### Self-Documenting Code
- Research papers co-located with implementations
- Comprehensive module docstrings
- Directory README files with relationships

### Hierarchical Organization
- High-level concepts in `/concepts/`
- Technical details in `/architecture/`
- Usage guides in `/api/`

### Cross-References
Every document includes "Related Resources" linking to:
- Implementation code
- Configuration files
- Other documentation
- Examples and tests

## Getting Started
1. Read `CORE_CONCEPTS.md` for theoretical foundation
2. Review `ARCHITECTURE.md` for system design
3. Check module READMEs in `/src/` for implementation details
4. See `MIGRATION_FROM_ORIGINAL.md` if coming from original HADES

## Contributing
When adding new documentation:
1. Place in appropriate subdirectory
2. Include "Related Resources" section
3. Update this README
4. Cross-reference from relevant code