# Filesystem Metadata Design

## Overview
This document describes how HADES uses filesystem structure and metadata files to create rich semantic relationships in the knowledge graph.

## Metadata Hierarchy

### 1. Directory-Level Metadata (README.md)
Each directory contains a README.md that provides:
- **Purpose**: What this directory contains
- **Contents**: List of files/subdirectories with descriptions
- **Related Resources**: Links to related parts of the codebase
- **Dependencies**: What this module depends on
- **Key Concepts**: Important ideas implemented here

### 2. File-Level Metadata (Docstrings)
Each Python file begins with a comprehensive module docstring:
```python
"""
Module Name and Purpose

Detailed description of what this module does and how it fits
into the larger system.

Module Organization:
- Class/Function structure
- Key algorithms
- Implementation status

Related Resources:
- Co-located research papers
- Configuration files
- Documentation
- Tests
"""
```

### 3. Co-located Research
Research papers are placed directly with their implementations:
- `/src/pathrag/PathRAG_paper.pdf`
- `/src/jina_v4/jina-embeddings-v4.pdf`

## Graph Relationships from Metadata

### Automatic Relationships
1. **Co-location**: Files in same directory
2. **Parent-Child**: Directory hierarchy
3. **Cross-references**: Links in README files
4. **Import relationships**: Code dependencies

### Enhanced Relationships from Metadata
```python
# From README.md parsing
relationships = {
    "related_to": ["/docs/api/pathrag_api.md"],
    "configured_by": ["/config/pathrag_config.yaml"],
    "tested_by": ["/tests/unit/pathrag/"],
    "documented_in": ["/docs/concepts/CORE_CONCEPTS.md"]
}
```

## Benefits for ISNE

### 1. Richer Training Signal
- More relationship types beyond co-location
- Explicit semantic connections
- Documentation-code bridges

### 2. Better Query Routing
```python
# Query about "PathRAG configuration"
# Metadata helps find:
1. /src/pathrag/ (implementation)
2. /config/pathrag_config.yaml (configuration)
3. /docs/api/pathrag_api.md (usage)
4. PathRAG_paper.pdf (theory)
```

### 3. Contextual Understanding
Late chunking preserves the module-level context:
- Chunks from a file retain the file's docstring context
- Chunks know their related resources
- Queries can follow metadata links

## JSON Structure with Metadata

```json
{
  "document_id": "src_pathrag_pathrag_rag_strategy.py",
  "file_metadata": {
    "docstring": "PathRAG Processor Implementation...",
    "related_resources": {
      "research": "PathRAG_paper.pdf",
      "config": "/config/pathrag_config.yaml",
      "docs": ["/docs/api/pathrag_api.md"],
      "tests": ["/tests/unit/pathrag/"]
    }
  },
  "directory_metadata": {
    "readme_content": "PathRAG Implementation...",
    "purpose": "Path-based retrieval implementation",
    "relationships": {
      "uses": ["/src/storage/", "/src/isne/"],
      "used_by": ["/src/api/"]
    }
  },
  "chunks": [
    {
      "content": "class PathRAGProcessor:...",
      "inherited_context": {
        "module_purpose": "PathRAG implementation",
        "related_theory": "PathRAG_paper.pdf"
      }
    }
  ]
}
```

## Implementation in Jina v4

The Jina v4 processor should:
1. Parse README.md files when processing directories
2. Extract module docstrings as document-level context
3. Build relationship graph from metadata links
4. Include metadata in chunk context for late chunking
5. Generate keywords that bridge documentation and code

## Best Practices

1. **Consistent Metadata**: Use same structure across directories
2. **Bidirectional Links**: If A references B, B should reference A
3. **Descriptive Docstrings**: First paragraph should be self-contained
4. **Regular Updates**: Keep metadata current with code changes
5. **Semantic Clarity**: Use clear, searchable terms

This metadata-driven approach creates a self-documenting codebase where the filesystem structure itself becomes a rich source of semantic information for the knowledge graph.