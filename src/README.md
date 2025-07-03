# HADES Source Code

## Overview
This directory contains the core implementation of HADES, organized into focused modules that work together to provide graph-enhanced retrieval.

## Directory Structure

### Core Components
- `jina_v4/` - Unified document processor using Jina v4
  - Co-located: `jina-embeddings-v4.pdf` research paper
  - Handles: parsing, embedding, chunking, keywords
  
- `isne/` - Graph-based embedding enhancement
  - Models: Core ISNE and directory-aware variants
  - Training: Pipeline and specialized trainers
  
- `pathrag/` - Path-based retrieval implementation
  - Co-located: `PathRAG_paper.pdf` research paper
  - Features: Query-as-graph, supra-weights

### Support Components
- `storage/` - Unified storage layer (ArangoDB)
- `api/` - FastAPI service implementation
- `types/` - Type definitions and protocols
- `utils/` - Utility functions and helpers

## Design Philosophy

### Research-Code Co-location
We co-locate research papers with their implementations to maintain strong theory-practice bridges. This allows:
- Immediate reference to theoretical foundations
- Clear traceability from research to code
- Better understanding of implementation decisions

### Metadata-Driven Organization
Each directory contains a README.md that:
- Describes the module's purpose
- Lists related resources across the codebase
- Documents dependencies and relationships
- Provides implementation status

### Filesystem as Taxonomy
Directory structure provides semantic information:
- Co-located files are semantically related
- Directory depth indicates specialization
- Parent directories provide context

## Key Relationships

### Processing Flow
```
jina_v4 → isne → storage
   ↓        ↓       ↓
Document  Enhance  Store
```

### Query Flow
```
api → pathrag → storage
 ↓       ↓         ↓
Query  Navigate  Retrieve
```

## Implementation Status
- ✅ Core architecture complete
- ✅ ISNE implementation ready
- ✅ PathRAG algorithm implemented
- ⚠️ Jina v4 has placeholder methods
- ⚠️ API layer needs implementation

## Getting Started
1. Review module READMEs for detailed information
2. Check co-located research papers for theory
3. Follow type definitions for interfaces
4. Use configuration files for customization