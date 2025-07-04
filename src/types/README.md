# Type Definitions

## Overview
This directory mirrors the src/ structure exactly, providing type definitions for each module. This pattern ensures consistency for both human maintainability and ISNE pattern learning.

## Structure (Mirrors src/)
- `common.py` - Common types, enums, base schemas, ComponentType, ComponentMetadata
- `api/` - API request/response types
- `concepts/` - Concept-related types
- `isne/` - ISNE models, training, and bootstrap types
- `jina_v4/` - Jina V4 processor types
- `pathrag/` - PathRAG strategy and RAG types
- `storage/` - Storage interfaces, Sequential-ISNE types
- `utils/` - Utility types
  - `filesystem/` - File and directory metadata types
- `validation/` - Validation result types

## Design Principles
1. **Mirror Structure**: Each src/ module has corresponding types/ module
2. **Pydantic Models**: For data validation and serialization
3. **Type Safety**: Comprehensive type hints throughout
4. **Backward Compatibility**: Field aliasing for legacy support
5. **Pattern Learning**: Consistent structure aids ISNE directory understanding

## Key Types
- `BaseSchema` - Base Pydantic model with custom config
- `ComponentType`, `ComponentMetadata` - Generic component types (in common.py)
- `StorageInput`, `StorageOutput` - Storage operations (in storage/types.py)
- `RAGStrategyInput`, `RAGStrategyOutput` - PathRAG types (in pathrag/types.py)
- `ISNEConfig`, `TrainingMetrics` - ISNE types (in isne/)
- `FileMetadata`, `DirectoryMetadata` - Filesystem types (in utils/filesystem/)

## Usage
```python
# Common types
from src.types.common import BaseSchema, ProcessingStatus, ComponentType

# Storage types
from src.types.storage.types import StorageInput, StorageOutput

# PathRAG types
from src.types.pathrag.types import RAGStrategyInput, RAGMode

# ISNE types
from src.types.isne.models import ISNEConfig
```

## Directory Mirroring Benefits
1. **3AM Debugging**: Easy to find types for any module
2. **Pattern Recognition**: ISNE can learn from consistent structure
3. **Maintainability**: Clear 1:1 mapping between code and types
4. **Discoverability**: Types location is predictable

## Dependencies
- pydantic for data validation
- typing and typing_extensions
- Python 3.8+ for advanced typing features