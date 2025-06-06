# Storage Type Definitions

This module contains all type definitions, protocols, and interfaces related to storage operations in HADES-PathRAG.

## Overview

The storage type system provides a clean separation between:
- **Type definitions** (static typing, protocols) - Located here
- **Implementations** (concrete classes) - Remain in `src/storage/`

## Type Hierarchy

### Protocol Interfaces

These are `@runtime_checkable` Protocol classes that define the expected interface for storage operations:

1. **`DocumentRepository`** - Document storage operations
   - `store_document()` - Store a document
   - `get_document()` - Retrieve by ID
   - `update_document()` - Update by ID
   - `search_documents()` - Text search

2. **`GraphRepository`** - Graph operations
   - `create_edge()` - Create edges between nodes
   - `get_edges()` - Get edges for a node
   - `traverse_graph()` - Graph traversal
   - `shortest_path()` - Path finding

3. **`VectorRepository`** - Vector/embedding operations
   - `store_embedding()` - Store vector embeddings
   - `get_embedding()` - Retrieve embeddings
   - `search_similar()` - Vector similarity search
   - `hybrid_search()` - Combined text + vector search

4. **`UnifiedRepository`** - Combined interface
   - Inherits from all three above
   - Adds `setup_collections()` and `collection_stats()`

### Abstract Base Class

**`AbstractUnifiedRepository`** - ABC version of the unified interface for implementations that prefer abstract base classes over protocols. This provides async methods for storage operations.

## Migration from Legacy Code

### Before (scattered types):
```python
# src/storage/interfaces.py
class DocumentRepository(Protocol):
    # Protocol definition

# src/storage/arango/repository_interfaces.py  
class UnifiedRepository(ABC):
    # ABC definition
```

### After (centralized types):
```python
# src/types/storage/interfaces.py
# All type definitions here

# src/storage/interfaces.py
# Re-exports for backward compatibility
from src.types.storage import DocumentRepository
```

## Usage

### For New Code:
```python
from src.types.storage import DocumentRepository, UnifiedRepository

class MyRepository:
    """Implementation of storage interfaces."""
    
    def store_document(self, document: NodeData) -> NodeID:
        # Implementation
        pass
```

### For Type Annotations:
```python
from src.types.storage import DocumentRepository

def process_documents(repo: DocumentRepository) -> None:
    """Process documents using any repository implementation."""
    doc = repo.get_document(NodeID("123"))
```

### For Protocol Checking:
```python
from src.types.storage import DocumentRepository

# Runtime check if object implements protocol
if isinstance(my_repo, DocumentRepository):
    # Safe to use document operations
    my_repo.store_document(doc)
```

## Backward Compatibility

The original modules (`src/storage/interfaces.py` and `src/storage/arango/repository_interfaces.py`) now re-export types from this centralized location for backward compatibility. Existing code will continue to work without changes.

## Benefits

1. **Single Source of Truth** - All storage type definitions in one place
2. **Clear Separation** - Types vs implementations cleanly separated  
3. **Protocol-Based Design** - Flexible interfaces using Python protocols
4. **Type Safety** - Full mypy support for static type checking
5. **Runtime Checking** - `@runtime_checkable` protocols for dynamic validation