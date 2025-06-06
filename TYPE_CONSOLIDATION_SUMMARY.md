# Type System Consolidation Summary

## Overview

This document summarizes the consolidation of the HADES-PathRAG type system, establishing clear boundaries between `src/types/` (static typing) and `src/schemas/` (runtime validation) while eliminating overlaps and creating a unified, maintainable architecture.

## Architectural Principles

### **Code Correctness vs Data Correctness**
- **`src/types/`** = **Development-time type safety** - TypedDict, Protocol, type aliases for mypy
- **`src/schemas/`** = **Runtime data validation** - Pydantic models for API boundaries and serialization

### **Dependency Direction**
```
src/schemas/ → src/types/ → (no dependencies)
```
- Schemas import types from `src.types.common` 
- Types are self-contained
- No circular dependencies

## Key Changes Made

### 1. **Centralized Common Types** (`src/types/common.py`)

#### **Type Aliases (Single Source of Truth)**
```python
EmbeddingVector: TypeAlias = Union[List[float], np.ndarray, bytes]  # Consolidated from 3 definitions
UUIDString: TypeAlias = str
PathSpec: TypeAlias = List[str]
MetadataDict: TypeAlias = Dict[str, Any]
```

#### **Standardized Enumerations**
```python
class DocumentType(str, Enum):
    """Unified document types across all modules"""
    TEXT = "text"
    PDF = "pdf" 
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"

class RelationType(str, Enum):
    """Unified relationship types replacing ISNERelationType, RelationshipType"""
    # Primary structural relationships
    CONTAINS = "contains"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    # ... (comprehensive unified set)
```

### 2. **Schema Import Updates**

#### **Before:**
```python
# src/schemas/common/types.py
EmbeddingVector: TypeAlias = Union[List[float], np.ndarray]  # Duplicate!

# src/schemas/common/enums.py  
class DocumentType(str, Enum):  # Duplicate!
    CODE = "code"
    # ...
```

#### **After:**
```python
# src/schemas/common/types.py
from src.types.common import EmbeddingVector, PathSpec, MetadataDict

# src/schemas/common/enums.py
from src.types.common import DocumentType, RelationType, ProcessingStage
```

### 3. **ISNE Duplication Elimination**

#### **Removed Duplicates:**
- `ISNEDocumentType` → Use standardized `DocumentType`
- `ISNERelationType` → Use standardized `RelationType`
- Duplicate enum definitions across `src/schemas/isne/` and `src/types/isne/`

#### **Clear Import Hierarchy:**
```python
# src/schemas/isne/models.py
from src.types.common import DocumentType, RelationType  # Use centralized types

# src/types/isne/models.py  
from ..common import DocumentType, RelationType  # Import from centralized location
```

### 4. **Document Schema Boundaries Clarified**

| Module | Purpose | Responsibility |
|--------|---------|----------------|
| **`src/schemas/documents/`** | Pydantic validation models | API contracts, runtime validation, serialization |
| **`src/types/chunking/`** | Static chunking types | Development-time type safety for chunking operations |
| **`src/types/docproc/`** | Document processing types | Interface contracts, pipeline type definitions |

#### **Clear Separation:**
- **`src/schemas/documents/base.py`**: `DocumentSchema` (Pydantic) - for validation and API
- **`src/types/chunking/document.py`**: `DocumentChunkInfo` (TypedDict) - for static typing
- **`src/types/docproc/document.py`**: `ProcessedDocument` (TypedDict) - for pipeline interfaces

## Benefits Achieved

### ✅ **Eliminated Overlaps**
- **EmbeddingVector**: 3 conflicting definitions → 1 comprehensive definition
- **DocumentType**: 3 separate enums → 1 standardized enum
- **RelationType**: Multiple conflicting relationship enums → 1 unified enum

### ✅ **Clear Responsibilities**
- **Development Safety**: `src/types/` provides mypy-strict type checking
- **Runtime Safety**: `src/schemas/` provides Pydantic validation
- **No more confusion** about where to define what

### ✅ **Maintainable Architecture**
- **Single source of truth** for common types in `src/types/common.py`
- **One-way dependencies** prevent circular imports
- **Backward compatibility** through re-exports in schema modules

### ✅ **Type System Quality**
- **Comprehensive EmbeddingVector** supports List[float], np.ndarray, AND bytes
- **Standardized enums** with consistent naming and comprehensive values
- **Clear documentation** and purpose statements

## Import Guidelines

### **For Application Code:**
```python
# Use types for function signatures and static typing
from src.types.common import EmbeddingVector, DocumentType
from src.types.embedding.base import EmbeddingAdapter

def process_embedding(vector: EmbeddingVector) -> float:
    # mypy ensures type safety
    pass
```

### **For API/Validation Code:**
```python
# Use schemas for data validation and serialization
from src.schemas.embedding.models import EmbeddingResult
from src.schemas.documents.base import DocumentSchema

def api_endpoint(data: dict) -> EmbeddingResult:
    # Pydantic validates runtime data
    return EmbeddingResult.model_validate(data)
```

### **For Schema Definitions:**
```python
# Schemas should import types from src.types.common
from src.types.common import DocumentType, RelationType
from pydantic import BaseModel

class MySchema(BaseModel):
    doc_type: DocumentType  # Uses centralized enum
    relation: RelationType  # Uses centralized enum
```

## Migration Status

### ✅ **Completed:**
1. Consolidated common types into `src/types/common.py`
2. Updated schema imports to use centralized types
3. Eliminated ISNE type duplication
4. Standardized enum naming and values
5. Established clear module boundaries

### 🔄 **Next Steps:**
1. Update remaining imports throughout codebase
2. Run comprehensive testing and type checking
3. Update documentation to reflect new architecture

## Validation Commands

```bash
# Type checking
poetry run mypy src/

# Import validation
python -c "from src.types.common import EmbeddingVector, DocumentType, RelationType; print('✅ Types import successfully')"
python -c "from src.schemas.common.types import EmbeddingVector; print('✅ Schemas import successfully')"

# Test that enums are unified
python -c "from src.types.common import DocumentType; from src.schemas.common.enums import DocumentType as SchemaDocType; assert DocumentType == SchemaDocType; print('✅ Enums are unified')"
```

## Architecture Summary

The consolidation successfully creates a **dual-purpose type system**:

1. **`src/types/`** - Static type safety and development-time correctness
2. **`src/schemas/`** - Runtime validation and data correctness

This architecture follows Python best practices and allows the codebase to benefit from both static type checking (mypy) and runtime validation (Pydantic) without confusion or duplication.