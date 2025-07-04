# Quick Fix Guide - Type Errors

## 🚀 Biggest Impact Fixes (Will resolve 100+ errors)

### 1. Create RAG Types (5 minutes, resolves ~50 errors)
```python
# Create: src/types/pathrag/rag_types.py
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RAGMode(str, Enum):
    STANDARD = "standard"
    PATHRAG = "pathrag"
    HYBRID = "hybrid"

class RAGStrategyInput(BaseModel):
    query: str
    mode: RAGMode = RAGMode.STANDARD
    top_k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)

class PathInfo(BaseModel):
    path: List[str]
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGResult(BaseModel):
    content: str
    score: float
    source: str
    path: Optional[PathInfo] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGStrategyOutput(BaseModel):
    results: List[RAGResult]
    total_results: int
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 2. Quick Storage Fixes (10 minutes, resolves ~30 errors)
```python
# Update: src/types/storage/interfaces.py

# Add to QueryInput class:
query_embedding: Optional[List[float]] = None
top_k: int = 10
search_options: Dict[str, Any] = Field(default_factory=dict)

# Add to StorageOutput class:
document_id: Optional[str] = None
documents: List[Dict[str, Any]] = Field(default_factory=list)
error_message: Optional[str] = None

# Add to ComponentMetadata in common.py:
processed_at: Optional[datetime] = None
metrics: Dict[str, Any] = Field(default_factory=dict)
```

### 3. Install Dependencies (2 minutes, resolves ~20 errors)
```bash
poetry add python-arango pathspec
poetry add --dev types-tqdm
```

### 4. Add Missing Imports (5 minutes, resolves ~10 errors)
```python
# In files showing ComponentType/ComponentMetadata errors, add:
from src.types.common import ComponentType, ComponentMetadata
```

## 📊 Expected Results After Quick Fixes
- Before: 256 errors
- After Phase 1: ~150 errors (-106)
- After Phase 2: ~80 errors (-70)
- After all fixes: 0 errors

## 🎯 Priority Order
1. Create RAG types file
2. Update storage interfaces
3. Install dependencies
4. Fix imports
5. Handle remaining issues

Total time estimate: 30-45 minutes for major impact