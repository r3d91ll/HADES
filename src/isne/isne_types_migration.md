# ISNE Types Migration Plan

## Overview

This document outlines the plan for migrating type definitions from `src/isne/types` to the centralized `src/types/isne` directory. Following the "replace-don't-migrate" approach outlined in the TODO.md file, we'll update imports across the ISNE module to use the centralized types.

## Types Migrated

- `DocumentType`
- `RelationType`
- `IngestDocument`
- `DocumentRelation`
- `LoaderResult`
- `EmbeddingVector`

## Migration Process

1. ✅ Add missing types to `src/types/isne/models.py`
2. ✅ Update `src/types/isne/__init__.py` to export all necessary types
3. ⬜ Update imports in ISNE module files to use the centralized types
4. ⬜ Add deprecation warnings to `src/isne/types/__init__.py` to guide future usage
5. ⬜ Create necessary tests to ensure type compatibility

## Files to Update

The following files need to be updated to use the centralized types:

- All files in the ISNE module that import from `src/isne/types`
- The most common import pattern to replace is:

  ```python
  from src.isne.types import DocumentType, RelationType, IngestDocument, DocumentRelation, LoaderResult, EmbeddingVector
  ```

  with:

  ```python
  from src.types.isne import DocumentType, RelationType, IngestDocument, DocumentRelation, LoaderResult, EmbeddingVector
  ```

## Deprecation Strategy

Rather than immediately removing the old types, we'll add deprecation warnings to guide users toward the new centralized types:

```python
# src/isne/types/__init__.py
import warnings

warnings.warn(
    "The types in src.isne.types are deprecated. Use src.types.isne instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the centralized location
from src.types.isne import (
    DocumentType,
    RelationType,
    IngestDocument,
    DocumentRelation,
    LoaderResult,
    EmbeddingVector
)
```

## Testing Strategy

1. Create unit tests to verify that the centralized types work correctly with existing ISNE code
2. Test all affected functionality to ensure the new types don't break existing behavior
3. Ensure proper type checking with mypy

## Implementation Timeline

1. Update imports in core ISNE modules first
2. Then update imports in dependent modules
3. Add tests to verify type compatibility
4. Add deprecation warnings to guide future usage
