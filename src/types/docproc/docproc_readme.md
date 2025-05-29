# Document Processing Types

## Overview

This directory contains the centralized type definitions for the document processing system (docproc). These types are used throughout the codebase to ensure consistent data structures and type safety, especially important for future portability to Mojo.

## Type Definitions

The following type definitions are provided:

- **Base Document Types** (`base.py`): Core document data structures
  - BaseEntity
  - BaseMetadata
  - BaseDocument

- **Document Schema Types** (`schema.py`): Validation schemas for documents
  - DocumentSchema
  - ChunkSchema
  - MetadataSchema
  - EntitySchema

## Migration Strategy

This directory replaces the previously named `documents` directory and consolidates all document processing types from the `src/docproc` module. When implementing code in `src/docproc`, always import types from this centralized location.

## Usage Examples

```python
# Correct usage - import from centralized type system
from src.types.docproc.base import BaseDocument, BaseMetadata
from src.types.docproc.schema import DocumentSchema

# Create document with proper typing
def process_document(doc: BaseDocument) -> DocumentSchema:
    # Implementation using proper types
    pass
```

## Testing

All types should have corresponding test cases in the `tests/types/docproc/` directory.
