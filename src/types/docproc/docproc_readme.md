# Document Processing (docproc) Type System

## Overview
This directory contains the centralized type definitions for the document processing subsystem of HADES. 
These types facilitate consistent representation of documents, code elements, and processing configurations 
across the codebase.

## Structure

- `__init__.py`: Exports all public types from the module
- `adapter.py`: Type definitions for document processing adapters and protocols
- `code_elements.py`: TypedDict definitions for code structure elements
- `config.py`: Configuration types for document processing operations
- `document.py`: Core document type definitions and processing result types
- `enums.py`: Enumeration types used throughout the document processing system
- `python.py`: Pydantic models for Python document processing

## Usage

Import types from this centralized location rather than from individual implementation files:

```python
# Preferred import style
from src.types.docproc import PythonDocument, CodeElement, DocumentProcessingResult

# Rather than from implementation modules
# from src.docproc.schemas.python_document import PythonDocument  # Avoid this
```

## Type Migration Status

| Source Component | Migration Status | Notes |
|------------------|-----------------|-------|
| Base adapter types | Complete | Migrated from src/docproc/adapters/base.py |
| Code element types | Complete | Already centralized |
| Document schemas | Complete | Migrated from src/docproc/schemas/* |
| Processing config | Complete | Migrated from src/docproc/core.py |
| Format-specific types | Complete | Migrated from format adapters |

## Testing

Types in this directory are validated through:

- Static type checking with mypy
- Unit tests in `test/types/docproc/`
- Runtime validation via Pydantic models

## References

- [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
