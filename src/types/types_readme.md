# Types Module

## Overview

The types module provides centralized type definitions for the entire HADES-PathRAG system. By centralizing type definitions, we achieve:

1. Better type safety through consistent definitions
2. Reduced circular imports
3. Ability to test modules in isolation
4. Clearer documentation of data structures

## Directory Structure

```bash
src/types/
  ├── __init__.py
  ├── types_readme.md        # This file
  ├── common.py              # Common types used across multiple modules
  ├── pipeline/              # Orchestration pipeline types
  │   ├── __init__.py
  │   ├── queue.py           # Queue and backpressure types
  │   └── worker.py          # Worker pool types
  ├── documents/             # Document and chunk types
  │   ├── __init__.py  
  │   ├── base.py            # Base document types
  │   └── schema.py          # Document schemas
  ├── embedding/             # Embedding-related types
  │   ├── __init__.py
  │   └── vector.py          # Embedding vector types
  └── isne/                  # ISNE-specific types
      ├── __init__.py
      └── models.py          # ISNE model types
```

## Usage Guidelines

1. **Domain-Specific Types**: Place types in the appropriate subdirectory based on the module they primarily relate to
2. **Common Types**: Types used across multiple modules should be placed in `common.py`
3. **Import Structure**: Import types directly from their module, e.g., `from src.types.documents.base import DocumentType`
4. **Type Exports**: Each module should export its types through `__init__.py` for convenience
5. **Documentation**: All types should have docstrings explaining their purpose and structure

## Type Safety

This module supports the team's commitment to type safety:

1. All type definitions include complete type annotations
2. We use TypedDict, Protocol, and structural types appropriately
3. Validation functions are provided where needed
4. All code should pass mypy validation with strict settings

### Mypy Configuration

The project uses a strict mypy configuration defined in `mypy.ini` at the project root with the following key settings:

- `disallow_untyped_defs = True`: All functions must have type annotations
- `disallow_incomplete_defs = True`: All parameters must have type annotations
- `check_untyped_defs = True`: Type-check the body of functions without annotations
- `no_implicit_optional = True`: Disallow implicit Optional types
- `warn_return_any = True`: Warn when returning Any from a function

To run mypy with this configuration:

```bash
python -m mypy --config-file=mypy.ini <module_path>
```

### Type Checking Standards

1. **Coverage Requirement**: Maintain a minimum of 85% type checking coverage
2. **Mixed Content Types**: When handling mixed content types (code vs. text), use appropriate type union
3. **Content Categories**: Use the standard content categories defined in `src.config.preprocessor_config`
4. **Error Handling**: Handle type errors gracefully, avoiding runtime type errors
5. **Generic Types**: Use generic types with type parameters instead of Any where possible

## Integration with Testing

Types defined here support unit testing by allowing test code to easily create valid test instances:

```python
from src.types.documents.schema import DocumentSchema

# Create test document schema
test_doc = DocumentSchema(
    id="test-doc-001",
    title="Test Document",
    content="This is a test document for unit testing",
    metadata={"source": "unit-test"}
)
```
