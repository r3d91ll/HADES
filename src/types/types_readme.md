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
  ├── docproc/               # Document processing types (renamed from documents/)
  │   ├── __init__.py  
  │   ├── base.py            # Base document types
  │   └── schema.py          # Document schemas
  ├── chunking/              # Chunking-related types
  │   ├── __init__.py
  │   └── chunk.py           # Chunk type definitions
  ├── storage/               # Storage-related types
  │   ├── __init__.py
  │   └── repository.py      # Storage repository types
  ├── pathrag/               # PathRAG-specific types
  │   ├── __init__.py
  │   └── path.py            # Path reasoning types
  ├── indexing/              # Indexing-related types
  │   ├── __init__.py
  │   └── index.py           # Index structure types
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
3. **Import Structure**: Import types directly from their module, e.g., `from src.types.docproc.base import DocumentType`
4. **Type Exports**: Each module should export its types through `__init__.py` for convenience
5. **Documentation**: All types should have docstrings explaining their purpose and structure
6. **Module Alignment**: Type directories should align with corresponding src/ module directories
7. **Migration Strategy**: Module-specific types should be migrated to the centralized type system

## Type Safety

This module supports the team's commitment to type safety:

1. All type definitions include complete type annotations
2. We use TypedDict, Protocol, and structural types appropriately
3. Validation functions are provided where needed
4. All code should pass mypy validation with strict settings

## Type Migration Strategy

To ensure consistent type usage across the codebase, follow this migration strategy:

1. **Syntax Fixes First**: Fix any syntax errors in type annotations within modules
2. **Type Inventory**: Identify all module-specific types that need to be centralized
3. **Create Centralized Types**: Create equivalent types in the appropriate src/types/ subdirectory
4. **Update Imports**: Update all imports to use the centralized types
5. **Run Type Checking**: Verify all changes with mypy (--disallow-untyped-defs --disallow-incomplete-defs)
6. **Documentation**: Update type documentation to reflect the changes

## Future Compatibility

Centralizing types provides several benefits:

1. **Easier Maintenance**: Single source of truth for type definitions
2. **Mojo Compatibility**: Facilitates future porting to Mojo by centralizing type information
3. **Consistent Validation**: Enables consistent validation across the codebase
4. **Clearer Boundaries**: Establishes clear boundaries between modules

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
