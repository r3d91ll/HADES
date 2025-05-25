# HADES-PathRAG Schemas

## Overview

This directory houses centralized Pydantic v2 schema models for the HADES-PathRAG system. The schemas provide consistent validation, serialization, and type safety across all modules, replacing the previous mix of TypedDict, Protocol, and Pydantic models.

## Directory Structure

```
schemas/
├── common/              # Common and shared schemas
│   ├── base.py          # Base schema classes
│   ├── enums.py         # Shared enumerations
│   ├── types.py         # Common type definitions
│   └── __init__.py
├── documents/           # Document-related schemas
│   ├── base.py          # Core document models
│   ├── dataset.py       # Dataset models
│   ├── relations.py     # Document relationship models
│   └── __init__.py
├── embedding/           # Embedding-related schemas
│   ├── adapters.py      # Embedding adapter models
│   ├── models.py        # Embedding result and config models
│   └── __init__.py
├── isne/                # ISNE-specific schemas
│   ├── documents.py     # ISNE document representations
│   ├── models.py        # ISNE model configurations
│   ├── relations.py     # ISNE relationship models
│   └── __init__.py
├── pipeline/            # Pipeline orchestration schemas
│   ├── base.py          # Base pipeline models
│   ├── queue.py         # Queue management models
│   ├── text.py          # Text pipeline models
│   ├── workers.py       # Worker configuration models
│   └── __init__.py
└── __init__.py
```

## Usage Guidelines

### Importing Schemas

Always import schemas using their full path to ensure clarity and prevent circular imports:

```python
from src.schemas.documents.base import DocumentSchema
from src.schemas.embedding.models import EmbeddingConfig
from src.schemas.common.enums import DocumentType
```

### Extending Schemas

When extending schemas, inherit from the base domain schema:

```python
from src.schemas.documents.base import DocumentSchema

class MyCustomDocument(DocumentSchema):
    custom_field: str
    
    class Config:
        # Inherit the parent config
        model_config = DocumentSchema.model_config
```

### JSON Serialization

Use the `model_dump_safe()` method for JSON serialization to ensure proper handling of numpy arrays and other complex types:

```python
document = DocumentSchema(...)
json_data = document.model_dump_safe()
```

### Validation

Leverage Pydantic's validation capabilities to ensure data integrity:

```python
try:
    config = EmbeddingConfig(**user_input)
except ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
```

### Working with Enumerations

Use the enumeration types for type safety and consistent values:

```python
from src.schemas.common.enums import DocumentType

document = DocumentSchema(
    id="doc1",
    content="Example content",
    source="example.txt",
    document_type=DocumentType.TEXT
)
```

## Migration Guidelines

When migrating existing code to use these schemas:

1. Replace TypedDict with Pydantic models
2. Replace Protocol interfaces with concrete Pydantic models
3. Update function signatures to use the new types
4. Add validation at module boundaries
5. Use model_dump_safe() for serialization

## Testing

All schemas include validation rules and should be covered by unit tests. Ensure that edge cases and validation errors are properly tested.

## Contributing

When adding new schemas:

1. Place them in the appropriate domain directory
2. Inherit from BaseSchema for consistent configuration
3. Add proper docstrings and field descriptions
4. Include field validators for complex validation rules
5. Update this README if adding new domain categories
