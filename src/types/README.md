# Type Definitions

## Overview
This directory contains all type definitions, protocols, and contracts used throughout HADES. Following Python's typing best practices, these provide clear interfaces and data structures.

## Structure
- `common.py` - Common types used across modules
- `components/` - Component protocols and contracts
- `isne/` - ISNE-specific types
- `pipeline/` - Pipeline data structures
- `storage/` - Storage interfaces and types

## Related Resources
- **Implementation**: Types are implemented in `/src/`
- **Documentation**: `/docs/api/types_reference.md`
- **Examples**: Used throughout codebase

## Design Principles
1. **Protocol-based**: Use protocols for interfaces
2. **Pydantic Models**: For data validation
3. **Type Safety**: Comprehensive type hints
4. **Documentation**: Types serve as documentation

## Key Protocols
- `ComponentProtocol` - Base for all components
- `StorageProtocol` - Storage interfaces
- `EmbedderProtocol` - Embedding generation
- `GraphEnhancerProtocol` - ISNE interface

## Usage
```python
from src.types.components.protocols import ComponentProtocol
from src.types.pipeline.document_schema import ProcessedDocument
```

## Dependencies
- pydantic for data validation
- typing and typing_extensions
- Python 3.8+ for protocol support