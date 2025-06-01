# Chunking Types Module Documentation

## Overview

The chunking types module (`src.types.chunking`) provides centralized type definitions for the HADES-PathRAG chunking system. This module defines the core data types, interfaces, and helper functions used throughout the chunking subsystem, ensuring type safety and consistent structure across implementations.

## Module Structure

The chunking types module is organized into four primary files:

1. **chunk.py**: Core chunk type definitions
   - Base `Chunk` and `ChunkMetadata` types
   - Generic chunk helper functions and utilities
   - Output format type definitions

2. **strategy.py**: Chunking strategy interfaces
   - `ChunkerProtocol` defining the chunker interface
   - `BaseChunkerABC` abstract base class for chunkers
   - Registry and factory type definitions

3. **text.py**: Text-specific chunking types
   - `TextChunk` and `TextChunkMetadata` specializations
   - Text chunking options and configuration types
   - Text-specific processing helpers

4. **code.py**: Code-specific chunking types
   - `CodeChunk` and `CodeChunkMetadata` specializations
   - Language-specific symbol information types
   - Code chunking options and configuration types

## Key Types

### Core Types

- `Chunk`: Base type representing a document chunk with content and metadata
- `ChunkMetadata`: Metadata associated with a chunk (e.g., line numbers, source info)
- `ChunkerProtocol`: Interface defining the contract for chunkers
- `BaseChunkerABC`: Abstract base class for implementing chunkers

### Specialized Types

- `TextChunk`/`CodeChunk`: Specialized chunk types for text and code content
- `TextChunkMetadata`/`CodeChunkMetadata`: Specialized metadata types
- `TextChunkingOptions`/`CodeChunkingOptions`: Configuration options for chunking
- `SymbolInfo`: Information about a symbol in code (e.g., function, class)

### Helper Functions

- `create_chunk_metadata()`: Create chunk metadata with default values
- `create_chunk()`: Create a new chunk with the given content and metadata
- `create_text_chunk_metadata()`: Create text-specific chunk metadata
- `create_code_chunk_metadata()`: Create code-specific chunk metadata

## Usage Examples

### Basic Chunk Creation

```python
from src.types.chunking import create_chunk, create_chunk_metadata

# Create chunk metadata
metadata = create_chunk_metadata(
    path="/path/to/document.txt",
    document_type="text",
    line_start=1,
    line_end=10,
    token_count=250
)

# Create a chunk with content and metadata
chunk = create_chunk(
    content="This is the content of the chunk.",
    metadata=metadata
)
```

### Creating a Text Chunk

```python
from src.types.chunking import create_text_chunk_metadata, TextChunk

# Create text-specific metadata
metadata = create_text_chunk_metadata(
    path="/path/to/document.txt",
    document_type="text",
    line_start=1,
    line_end=10,
    token_count=250,
    paragraph_type="paragraph",
    sentence_count=5,
    semantic_score=0.85
)

# Create a text chunk
text_chunk: TextChunk = {
    "id": "chunk_123",
    "content": "This is a paragraph of text content.",
    "metadata": metadata
}
```

### Creating a Code Chunk

```python
from src.types.chunking import create_code_chunk_metadata, CodeChunk

# Create code-specific metadata
metadata = create_code_chunk_metadata(
    path="/path/to/script.py",
    document_type="code",
    language="python",
    line_start=10,
    line_end=25,
    token_count=180,
    symbol_type="function",
    name="process_data",
    complexity=5.2
)

# Create a code chunk
code_chunk: CodeChunk = {
    "id": "chunk_456",
    "content": "def process_data(data):\n    return data.transform()",
    "metadata": metadata
}
```

## Integration with Chunking System

These types are designed to be used by the chunking implementations in `src.chunking`. The next step in the migration process is to update the existing chunker implementations to use these centralized types instead of their local type definitions.

## Future Work

- Migrate existing chunker implementations to use these types
- Create unit tests for type validation
- Expand language-specific type definitions for additional programming languages
- Add specialized types for other document formats (e.g., markdown, HTML, JSON)

## Test Coverage and Type Checking

- Unit tests will be developed to validate type consistency
- Type checking using mypy will be implemented to ensure type safety
- Validation functions will be added to ensure runtime type safety

## Performance Considerations

- TypedDict and Protocol types provide static type checking without runtime overhead
- Helper functions create properly structured objects with minimal boilerplate
- Type definitions are designed to be serializable for storage and transmission
