# Python Code Chunker

## Overview

The Python Code Chunker is a specialized chunking component for processing Python source files in the HADES-PathRAG system. Unlike standard text chunkers that split documents based on arbitrary character or token counts, this chunker preserves the semantic structure of Python code by:

1. Using Abstract Syntax Tree (AST) parsing to identify code entities
2. Creating chunks based on logical code structures (functions, classes, methods)
3. Preserving relationships between code entities
4. Maintaining context for each chunk (docstrings, imports, type hints)

## Features

- **Structure-Preserving Chunking**: Creates chunks based on code structure rather than arbitrary boundaries
- **Relationship Tracking**: Captures relationships between code entities (calls, inheritance, imports)
- **Hierarchical Structure**: Preserves class-method hierarchies in the chunks
- **Docstring Integration**: Includes docstrings with their associated code entities
- **Fallback Mechanism**: Falls back to text-based chunking if code parsing fails

## Usage

```python
from src.chunking.code_chunkers.python_chunker import PythonCodeChunker

# Initialize the chunker with custom configuration
chunker = PythonCodeChunker(config={
    "include_imports": False,        # Whether to create chunks for import statements
    "include_docstrings": True,      # Whether to include docstrings in chunks
    "include_source": True,          # Whether to include source code in chunks
    "min_chunk_size": 50,            # Minimum chunk size in characters
    "max_chunk_size": 1000           # Maximum chunk size in characters
})

# Process a Python file
with open("example.py", "r") as f:
    code = f.read()

chunks = chunker.chunk(code, metadata={"file_path": "example.py"})

# Each chunk contains:
# - chunk_id: Unique identifier for the chunk
# - type: Type of code entity (function, class, method, module)
# - text: Content of the chunk (docstring + code)
# - metadata: Additional information about the chunk
#   - source: Source file path
#   - line_range: Start and end line numbers
#   - type: Type of code entity
#   - name: Name of the code entity
#   - qualified_name: Fully qualified name of the entity
#   - references: List of relationships to other chunks
```

## Integration with ISNE

The Python Code Chunker is designed to work seamlessly with the ISNE (Inductive Shallow Node Embedding) model for generating enhanced embeddings. The relationship data captured by the chunker serves as the graph structure for ISNE processing, allowing the model to learn from both the content of code entities and their structural relationships.

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_imports` | bool | `False` | Whether to create chunks for import statements |
| `include_docstrings` | bool | `True` | Whether to include docstrings in chunks |
| `include_source` | bool | `True` | Whether to include source code in chunks |
| `min_chunk_size` | int | `50` | Minimum chunk size in characters |
| `max_chunk_size` | int | `1000` | Maximum chunk size in characters |

## Chunk Metadata

Each chunk produced by the Python Code Chunker includes metadata that captures the context and relationships of the code entity:

```json
{
  "chunk_id": "file123_function_calculate_distance",
  "type": "function",
  "text": "def calculate_distance(point1, point2):\n    \"\"\"Calculate Euclidean distance between two points.\"\"\"\n    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)",
  "metadata": {
    "source": "geometry.py",
    "line_range": [15, 17],
    "type": "function",
    "name": "calculate_distance",
    "qualified_name": "geometry.calculate_distance",
    "parameters": ["point1", "point2"],
    "returns": "float",
    "decorators": [],
    "complexity": 2,
    "references": [
      {
        "type": "CALLS",
        "target": "file123_module_math",
        "weight": 0.8
      }
    ]
  }
}
```

## Future Improvements

- Add support for type hint analysis in chunk metadata
- Implement more advanced complexity metrics
- Improve handling of large Python files with nested classes/functions
- Add support for Python modules with multiple files (project-level analysis)
