# Code Chunkers Module

This module provides specialized chunkers for different code formats (Python, YAML, JSON) that create structure-aware chunks for improved semantic understanding.

## Overview

The code chunkers use structure-aware parsing techniques to break down code files into meaningful chunks based on their syntax structure rather than arbitrary text boundaries. This approach preserves the semantic meaning of the code and produces better embeddings for retrieval.

## Components

### Python Code Chunker

- Uses AST parsing to extract meaningful code entities (functions, classes, methods)
- Creates chunks that preserve the semantic structure of Python code
- Extracts relationships between code entities for better graph construction

### YAML Code Chunker

- Parses YAML files to extract hierarchical structure
- Identifies key elements and their relationships
- Preserves parent-child relationships between YAML components

### JSON Code Chunker

- Analyzes JSON structure including objects and arrays
- Breaks down complex JSON into semantically meaningful chunks
- Preserves path information and hierarchical relationships

## Integration with Multi-Model Embedding

Each chunker is integrated with the multi-model embedding approach:

1. **Model Selection**: Different embedding models are selected based on file type
2. **Metadata Tracking**: Each chunk includes `embedding_model` and `embedding_type` metadata
3. **ISNE Awareness**: The ISNE stage uses this metadata to properly bridge different embedding spaces

## Usage

The chunkers are automatically selected based on file type in the processing pipeline:

```python
# File type detection is handled by the format_detector
format_type = detect_format(file_path)

# Chunker selection based on format
if format_type == "python":
    chunker = get_chunker("python_code")
elif format_type in ["yaml", "yml"]:
    chunker = get_chunker("yaml_code")
elif format_type == "json":
    chunker = get_chunker("json_code")
else:
    chunker = get_chunker("text")  # Default text chunker
```

## Graph Construction

The multi-model approach includes enhanced graph construction that is aware of different embedding models:

1. Same-model connections receive full edge weight (1.0)
2. Cross-model connections receive reduced edge weight (0.7)
3. Node metadata includes embedding model information for further analysis

## Performance Considerations

- AST-based parsing is more computationally intensive than simple text chunking
- Each specialized chunker includes a fallback mode for invalid syntax
- Batch processing for embeddings helps manage memory usage
