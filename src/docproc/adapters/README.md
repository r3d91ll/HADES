# Document Processing Adapters

This directory contains adapters for processing various document formats in the HADES-PathRAG system.

## Overview

The adapters follow a standardized interface defined in `base.py` and produce consistent JSON output that can be used throughout the system. Each adapter specializes in handling a specific document format.

## Available Adapters

- **DoclingAdapter**: Processes PDF and other document formats using the Docling library
- **PythonAdapter**: Processes Python source code, extracting code structure and relationships
- **HTMLAdapter**: Processes HTML documents, extracting content and structure

## Python Adapter

The Python adapter (`python_adapter_impl.py`) provides specialized functionality for processing Python source code files:

### Features

- AST-based parsing of Python code structure
- Extraction of symbols (classes, functions, imports)
- Relationship detection between code elements
- Hierarchical representation of code structure
- Type annotations and docstring extraction
- Support for various Python language features (async, decorators, etc.)

### Output Format

The adapter produces a standardized JSON structure with the following key components:

```json
{
  "id": "python_a1b2c3d4",            // Unique identifier for the document
  "source": "/path/to/file.py",       // Source file path
  "content": "```python\n...\n```",   // Markdown-formatted content
  "content_type": "markdown",         // Content format type
  "format": "python",                 // Document format identifier
  "raw_content": "def some_func()...", // Original source code
  
  "metadata": {
    "language": "python",             // Programming language
    "file_size": 1024,                // Size in bytes
    "line_count": 120,                // Number of lines
    "function_count": 5,              // Number of functions
    "class_count": 2,                 // Number of classes
    "import_count": 8,                // Number of imports
    "has_module_docstring": true,     // Whether module has docstring
  },
  
  "entities": [                       // Flat list of entities for quick reference
    {
      "type": "function",
      "name": "process_data",
      "line": 45,
      "docstring": "Process the given data...",
      "parameters": ["data", "options"],
      "confidence": 1.0
    },
    // More entities...
  ],
  
  "symbol_table": {                   // Hierarchical symbol table
    "type": "module",
    "name": "mymodule",
    "docstring": "Module docstring",
    "elements": [
      {
        "type": "class",
        "name": "MyClass",
        "docstring": "Class docstring",
        "elements": [
          {
            "type": "method",
            "name": "my_method",
            "parameters": ["self", "arg1"],
            "returns": "str",
            "docstring": "Method docstring"
          }
        ]
      }
    ]
  },
  
  "relationships": [                 // Cross-references between symbols
    {
      "source": "function_789",
      "target": "function_102",
      "type": "CALLS",
      "weight": 0.9,
      "line": 25
    }
  ]
}
```

### Relationship Types

The adapter identifies various relationship types between code elements:

- **Primary relationships** (weight 0.8-1.0):
  - `CALLS`: Function calling another function
  - `CONTAINS`: Parent-child relationship (e.g., class contains method)
  - `IMPLEMENTS`: Implementation of an interface or protocol

- **Secondary relationships** (weight 0.5-0.7):
  - `IMPORTS`: Import relationship
  - `REFERENCES`: Reference to another code element
  - `EXTENDS`: Inheritance relationship

- **Tertiary relationships** (weight 0.2-0.4):
  - `SIMILAR_TO`: Semantic similarity
  - `RELATED_TO`: General relationship

### Usage

```python
from pathlib import Path
from src.docproc.adapters.python_adapter_impl import PythonAdapter

# Create the adapter
adapter = PythonAdapter()

# Process a Python file
result = adapter.process(Path("/path/to/file.py"))

# Process Python code as text
code = "def hello(): print('Hello, world!')"
result = adapter.process_text(code)
```

## Testing

The adapters are thoroughly tested with unit tests that ensure proper functionality and error handling. The Python adapter has 90%+ test coverage, meeting the project's quality requirements.

Run the tests with:

```bash
poetry run pytest tests/docproc/adapters/
```

Run with coverage:

```bash
poetry run pytest tests/docproc/adapters/ --cov=src.docproc.adapters
```
