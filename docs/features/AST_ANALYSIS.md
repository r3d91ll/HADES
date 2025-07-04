# AST Analysis for Code Understanding

## Overview

HADES includes sophisticated AST (Abstract Syntax Tree) analysis for code files, providing deeper semantic understanding than treating code as plain text. This feature extracts structural information, tracks dependencies, and creates intelligent chunks based on code symbols rather than arbitrary text boundaries.

## Key Benefits

1. **Semantic Code Chunking**: Chunks align with logical code units (functions, classes, methods)
2. **Dependency Tracking**: Understands import relationships and function calls
3. **Symbol Extraction**: Identifies and indexes all code symbols for better retrieval
4. **Preserves Context**: Maintains relationships between code elements

## How It Works

### 1. Code Detection

When processing files, HADES automatically detects Python files by:
- File extension (`.py`)
- Content analysis
- Explicit file type specification

### 2. AST Parsing

For Python files, the system:
```python
# Parse the code into an AST
tree = ast.parse(code, filename=filename)

# Extract semantic information
- Classes and their methods
- Functions and their signatures
- Import statements
- Variable definitions
- Decorators and metadata
```

### 3. Symbol-Based Chunking

Instead of splitting code at arbitrary boundaries, chunks are created at:
- Class definitions
- Function/method boundaries
- Logical code blocks

Example:
```python
# Original code file
import numpy as np
from typing import List

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self):
        self.data = []
    
    def process(self, items: List[float]) -> np.ndarray:
        """Process items into array."""
        return np.array(items)

def helper_function(x):
    return x * 2
```

Becomes chunks:
1. **Import Summary Chunk**: Contains all imports and file structure
2. **DataProcessor Class Chunk**: Complete class with methods
3. **helper_function Chunk**: Standalone function

### 4. Enhanced Metadata

Each code chunk includes:
- **Symbol Information**: Name, type, signature
- **Line Ranges**: Exact location in source file
- **Dependencies**: What the code imports/calls
- **Docstrings**: Embedded documentation
- **AST Keywords**: Extracted patterns and libraries

## Configuration

Enable AST analysis in `src/config/jina_v4/config.yaml`:

```yaml
features:
  ast_analysis:
    enabled: true
    languages: ["python"]
    extract_imports: true
    extract_symbols: true
    extract_call_graph: true
    extract_keywords: true
    use_symbol_boundaries: true
    include_docstrings: true
    max_symbol_size: 1000
```

## Example Output

When processing a Python file, the AST analyzer provides:

```json
{
  "symbols": {
    "DataProcessor": {
      "type": "class",
      "line_start": 5,
      "line_end": 15,
      "docstring": "Process data efficiently.",
      "children": ["__init__", "process"]
    },
    "DataProcessor.process": {
      "type": "method",
      "line_start": 11,
      "line_end": 14,
      "signature": "(self, items: List[float]) -> np.ndarray",
      "docstring": "Process items into array.",
      "dependencies": ["numpy.array"]
    }
  },
  "imports": {
    "np": "numpy",
    "List": "typing.List"
  },
  "call_graph": {
    "DataProcessor.process": ["numpy.array"]
  },
  "keywords": {
    "symbols": ["DataProcessor", "process", "helper_function"],
    "libraries": ["numpy", "typing"],
    "patterns": [],
    "concepts": ["Process", "List", "array"]
  }
}
```

## Integration with ISNE

The AST analysis enhances ISNE's graph construction by:

1. **Creating Symbol Nodes**: Each symbol becomes a node in the knowledge graph
2. **Import Edges**: Dependencies create edges between files
3. **Call Graph Edges**: Function calls create relationships
4. **Hierarchical Structure**: Classes contain methods, modules contain classes

## Benefits for Retrieval

When querying for code-related information:

1. **Precise Symbol Search**: "Find the DataProcessor class" returns exact matches
2. **Dependency Queries**: "What uses numpy?" finds all dependent code
3. **Semantic Understanding**: "How to process data?" finds relevant methods
4. **Context Preservation**: Retrieved chunks include necessary imports and context

## Future Extensions

The AST analysis framework is designed to support:

1. **Additional Languages**: JavaScript, TypeScript, Java, etc.
2. **Cross-Language Dependencies**: Track relationships across languages
3. **Code Quality Metrics**: Complexity analysis, style checking
4. **Refactoring Suggestions**: Based on code patterns
5. **Security Analysis**: Identify potential vulnerabilities

## Performance Considerations

- AST parsing adds minimal overhead (~10-50ms per file)
- Symbol-based chunks are typically more meaningful than text chunks
- Reduces total chunk count while improving relevance
- GPU acceleration not required for AST analysis

## Best Practices

1. **Keep Functions Focused**: Smaller functions create better chunks
2. **Use Docstrings**: Improves keyword extraction and retrieval
3. **Clear Imports**: Explicit imports help dependency tracking
4. **Logical Organization**: Well-structured code produces better chunks

The AST analysis makes HADES particularly effective for:
- Code search and navigation
- API documentation
- Dependency analysis
- Code understanding tasks
- Technical knowledge bases