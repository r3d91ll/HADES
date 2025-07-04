# Theory-Practice Bridge Architecture

## Overview

This document describes how HADES maintains connections between theoretical concepts (research papers, documentation) and practical implementations (code, configurations) while leveraging Jina v4's unified multimodal embedding space.

## Core Principles

1. **Unified Semantic Space**: All content shares Jina v4's embedding space for semantic search
2. **Structural Diversity**: File-type-specific processing preserves domain-specific relationships
3. **Explicit Bridges**: Direct links between concepts and their implementations
4. **Bidirectional Navigation**: From theory to practice and vice versa
5. **Context Preservation**: Maintain document structure while embedding

## File Type Processing Pipeline

### 1. Python Files (.py)
```
Input: Python source code
↓
AST Analysis
├── Extract symbols (classes, functions, variables)
├── Analyze imports and dependencies
├── Extract docstrings and comments
└── Map line numbers to symbols
↓
Theory Bridge Detection
├── Find references to papers (citations in comments)
├── Match algorithm names to research
└── Link to documentation files
↓
Jina v4 Embedding
├── Embed full file content
├── Embed individual symbols
└── Embed docstrings separately
↓
Output: Structured node with AST metadata + embeddings
```

### 2. PDF Files (Research Papers)
```
Input: PDF document
↓
Structure Extraction
├── Extract sections and headings
├── Identify figures and tables
├── Extract citations and references
└── Detect algorithms and equations
↓
Practice Bridge Detection
├── Find code references
├── Match algorithm names to implementations
└── Extract evaluation metrics
↓
Jina v4 Embedding
├── Embed full document
├── Embed sections individually
└── Embed figure captions
↓
Output: Structured node with academic metadata + embeddings
```

### 3. Markdown Files (.md)
```
Input: Markdown documentation
↓
Hierarchy Parsing
├── Extract heading structure
├── Parse internal links
├── Extract code blocks
└── Identify API references
↓
Bridge Analysis
├── Link to source code
├── Connect to research papers
└── Map examples to implementations
↓
Jina v4 Embedding
├── Embed full document
├── Embed sections by heading
└── Embed code examples
↓
Output: Structured node with documentation metadata + embeddings
```

## Enhanced Node Structure

```javascript
{
  "_key": "unique_file_identifier",
  "filesystem_path": "/absolute/path/to/file",
  "type": "python|pdf|markdown|...",
  
  // Semantic embeddings
  "embeddings": {
    "jina_v4": [...],      // Full document embedding
    "isne": [...],         // Graph structure embedding
    "chunks": [            // Chunk-level embeddings
      {
        "id": "chunk_001",
        "embedding": [...],
        "type": "symbol|section|paragraph",
        "content": "...",
        "metadata": {...}
      }
    ]
  },
  
  // File-type specific structure
  "structure": {
    // For Python files
    "ast": {
      "symbols": {...},
      "imports": {...},
      "line_mappings": {...}
    },
    
    // For PDFs
    "sections": [...],
    "figures": [...],
    "citations": [...],
    
    // For Markdown
    "headings": [...],
    "links": [...],
    "code_blocks": [...]
  },
  
  // Theory-Practice bridges
  "bridges": {
    "implements": [
      {
        "concept": "late_chunking",
        "source": "jina-embeddings-v4.pdf#section-3.2",
        "confidence": 0.95
      }
    ],
    "documented_in": [
      {
        "path": "docs/api/jina_v4_api.md#late-chunking",
        "type": "api_reference"
      }
    ],
    "references": [
      {
        "type": "citation",
        "target": "research/papers/pathrag.pdf",
        "context": "Based on PathRAG multi-hop reasoning"
      }
    ],
    "used_by": [
      {
        "path": "src/api/server.py",
        "symbol": "process_file",
        "line": 248
      }
    ]
  }
}
```

## Bridge Types

### 1. Implementation Bridges
- `implements`: Code that implements a theoretical concept
- `algorithm_of`: Direct implementation of a published algorithm
- `based_on`: Inspired by or adapted from research

### 2. Documentation Bridges
- `documented_in`: Where this code/concept is documented
- `api_reference`: API documentation for this implementation
- `tutorial_in`: Tutorial or guide featuring this code

### 3. Reference Bridges
- `cites`: Academic citations in code comments
- `references`: General references to other work
- `extends`: Extensions or improvements of existing work

### 4. Usage Bridges
- `used_by`: Other code that uses this implementation
- `example_in`: Example usage in documentation
- `tested_in`: Test files that validate this code

## .hades/relationships.json Enhancement

```json
{
  "version": "2.0",
  "bridges": {
    "theory_practice": [
      {
        "source": {
          "type": "research_paper",
          "path": "research/jina-embeddings-v4.pdf",
          "section": "3.2"
        },
        "target": {
          "type": "implementation",
          "path": "src/jina_v4/jina_processor.py",
          "symbol": "_perform_late_chunking",
          "lines": [493, 614]
        },
        "relationship": "implements",
        "confidence": 0.95,
        "notes": "Direct implementation of late chunking algorithm"
      },
      {
        "source": {
          "type": "code",
          "path": "src/pathrag/pathrag_rag_strategy.py",
          "symbol": "PathRAG"
        },
        "target": {
          "type": "documentation",
          "path": "docs/concepts/CORE_CONCEPTS.md",
          "section": "pathrag-algorithm"
        },
        "relationship": "documented_in",
        "bidirectional": true
      }
    ],
    "semantic": [
      {
        "source": "current_directory",
        "target": "/src/storage/arangodb",
        "type": "depends_on",
        "strength": 0.9
      }
    ]
  }
}
```

## PathRAG Query Enhancement

### Query Processing with Bridges
1. **Concept Queries**: "How does late chunking work?"
   - Find theory nodes (papers mentioning late chunking)
   - Follow `implements` bridges to code
   - Include `documented_in` bridges for explanations

2. **Implementation Queries**: "Show me PathRAG implementation"
   - Find code nodes with PathRAG symbols
   - Follow `based_on` bridges to research
   - Include `used_by` bridges for examples

3. **Cross-Domain Queries**: "Compare theory and practice of ISNE"
   - Find both paper and code nodes
   - Traverse bidirectional bridges
   - Aggregate theoretical and practical perspectives

### Supra-Weight Enhancement
```python
def calculate_bridge_weight(query_node, target_node, bridge_type):
    """Calculate weight based on theory-practice bridges."""
    
    weights = {
        'implements': 0.9,      # Direct implementation
        'documented_in': 0.8,   # Documentation link
        'based_on': 0.7,        # Inspired by
        'references': 0.6,      # General reference
        'used_by': 0.5         # Usage example
    }
    
    # Check for bridges between nodes
    if has_bridge(query_node, target_node, bridge_type):
        return weights.get(bridge_type, 0.5)
    
    return 0.0
```

## Implementation Modules

### 1. Document Parsers
- `src/jina_v4/parsers/pdf_parser.py` - PDF structure extraction
- `src/jina_v4/parsers/markdown_parser.py` - Markdown hierarchy parsing
- `src/jina_v4/parsers/bridge_detector.py` - Theory-practice bridge detection

### 2. Bridge Management
- `src/bridges/bridge_manager.py` - Create and maintain bridges
- `src/bridges/bridge_types.py` - Bridge type definitions
- `src/bridges/bridge_validator.py` - Validate bridge consistency

### 3. PathRAG Extensions
- `src/pathrag/bridge_traversal.py` - Navigate theory-practice bridges
- `src/pathrag/query_enhancer.py` - Enhance queries with bridge context