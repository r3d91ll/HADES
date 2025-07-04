# Structured Text File Processing Architecture

## Overview

This document extends our theory-practice bridge architecture to handle structured text files (XML, YAML, JSON, LaTeX, TOML) while leveraging Jina v4's multimodal capabilities.

## Key Principles

1. **Structure Preservation**: Extract and maintain hierarchical/relational structures
2. **Schema Awareness**: Detect and validate against schemas when available
3. **Reference Resolution**: Follow and resolve cross-file references
4. **Semantic Typing**: Understand data types and their relationships
5. **Bridge Detection**: Identify connections to code and documentation

## File Type Processing Strategies

### 1. XML Files
```
Input: XML document
↓
Structure Extraction
├── Parse DOM tree
├── Extract namespaces and schemas
├── Identify XPath patterns
└── Detect DTD/XSD references
↓
Semantic Analysis
├── Identify configuration vs data
├── Extract entity relationships
├── Detect API endpoints/methods
└── Find documentation links
↓
Bridge Detection
├── Match element names to code symbols
├── Link schemas to classes
└── Connect configs to implementations
↓
Jina v4 Embedding
├── Embed full document
├── Embed major sections
└── Embed schema definitions
```

**Key Features:**
- Namespace-aware processing
- XPath-based chunking for large files
- Schema validation and type inference
- XSLT transformation detection

### 2. YAML/JSON Files
```
Input: YAML/JSON document
↓
Structure Analysis
├── Parse object hierarchy
├── Detect schema patterns
├── Identify references ($ref, !include)
└── Extract type information
↓
Purpose Classification
├── Configuration file
├── API specification (OpenAPI/Swagger)
├── Data file
├── Schema definition
└── Package manifest
↓
Bridge Detection
├── Config keys → code parameters
├── API specs → implementations
├── Schema → data models
└── Package deps → code imports
↓
Jina v4 Embedding
├── Embed with structural context
├── Preserve hierarchy in chunks
└── Maintain reference links
```

**Key Features:**
- JSON Schema validation
- YAML anchor/alias resolution
- OpenAPI/AsyncAPI detection
- Package.json/requirements understanding

### 3. LaTeX Files
```
Input: LaTeX document
↓
Document Structure
├── Extract preamble and packages
├── Parse sections/chapters
├── Identify equations and theorems
└── Extract citations and references
↓
Academic Content Extraction
├── Equations → algorithm implementations
├── Theorems → test cases
├── Figures → data visualizations
└── Citations → related work
↓
Bridge Detection
├── Algorithm blocks → code
├── Equations → implementations
├── Citations → papers/code
└── Figures → notebooks
↓
Jina v4 Embedding
├── Embed sections
├── Special handling for math
└── Preserve citation context
```

**Key Features:**
- Math notation preservation
- BibTeX integration
- Cross-reference resolution
- Package dependency tracking

### 4. TOML Files
```
Input: TOML document
↓
Structure Parsing
├── Parse table hierarchy
├── Detect array tables
├── Extract type information
└── Identify sections
↓
Purpose Detection
├── Project configuration (Cargo.toml, pyproject.toml)
├── Application settings
├── CI/CD configuration
└── Tool configuration
↓
Bridge Creation
├── Dependencies → code imports
├── Build configs → artifacts
├── Tool settings → behavior
└── Metadata → documentation
```

## Implementation Details

### Unified Parser Interface
```python
class StructuredTextParser:
    """Base class for structured text parsing."""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse file and extract structure."""
        content = file_path.read_text()
        
        # Detect file type and purpose
        file_type = self._detect_file_type(file_path)
        purpose = self._classify_purpose(content, file_type)
        
        # Extract structure based on type
        structure = self._extract_structure(content, file_type)
        
        # Detect bridges
        bridges = self._detect_bridges(structure, purpose)
        
        return {
            'content': content,
            'file_type': file_type,
            'purpose': purpose,
            'structure': structure,
            'bridges': bridges,
            'metadata': self._extract_metadata(content, file_type)
        }
```

### Configuration Language Router
```python
def route_config_file(file_path: Path) -> Dict[str, Any]:
    """Route configuration files to appropriate parser."""
    
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    
    # Special case handling
    if name == 'package.json':
        return parse_npm_package(file_path)
    elif name == 'cargo.toml':
        return parse_cargo_manifest(file_path)
    elif name == 'pyproject.toml':
        return parse_python_project(file_path)
    elif name.endswith('compose.yml') or name.endswith('compose.yaml'):
        return parse_docker_compose(file_path)
    elif suffix in ['.yaml', '.yml']:
        # Check for OpenAPI/Swagger
        if is_openapi_spec(file_path):
            return parse_openapi(file_path)
        return parse_yaml_config(file_path)
    
    # Default parsing
    return parse_generic_config(file_path)
```

### Schema-Aware Processing
```python
class SchemaAwareParser:
    """Parse with schema validation and type inference."""
    
    def __init__(self):
        self.schema_cache = {}
        self.type_inference = TypeInferenceEngine()
    
    def parse_with_schema(self, content: str, schema_path: Optional[str] = None):
        """Parse content with optional schema validation."""
        
        # Auto-detect schema if not provided
        if not schema_path:
            schema_path = self._detect_schema(content)
        
        # Load and cache schema
        if schema_path:
            schema = self._load_schema(schema_path)
            self._validate_against_schema(content, schema)
        
        # Extract typed structure
        structure = self._parse_structure(content)
        
        # Infer types for untyped fields
        typed_structure = self.type_inference.infer_types(structure)
        
        return typed_structure
```

### Bridge Detection Patterns

#### XML → Code Bridges
- Element names → Class names
- Attribute names → Method parameters
- Namespace URIs → Package imports
- XSD types → Data classes

#### YAML/JSON → Code Bridges
- Config keys → Constructor parameters
- API paths → Route handlers
- Schema definitions → Model classes
- Environment variables → Runtime config

#### LaTeX → Code Bridges
- Algorithm environments → Function implementations
- Equation labels → Variable names
- Package imports → Library dependencies
- Citation keys → Reference implementations

## Integration with Jina v4

### Structured Embedding Generation
```python
def generate_structured_embeddings(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate embeddings that preserve structure."""
    
    # Full document embedding
    full_embedding = jina_v4.encode(parsed_data['content'])
    
    # Section-level embeddings
    section_embeddings = {}
    for section_id, section_content in parsed_data['structure'].items():
        # Add structural context
        contextualized = f"[{parsed_data['file_type']}:{section_id}] {section_content}"
        section_embeddings[section_id] = jina_v4.encode(contextualized)
    
    # Bridge-aware embeddings
    bridge_embeddings = {}
    for bridge in parsed_data['bridges']:
        bridge_context = f"{bridge.source} -> {bridge.target}: {bridge.relationship}"
        bridge_embeddings[bridge.id] = jina_v4.encode(bridge_context)
    
    return {
        'full': full_embedding,
        'sections': section_embeddings,
        'bridges': bridge_embeddings
    }
```

### Chunking Strategies

#### Hierarchical Chunking
- Respect natural boundaries (sections, objects, tables)
- Maintain parent-child relationships
- Preserve context through path notation

#### Reference-Aware Chunking
- Keep references with their targets
- Inline resolved references when needed
- Maintain bidirectional links

#### Schema-Driven Chunking
- Use schema to determine boundaries
- Group related fields
- Respect required/optional distinctions

## Theory-Practice Examples

### Example 1: OpenAPI → Implementation
```yaml
# api.yaml
paths:
  /users/{id}:
    get:
      operationId: getUser
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
```

Bridges to:
```python
# users.py
@app.get("/users/{id}")
def get_user(id: int):  # Bridge: operationId -> function name
    ...
```

### Example 2: LaTeX Algorithm → Code
```latex
\begin{algorithm}
\caption{Flow-based Pruning}
\label{alg:flow-pruning}
\begin{algorithmic}
\State $R \gets \text{InitResources}(G, S)$
\For{$i = 1$ to $MaxIter$}
    \State $R \gets \text{Propagate}(R, \alpha)$
\EndFor
\end{algorithmic}
\end{algorithm}
```

Bridges to:
```python
def flow_based_pruning(graph, start_nodes, alpha, max_iter):
    resources = init_resources(graph, start_nodes)
    for i in range(max_iter):
        resources = propagate(resources, alpha)
    return resources
```

### Example 3: Config → Runtime Behavior
```toml
# pyproject.toml
[tool.hades.pathrag]
flow_decay_factor = 0.8
pruning_threshold = 0.3
max_iterations = 10
```

Bridges to:
```python
# pathrag_config.py
class PathRAGConfig:
    flow_decay_factor: float = 0.8  # From pyproject.toml
    pruning_threshold: float = 0.3
    max_iterations: int = 10
```

## Performance Considerations

1. **Lazy Parsing**: Parse structure on-demand for large files
2. **Schema Caching**: Cache parsed schemas for validation
3. **Incremental Updates**: Track changes in structured files
4. **Batch Processing**: Group similar files for efficiency

## Future Enhancements

1. **Language Server Protocol**: Real-time config validation
2. **GraphQL Schema**: Support for GraphQL SDL files
3. **Protocol Buffers**: Binary format support
4. **Custom DSLs**: Extensible parser framework