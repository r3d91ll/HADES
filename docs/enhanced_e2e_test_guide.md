# Enhanced End-to-End Test Script Guide

The `enhanced_end_to_end_test.py` script provides comprehensive testing of the HADES pipeline with multi-format support, debug mode, and specialized Python code processing.

## Key Features

### Multi-Format Document Support
- **Document formats**: PDF, Word (DOCX/DOC), PowerPoint (PPTX/PPT), Excel (XLSX/XLS), HTML, XML, EPUB, RTF, ODT, CSV
- **Text formats**: TXT, Markdown, JSON, YAML
- **Python code files**: Full AST processing with symbol table generation

### Python Code Processing
The script leverages the existing specialized methods built into each module:

1. **Document Processing**: Uses `PythonAdapter._process_python_file()` and `EntityExtractor` classes to generate AST symbol tables
2. **Chunking**: Uses `PythonCodeChunker.chunk()` and `chunk_python_code()` for AST-aware chunking that preserves code boundaries  
3. **Embedding**: Standard embedding adapters treat the AST-processed chunks as enhanced text
4. **ISNE Enhancement**: Uses `create_graph_from_documents()` which automatically processes AST relationships from Python files

### Debug Mode
Saves intermediate JSON files at each pipeline stage:
```
debug_output/
├── 01_docproc/           # Document processing output with AST symbols
├── 02_chunking/          # AST-aware chunked documents  
├── 03_embedding/         # Embedded chunks
└── 04_isne/             # ISNE-enhanced embeddings with code relationships
```

## Usage Examples

### Basic Multi-Format Processing
```bash
# Process all supported files in a directory
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output

# Enable debug mode for troubleshooting
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output --debug
```

### Python Code Analysis
```bash
# Process Python files with full AST analysis
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./src -o ./code-analysis --debug

# Focus only on chunking stage to see AST-based chunking
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./src -o ./code-analysis --stage-only chunking --debug
```

### Performance Testing
```bash
# Parallel processing with custom batch size
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./large-dataset -o ./performance-test \
  --parallel --max-workers 8 --batch-size 20

# Limit files for quick testing
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./quick-test \
  --max-files 5 --debug
```

### Comparison and Troubleshooting
```bash
# Compare with previous run
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./run2 \
  --debug --diff-with ./run1

# Test specific configuration changes
python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./chunking-test \
  --stage-only chunking --chunk-size 500 --chunk-overlap 50 --debug
```

## Python AST Processing Details

### Method-Specific Processing

**Document Processing (`PythonAdapter`)**:
- **`_process_python_file()`**: Core AST processing using Python's `ast` module
- **`EntityExtractor.visit_ClassDef()`**: Extracts class definitions with inheritance
- **`EntityExtractor.visit_FunctionDef()`**: Extracts functions with parameters and docstrings
- **`EntityExtractor.visit_Import*()`**: Processes import statements and dependencies
- **`CallFinder.visit_Call()`**: Identifies function calls for relationship building

**Chunking (`PythonCodeChunker`)**:
- **`chunk()`**: Main entry point for Python code chunking
- **`_extract_code_chunks()`**: Processes chunks from AST-analyzed code
- **`chunk_python_code()`**: AST-based chunking that preserves code boundaries
- **`_process_relationships()`**: Adds AST relationships to chunks

**ISNE Graph Construction**:
- **`create_graph_from_documents()`**: Automatically detects `.py` files and processes code relationships
- **Relationship Types**: CALLS, CONTAINS, EXTENDS based on AST analysis
- **Edge Weighting**: Adjusts weights based on embedding model and code structure

## Output Structure

### Debug Files
Each stage produces debug files with detailed information:

1. **01_docproc**: Raw document processing with AST symbols for Python files
2. **02_chunking**: Chunked documents with preserved code structure
3. **03_embedding**: Vector embeddings for each chunk
4. **04_isne**: Enhanced embeddings with graph relationships

### Statistics Files
- **Processing metrics**: Time, file counts, success rates
- **Stage statistics**: Documents processed per stage
- **Comparison results**: Differences between runs (if using --diff-with)

## Configuration Options

### Document Processing
- `python_ast_processing`: Enable AST analysis for Python files
- `generate_symbol_tables`: Create comprehensive symbol tables
- `code_analysis_depth`: Depth of AST analysis ("full" or "basic")

### Chunking
- `python_ast_chunking`: Use AST-aware chunking for Python files
- `preserve_function_boundaries`: Never split functions across chunks
- `preserve_class_boundaries`: Keep class definitions together

### ISNE Enhancement
- `use_ast_relationships`: Use AST symbol tables for graph relationships
- `symbol_table_enhanced_graphs`: Enhanced graph construction using code symbols
- `code_structure_weights`: Weight relationships based on code structure

This enhanced script provides a comprehensive testing and debugging framework for the HADES pipeline, with specialized support for Python code analysis through AST processing.