# Document Processing Module (docproc)

## Type System Improvements (May 2025)

The document processing module has been updated with improved type annotations to ensure compatibility with mypy type checking. Key improvements include:

### Type Fixes Summary

- Added proper type annotations for entity extractors in the markdown adapter
- Fixed dictionary update operations in YAML and JSON adapters with appropriate type casts
- Resolved incompatible types in collection assignments and dictionary updates
- Added explicit casts for return values to match declared return types
- Improved error handling with proper type annotations for error responses

### Remaining Type Issues

Some type issues remain that would require further work:

- Missing type stubs for the YAML library (could be resolved by installing `types-PyYAML`)
- Some complex type incompatibilities in the YAML and JSON adapters related to TypedDict usage
- Return type issues in the schemas/utils.py module

### Type Annotation Guidelines

When making future changes to the document processing module, follow these guidelines:

1. Use explicit type annotations for all function parameters and return values
2. When working with collections, prefer concrete types like `Dict[str, Any]` over `Collection[str]`
3. Use `cast()` from the typing module when type inference isn't sufficient
4. For TypedDict classes like YAMLNodeInfo, ensure assignments maintain type compatibility
5. Run mypy periodically during development to catch type issues early

## Overview

The document processing module provides a unified interface for processing a wide range of document formats. It converts them to standardized formats for both RAG pipeline ingestion and direct model inference, supporting comprehensive document processing needs.

## Key Components

### Core Functionality (`core.py`)

The core module provides the primary interface for document processing:

- `process_document(file_path, options)` - Process a document file and convert to standardized format
- `process_text(text, format_type, options)` - Process text content directly with a specified format
- `detect_format(file_path)` - Detect the format of a document based on file extension and content
- `get_format_for_document(file_path)` - Get the format for a document, handling special cases
- `save_processed_document(document, output_path)` - Save a processed document to disk as JSON
- `process_documents_batch(file_paths, options)` - Process multiple documents in batch

### Document Processing Manager (`manager.py`)

The manager provides a high-level interface with a caching layer:

- `DocumentProcessorManager` - Class for managing document processing operations
  - `process_document(content, path, doc_type, options)` - Flexible interface to process content or files
  - `batch_process(paths, options)` - Process a batch of documents from paths
  - `get_adapter_for_doc_type(doc_type)` - Get the appropriate adapter with caching

### Format Adapters (`adapters/`)

Adapter implementations for specific document formats:

- `BaseAdapter` - Abstract base class for all format adapters
- `DoclingAdapter` - Processes multiple document formats via Docling, including:
  - Document formats: PDF, Markdown, Text, Word (DOCX/DOC), PowerPoint (PPTX/PPT), Excel (XLSX/XLS), HTML, XML, EPUB, RTF, ODT, CSV, JSON, YAML
  - Code formats: Python, JavaScript, TypeScript, Java, C++, C, Go, Ruby, PHP, C#, Rust, Swift, Kotlin, Scala, R, Shell scripts, Jupyter notebooks
- `MarkdownAdapter` - Specialized handling for Markdown files
- `PythonAdapter` - Code-aware processing for Python files

### Schemas (`schemas/`)

Document validation schemas:

- `BaseDocument` - Base Pydantic model for document validation
- `PythonDocument` - Specialized schema for Python code documents

### Utilities (`utils/`)

Supporting utilities:

- `format_detector.py` - Document format detection
- `metadata_extractor.py` - Extract metadata from documents
- `markdown_entity_extractor.py` - Extract entities from markdown content

## Usage Examples

### Basic Document Processing

```python
from src.docproc.core import process_document

# Process a file
result = process_document("/path/to/document.pdf")
print(f"Processed document: {result['id']}")
print(f"Content length: {len(result['content'])} characters")
print(f"Format: {result['format']}")
print(f"Metadata: {result['metadata']}")
```

### Using the Document Manager

```python
from src.docproc.manager import DocumentProcessorManager

# Initialize manager
manager = DocumentProcessorManager()

# Process from file path
doc1 = manager.process_document(path="/path/to/document.md")

# Process direct content
doc2 = manager.process_document(
    content="# Sample Markdown\nThis is some text content.",
    doc_type="markdown"
)

# Batch processing
documents = manager.batch_process([
    "/path/to/doc1.txt",
    "/path/to/doc2.pdf",
    "/path/to/code.py"
])
```

### Custom Processing Options

```python
from src.docproc.core import process_document

# Process with custom options
result = process_document(
    "/path/to/document.py",
    options={
        "extract_docstrings": True,
        "include_comments": True,
        "max_line_length": 120,
        "include_imports": True
    }
)
```

## Integration with Ingestion Pipeline

The document processing module serves as the first stage in the HADES-PathRAG ingestion pipeline:

1. **Document Processing** (docproc): Convert raw documents to standardized format
2. **Chunking** (chunking): Split documents into appropriate chunks
3. **Embedding** (embeddings): Generate embeddings for chunks
4. **Storage** (storage): Store chunks, embeddings, and relationships

## Extension Points

To add support for new document formats:

1. Create a new adapter class that extends `BaseAdapter`
2. Implement required methods: `process`, `extract_metadata`, `extract_entities`, and `process_text`
3. Register the adapter in the registry through the `register_adapter` function

## Testing

The module has a comprehensive test suite with over 85% coverage for key components:
- Unit tests for the manager (`tests/unit/docproc/test_manager.py`)
- Tests for processing functions and adapters
- Validation tests for document schemas

## Supported File Types

The document processing module now supports a comprehensive range of file types through the DoclingAdapter:

### Document Formats

- PDF (`.pdf`) - Full document processing with OCR support
- Microsoft Word (`.docx`, `.doc`)
- Microsoft PowerPoint (`.pptx`, `.ppt`)
- Microsoft Excel (`.xlsx`, `.xls`, `.csv`)
- Text (`.txt`)
- Markdown (`.md`, `.markdown`)
- HTML (`.html`, `.htm`)
- XML (`.xml`)
- E-books (`.epub`)
- OpenDocument (`.odt`, `.rtf`)
- Structured data (`.json`, `.yaml`, `.yml`)

### Code Formats

- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- Java (`.java`)
- C++ (`.cpp`)
- C (`.c`)
- Go (`.go`)
- Ruby (`.rb`)
- PHP (`.php`)
- C# (`.cs`)
- Rust (`.rs`)
- Swift (`.swift`)
- Kotlin (`.kt`)
- Scala (`.scala`)
- R (`.r`)
- Shell scripts (`.sh`)
- Jupyter notebooks (`.ipynb`)

## Future Improvements

- Enhance entity extraction with NER models
- Add parallel processing for large document batches
- Implement content summarization pre-processor
- Add more specialized format-specific features
