# Document Processing Module (docproc)

## Centralized Type System (May 2025)

The document processing module has been refactored to use a centralized type system that improves type safety, consistency, and maintainability. This update replaces the previous mixed type approach with a clean, comprehensive type system located in `src/types/docproc/`.

### Major Type System Improvements

- **Centralized Type Definitions**: All types are now defined in `src/types/docproc/` instead of scattered throughout the module
- **Consistent Type Usage**: Updated all function signatures to use the centralized types
- **Comprehensive TypedDict Definitions**: Created detailed TypedDict definitions for all data structures
- **Protocol Classes**: Added Protocol classes like `DocumentProcessorType` to define interfaces
- **Improved Error Handling**: Added specific result types for success/failure conditions
- **Type Safety for Processing Options**: Created structured types for processing options

### Type System Structure

The new type system is organized into the following files:

- `src/types/docproc/base.py`: Base document type classes
- `src/types/docproc/schema.py`: TypedDict schemas for document structures
- `src/types/docproc/document.py`: Document-specific type definitions
- `src/types/docproc/metadata.py`: Metadata and entity type definitions
- `src/types/docproc/adapter.py`: Adapter interface and configuration types
- `src/types/docproc/processing.py`: Types for document processing functions

### Core Type Definitions

1. **Document Types**:
   - `DocumentDict`: Standard document structure
   - `EntityDict`: Entity extracted from document
   - `MetadataDict`: Document metadata

2. **Processing Types**:
   - `ProcessingOptions`: Options for document processing
   - `ProcessingResult`: Result of document processing
   - `BatchProcessingResult`: Results from batch processing
   - `FormatDetectionResult`: Result of format detection

3. **Adapter Types**:
   - `DocumentProcessorType`: Protocol for document adapters
   - `AdapterConfig`: Configuration for adapters
   - `AdapterOptions`: Processing options for adapters
   - `ProcessedDocument`: Standard output format

### Migration Status

All key components have been updated to use the new type system:

- ✅ Core processing functions in `core.py`
- ✅ Base adapter interface in `adapters/base.py`
- ✅ Adapter registry in `adapters/registry.py`
- ✅ Adapter selector in `adapters/adapter_selector.py`
- ✅ Document manager in `manager.py`

Some remaining work:

- Individual adapter implementations need to be updated
- Schema utility functions need further refinement
- Unit tests need to be updated to use the new types

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
