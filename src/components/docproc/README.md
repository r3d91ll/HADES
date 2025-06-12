# Document Processing Components

## Overview

The Document Processing (DocProc) components provide a unified interface for processing various document formats in the HADES system. These components implement the `DocumentProcessor` protocol to extract content, metadata, and structural information from documents while maintaining consistency across different file types.

## Architecture

### Component Structure

```text
src/components/docproc/
├── __init__.py           # Package exports and documentation
├── core/                 # Core document processor
│   ├── __init__.py
│   └── processor.py      # Multi-format adapter-based processor
├── docling/              # Docling-based processor
│   ├── __init__.py
│   └── processor.py      # Complex document handling (PDF, Word, etc.)
└── factory.py            # Component factory and registry
```

### Protocol Implementation

All document processors implement the `DocumentProcessor` protocol:

```python
class DocumentProcessor(Protocol):
    """Protocol for document processing components."""
    
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """Process documents according to the contract."""
        ...
    
    def process_single(self, file_path: Union[str, Path], 
                      content: Optional[Union[str, bytes]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """Process a single document."""
        ...
    
    def detect_category(self, file_path: Union[str, Path], 
                       content: Optional[Union[str, bytes]] = None) -> ContentCategory:
        """Detect document category."""
        ...
```

## Available Components

### 1. Core Document Processor

**Purpose**: Handles common document formats using format-specific adapters

**Supported Formats**:

- **Code**: Python (.py), JavaScript (.js), TypeScript (.ts), Java (.java), C/C++ (.c, .cpp, .h, .hpp)
- **Data**: JSON (.json), YAML (.yaml, .yml), XML (.xml)
- **Text**: Markdown (.md), Plain text (.txt), HTML (.html)

**Key Features**:

- Automatic format detection based on file extension and content
- Adapter-based architecture for extensibility
- Metadata extraction (file info, content statistics)
- Schema validation for structured formats
- Batch processing support

**Configuration**:

```yaml
# src/config/components/docproc/core/config.yaml
format_detection:
  enabled: true
  confidence_threshold: 0.8
  fallback_adapter: "markdown"

processing:
  batch_size: 10
  max_file_size_mb: 100
  timeout_seconds: 60
  enable_caching: true
```

### 2. Docling Document Processor

**Purpose**: Specializes in complex document formats requiring advanced parsing

**Supported Formats**:

- PDF documents (.pdf)
- Word documents (.docx)
- PowerPoint presentations (.pptx) [if enabled]
- Excel spreadsheets (.xlsx) [if enabled]

**Key Features**:

- Layout-aware text extraction
- Table and figure recognition
- Metadata preservation (author, creation date, etc.)
- Multi-page document handling
- OCR support for scanned documents [if configured]

**Configuration**:

```yaml
# src/config/components/docproc/docling/config.yaml
docling:
  enable_ocr: false
  extract_tables: true
  extract_figures: true
  preserve_formatting: false
  
pdf:
  extract_method: "pdfplumber"  # or "pypdf2", "pdfminer"
  extract_images: false
  max_pages: 1000
```

## Usage

### Basic Usage

```python
from src.components.docproc import create_docproc_component

# Create a core processor for general documents
processor = create_docproc_component("core")

# Process a single file
result = processor.process_single("document.py")
print(f"Content: {result.content[:100]}...")
print(f"Category: {result.metadata.content_category}")
print(f"Lines: {result.metadata.content_metadata.get('line_count')}")
```

### Batch Processing

```python
from src.types.components.contracts import DocumentProcessingInput

# Create input for batch processing
input_data = DocumentProcessingInput(
    file_paths=["file1.py", "file2.json", "file3.md"],
    options={"extract_metadata": True}
)

# Process batch
output = processor.process(input_data)

for doc in output.processed_documents:
    print(f"{doc.metadata.file_path}: {doc.metadata.content_category}")
```

### Advanced Configuration

```python
# Create processor with custom configuration
config = {
    "format_detection": {
        "enabled": True,
        "confidence_threshold": 0.9
    },
    "processing": {
        "batch_size": 20,
        "enable_caching": False
    }
}

processor = create_docproc_component("core", config=config)
```

### Format-Specific Processing

```python
# Use Docling for PDF processing
pdf_processor = create_docproc_component("docling", config={
    "docling": {
        "extract_tables": True,
        "extract_figures": True
    }
})

pdf_result = pdf_processor.process_single("report.pdf")
tables = pdf_result.metadata.content_metadata.get("tables", [])
```

## Component Selection Guide

Choose the appropriate processor based on your document types:

| Document Type | Recommended Component | Reason |
|--------------|----------------------|---------|
| Source code | Core | Optimized for code structure and syntax |
| JSON/YAML | Core | Schema validation and structure preservation |
| Markdown/Text | Core | Efficient text processing with metadata |
| PDF | Docling | Advanced layout extraction and table recognition |
| Word/Excel | Docling | Native format support with formatting preservation |

## Integration with Other Components

### Chunking Pipeline

Document processors integrate seamlessly with chunking components:

```python
# Process document first
doc_processor = create_docproc_component("core")
processed_doc = doc_processor.process_single("large_file.py")

# Then chunk based on document type
from src.components.chunking import create_chunking_component

if processed_doc.metadata.content_category == ContentCategory.CODE:
    chunker = create_chunking_component("code")
else:
    chunker = create_chunking_component("text")

chunks = chunker.chunk(processed_doc.content)
```

### Embedding Pipeline

Document metadata enhances embedding quality:

```python
# Process document with metadata
processed_doc = doc_processor.process_single("technical_doc.md")

# Use metadata for embedding context
embedding_input = {
    "text": processed_doc.content,
    "metadata": {
        "category": processed_doc.metadata.content_category,
        "language": processed_doc.metadata.content_metadata.get("language"),
        "format": processed_doc.metadata.format
    }
}
```

## Performance Considerations

### Caching

The core processor includes an LRU cache for repeated document processing:

```python
# Enable caching (default)
processor = create_docproc_component("core", config={
    "processing": {
        "enable_caching": True,
        "cache_size": 100  # Number of documents to cache
    }
})
```

### Performance Considerations Batch Processing

For multiple documents, use batch processing for better performance:

```python
# Process 100 files in batches of 20
files = ["file1.py", "file2.py", ...]  # 100 files

processor = create_docproc_component("core", config={
    "processing": {"batch_size": 20}
})

# Processes in 5 batches internally
output = processor.process(DocumentProcessingInput(file_paths=files))
```

### Memory Management

For large documents:

```python
# Configure size limits
processor = create_docproc_component("core", config={
    "processing": {
        "max_file_size_mb": 50,  # Skip files larger than 50MB
        "timeout_seconds": 30     # Timeout long-running processing
    }
})
```

## Error Handling

Document processors provide detailed error information:

```python
output = processor.process(input_data)

if output.errors:
    for error in output.errors:
        print(f"Error: {error}")

# Check individual document status
for doc in output.processed_documents:
    if not doc.success:
        print(f"Failed to process {doc.metadata.file_path}: {doc.error}")
```

## Extending the System

### Adding New Adapters

1. Implement the adapter interface in the core processor
2. Register the adapter in the format mapping
3. Update configuration with file extensions

### Creating Custom Processors

```python
from src.types.components.protocols import DocumentProcessor

class CustomDocumentProcessor(DocumentProcessor):
    """Custom processor for specialized formats."""
    
    def process(self, input_data):
        # Custom implementation
        pass

# Register the processor
from src.components.docproc.factory import register_docproc_component

register_docproc_component("custom", CustomDocumentProcessor)
```

## Configuration Reference

See configuration files for detailed options:

- Core: `src/config/components/docproc/core/config.yaml`
- Docling: `src/config/components/docproc/docling/config.yaml`

## Testing

Run component tests:

```bash
pytest test/components/docproc/
```

## Dependencies

- **Core**: No external dependencies (uses standard library)
- **Docling**: Requires `docling` package for PDF/Word processing

## Related Documentation

- [Component Architecture](../README.md)
- [Chunking Components](../chunking/README.md)
- [Type Contracts](../../types/components/README.md)
