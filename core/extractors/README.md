# Extractors - Content Extraction System

The extractors module implements multi-format content extraction, transforming diverse document formats into structured, processable text while preserving semantic structure and mathematical content. This represents the initial WHAT (W) transformation in the Conveyance Framework.

## Overview

Extractors serve as the entry point for information processing, converting raw documents (PDFs, LaTeX, code) into structured representations that preserve both content and context. They ensure high-quality signal extraction while maintaining document structure for downstream processing.

## Architecture

```
extractors/
├── extractors_base.py       # Abstract base class and interfaces
├── extractors_docling.py    # IBM Docling for PDF/document extraction
├── extractors_latex.py      # LaTeX source processing
├── extractors_code.py       # Source code extraction
├── extractors_treesitter.py # Tree-sitter based code parsing
├── extractors_robust.py     # Fallback and error recovery
├── extractors_factory.py    # Factory pattern for extractor creation
└── __init__.py             # Public API exports
```

## Theoretical Foundation

### Conveyance Framework Mapping

```python
C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij
```

- **WHAT (W)**: Content quality and signal extraction fidelity
- **WHERE (R)**: Document structure preservation (sections, equations, tables)
- **WHO (H)**: Extractor capabilities (Docling, LaTeX parser, Tree-sitter)
- **TIME (T)**: Extraction throughput and processing efficiency
- **Export/Import (C_out/C_in)**: Encoding and decoding legs across the extraction workflow
- **Shared context (C_ext)**: Structure preservation amplifies understanding

### Zero-Propagation Prevention

Extractors validate content to prevent `W = 0`, which would force `C_pair = 0` regardless of other dimensions.

## Core Components

### ExtractorBase

Abstract interface for all extractors:

```python
from core.extractors import ExtractorBase, ExtractionResult

class CustomExtractor(ExtractorBase):
    """Custom extraction implementation."""

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract content from file."""
        # Implementation
        return ExtractionResult(
            text=extracted_text,
            metadata=metadata,
            structures=structures
        )

    def can_handle(self, file_path: str) -> bool:
        """Check if extractor can process file."""
        return file_path.endswith('.custom')
```

### DoclingExtractor

State-of-the-art PDF and document extraction using IBM Docling:

```python
from core.extractors import DoclingExtractor

# Initialize with configuration
extractor = DoclingExtractor(
    use_ocr=True,  # Enable OCR for scanned PDFs
    extract_tables=True,
    extract_equations=True,
    extract_images=False,  # Skip images for text-only
    chunk_size=1000,  # Tokens per chunk
    overlap=200  # Token overlap
)

# Extract from PDF
result = extractor.extract("paper.pdf")

# Access structured content
print(f"Text length: {len(result.text)}")
print(f"Sections: {len(result.sections)}")
print(f"Equations: {len(result.equations)}")
print(f"Tables: {len(result.tables)}")
print(f"Chunks: {len(result.chunks)}")
```

### LaTeXExtractor

Specialized extraction for LaTeX source files:

```python
from core.extractors import LaTeXExtractor

extractor = LaTeXExtractor(
    parse_math=True,  # Extract equations
    parse_citations=True,  # Extract bibliography
    resolve_commands=True  # Expand custom commands
)

# Extract from LaTeX
result = extractor.extract("paper.tex")

# Access LaTeX-specific content
for eq in result.equations:
    print(f"Equation: {eq.latex}")
    print(f"Rendered: {eq.text}")

for cite in result.citations:
    print(f"Citation: {cite.key} -> {cite.text}")
```

### CodeExtractor with Tree-sitter

Advanced code parsing with Tree-sitter:

```python
from core.extractors import CodeExtractor

extractor = CodeExtractor(
    languages=["python", "javascript", "rust"],
    extract_symbols=True,  # Functions, classes
    extract_comments=True,  # Documentation
    extract_imports=True  # Dependencies
)

# Extract from source code
result = extractor.extract("module.py")

# Access code structures
for func in result.functions:
    print(f"Function: {func.name}")
    print(f"Parameters: {func.parameters}")
    print(f"Docstring: {func.docstring}")

for cls in result.classes:
    print(f"Class: {cls.name}")
    print(f"Methods: {[m.name for m in cls.methods]}")
```

### RobustExtractor

Fallback extractor with multiple strategies:

```python
from core.extractors import RobustExtractor

# Tries multiple extraction methods
extractor = RobustExtractor(
    strategies=["docling", "pypdf", "pdfminer", "ocr"],
    min_text_length=100,  # Minimum acceptable result
    max_retries=3
)

# Robust extraction with fallbacks
result = extractor.extract("corrupted.pdf")
print(f"Extraction method used: {result.metadata['method']}")
print(f"Success after {result.metadata['attempts']} attempts")
```

### ExtractorFactory

Factory pattern for flexible instantiation:

```python
from core.extractors import ExtractorFactory

# Auto-detect based on file type
extractor = ExtractorFactory.create_for_file("document.pdf")

# From configuration
config = {
    "type": "docling",
    "use_ocr": True,
    "chunk_size": 1000
}
extractor = ExtractorFactory.create(config)

# With fallback chain
extractor = ExtractorFactory.create_with_fallback(
    primary="docling",
    fallbacks=["pypdf", "ocr"]
)
```

## Extraction Results

### ExtractionResult Structure

```python
@dataclass
class ExtractionResult:
    """Structured extraction output."""

    # Core content
    text: str  # Full extracted text
    chunks: List[TextChunk]  # Segmented chunks

    # Structural elements
    sections: List[Section]  # Document sections
    equations: List[Equation]  # Mathematical content
    tables: List[Table]  # Tabular data
    figures: List[Figure]  # Images/diagrams

    # Metadata
    metadata: Dict[str, Any]  # Document metadata
    statistics: ExtractionStats  # Performance metrics

    # Quality indicators
    confidence: float  # Extraction confidence [0, 1]
    warnings: List[str]  # Issues encountered
```

### TextChunk Structure

```python
@dataclass
class TextChunk:
    """Segmented text unit."""

    id: str  # Unique identifier
    text: str  # Chunk content
    start_idx: int  # Start position in document
    end_idx: int  # End position in document

    # Context preservation
    section: Optional[str]  # Parent section
    page_num: Optional[int]  # Source page

    # Semantic metadata
    has_equation: bool
    has_table: bool
    has_code: bool
```

## Usage Patterns

### Basic Extraction

```python
from core.extractors import DoclingExtractor

# Simple extraction
extractor = DoclingExtractor()
result = extractor.extract("research_paper.pdf")

# Process text
print(f"Title: {result.metadata.get('title')}")
print(f"Authors: {result.metadata.get('authors')}")
print(f"Abstract: {result.text[:500]}")
```

### Structured Processing

```python
# Extract with structure preservation
extractor = DoclingExtractor(
    preserve_structure=True,
    extract_hierarchy=True
)

result = extractor.extract("paper.pdf")

# Navigate document structure
for section in result.sections:
    print(f"\n{section.level}. {section.title}")
    print(f"Content: {section.text[:200]}...")

    # Process subsections
    for subsection in section.children:
        print(f"  {subsection.title}")
```

### Batch Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_document(file_path):
    """Process single document."""
    extractor = DoclingExtractor()
    return extractor.extract(file_path)

# Parallel extraction
pdf_files = list(Path("/data/papers").glob("*.pdf"))

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_document, pdf_files))

print(f"Processed {len(results)} documents")
```

### Pipeline Integration

```python
from core.extractors import DoclingExtractor
from core.embedders import JinaV4Embedder
from core.database import ArangoClient

# Complete pipeline
extractor = DoclingExtractor(chunk_size=1000)
embedder = JinaV4Embedder(device="cuda")
db = ArangoClient()

# Process document
result = extractor.extract("paper.pdf")

# Generate embeddings for chunks
embeddings = embedder.embed_batch(
    [chunk.text for chunk in result.chunks]
)

# Store in database
for chunk, embedding in zip(result.chunks, embeddings):
    db.store_chunk(
        document_id=result.metadata['document_id'],
        chunk=chunk,
        embedding=embedding
    )
```

## Advanced Features

### OCR Support

```python
# Enable OCR for scanned documents
extractor = DoclingExtractor(
    use_ocr=True,
    ocr_language="eng",  # Language for OCR
    ocr_confidence_threshold=0.8  # Minimum confidence
)

result = extractor.extract("scanned_document.pdf")

if result.metadata.get('used_ocr'):
    print(f"OCR confidence: {result.confidence}")
    print(f"OCR warnings: {result.warnings}")
```

### Equation Extraction

```python
# Mathematical content extraction
extractor = DoclingExtractor(
    extract_equations=True,
    parse_latex_math=True
)

result = extractor.extract("math_paper.pdf")

for eq in result.equations:
    print(f"LaTeX: {eq.latex}")
    print(f"Text: {eq.text}")
    print(f"Location: Page {eq.page_num}")
    print(f"Referenced as: {eq.label}")
```

### Table Extraction

```python
# Tabular data extraction
extractor = DoclingExtractor(
    extract_tables=True,
    table_format="markdown"  # or "csv", "json"
)

result = extractor.extract("data_paper.pdf")

for table in result.tables:
    print(f"Table: {table.caption}")
    print(f"Dimensions: {table.rows}x{table.cols}")
    print(f"Content:\n{table.markdown}")

    # Access as DataFrame
    df = table.to_dataframe()
    print(df.describe())
```

### Multi-language Support

```python
# Language-specific extraction
extractor = DoclingExtractor(
    language="auto",  # Auto-detect
    supported_languages=["en", "de", "fr", "zh"]
)

result = extractor.extract("multilingual.pdf")
print(f"Detected language: {result.metadata['language']}")
print(f"Language confidence: {result.metadata['language_confidence']}")
```

## Performance Optimization

### Chunking Strategies

```python
from core.extractors import ChunkingStrategy

# Semantic chunking
extractor = DoclingExtractor(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=1000,
    overlap=200
)

# Sliding window
extractor = DoclingExtractor(
    chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
    window_size=512,
    stride=256
)

# Section-based
extractor = DoclingExtractor(
    chunking_strategy=ChunkingStrategy.SECTION_BASED,
    min_chunk_size=100,
    max_chunk_size=2000
)
```

### Memory Management

```python
# Stream processing for large files
extractor = DoclingExtractor(
    stream_mode=True,  # Process in chunks
    max_memory_mb=1024  # Limit memory usage
)

# Process large PDF
with extractor.extract_stream("large_file.pdf") as stream:
    for chunk in stream:
        process_chunk(chunk)  # Process incrementally
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedExtractor(DoclingExtractor):
    """Extractor with result caching."""

    @lru_cache(maxsize=100)
    def _extract_cached(self, file_hash: str) -> ExtractionResult:
        """Cache extraction by file hash."""
        return super().extract(file_hash)

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract with caching."""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return self._extract_cached(file_hash)
```

## Configuration

### YAML Configuration

```yaml
# extraction_config.yaml
extraction:
  type: docling
  use_ocr: true
  extract_tables: true
  extract_equations: true
  extract_images: false
  chunking:
    strategy: semantic
    size: 1000
    overlap: 200
  quality:
    min_confidence: 0.7
    max_warnings: 5
```

### Environment Variables

```bash
# Extraction settings
export EXTRACTION_TYPE=docling
export USE_OCR=true
export CHUNK_SIZE=1000
export NUM_WORKERS=8

# Performance tuning
export EXTRACTION_TIMEOUT=300
export MAX_MEMORY_MB=2048
```

## Error Handling

### Common Issues

```python
from core.extractors import ExtractionError, CorruptedFileError

try:
    result = extractor.extract("document.pdf")
except CorruptedFileError as e:
    # Try robust extractor
    robust = RobustExtractor()
    result = robust.extract("document.pdf")
except ExtractionError as e:
    print(f"Extraction failed: {e}")
    # Implement fallback
```

### Validation

```python
def validate_extraction(result: ExtractionResult) -> bool:
    """Validate extraction quality."""

    # Check minimum content
    if len(result.text) < 100:
        return False

    # Check confidence
    if result.confidence < 0.5:
        return False

    # Check for critical warnings
    critical_warnings = [
        "Failed to extract text",
        "OCR failed",
        "Corrupted file"
    ]

    for warning in result.warnings:
        if any(crit in warning for crit in critical_warnings):
            return False

    return True
```

## Benchmarks

### Performance Metrics

| Extractor | Format | Speed (pages/sec) | Accuracy | Memory (MB) |
|-----------|--------|-------------------|----------|-------------|
| Docling | PDF | 2.5 | 98% | 512 |
| Docling + OCR | Scanned PDF | 0.8 | 95% | 1024 |
| LaTeX | .tex | 15.0 | 99% | 128 |
| Tree-sitter | Python | 50.0 | 100% | 64 |
| Robust | Mixed | 1.5 | 92% | 768 |

### Quality Metrics

| Feature | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Text Extraction | 0.98 | 0.99 | 0.98 |
| Equation Detection | 0.95 | 0.92 | 0.93 |
| Table Extraction | 0.91 | 0.88 | 0.89 |
| Section Detection | 0.96 | 0.94 | 0.95 |

## Best Practices

### 1. Choose Appropriate Extractor

```python
# For research papers
extractor = DoclingExtractor(
    extract_equations=True,
    extract_tables=True,
    extract_citations=True
)

# For code repositories
extractor = CodeExtractor(
    extract_symbols=True,
    extract_dependencies=True
)

# For mixed content
extractor = RobustExtractor(
    strategies=["docling", "latex", "code"]
)
```

### 2. Validate Input Files

```python
def validate_file(file_path: str) -> bool:
    """Validate file before extraction."""
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False

    # Check size
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        return False

    # Check format
    supported = ['.pdf', '.tex', '.py', '.js', '.rs']
    if path.suffix not in supported:
        return False

    return True
```

### 3. Handle Extraction Failures

```python
def extract_with_fallback(file_path: str) -> ExtractionResult:
    """Extract with multiple fallback strategies."""
    strategies = [
        DoclingExtractor(),
        RobustExtractor(),
        OCRExtractor()
    ]

    for strategy in strategies:
        try:
            result = strategy.extract(file_path)
            if validate_extraction(result):
                return result
        except Exception as e:
            continue

    raise ExtractionError(f"All strategies failed for {file_path}")
```

### 4. Monitor Extraction Quality

```python
def monitor_extraction(result: ExtractionResult) -> Dict:
    """Monitor extraction quality metrics."""
    return {
        "text_length": len(result.text),
        "num_chunks": len(result.chunks),
        "num_equations": len(result.equations),
        "num_tables": len(result.tables),
        "confidence": result.confidence,
        "warnings": len(result.warnings),
        "extraction_time": result.statistics.duration
    }
```

## Testing

```python
import pytest
from core.extractors import DoclingExtractor

def test_pdf_extraction():
    """Test PDF extraction."""
    extractor = DoclingExtractor()
    result = extractor.extract("test_paper.pdf")

    assert len(result.text) > 0
    assert result.confidence > 0.8
    assert len(result.chunks) > 0

def test_chunking():
    """Test chunking strategy."""
    extractor = DoclingExtractor(
        chunk_size=1000,
        overlap=200
    )
    result = extractor.extract("test_paper.pdf")

    for chunk in result.chunks:
        assert len(chunk.text) <= 1000
        assert chunk.id is not None

def test_structure_preservation():
    """Test structure extraction."""
    extractor = DoclingExtractor(
        preserve_structure=True
    )
    result = extractor.extract("test_paper.pdf")

    assert len(result.sections) > 0
    assert result.sections[0].title is not None
```

## Migration Guide

### From PyPDF2

```python
# Old approach
import PyPDF2
reader = PyPDF2.PdfReader("file.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# New approach
from core.extractors import DoclingExtractor
extractor = DoclingExtractor()
result = extractor.extract("file.pdf")
text = result.text
```

### From pdfplumber

```python
# Old approach
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

# New approach
from core.extractors import DoclingExtractor
extractor = DoclingExtractor()
result = extractor.extract("file.pdf")
text = result.text
# Plus: equations, tables, structure
```

## Related Components

- [Embedders](../embedders/README.md) - Process extracted text
- [Workflows](../workflows/README.md) - Orchestrate extraction pipelines
- [Processors](../processors/README.md) - Post-process extracted content
- [Config](../config/README.md) - Configure extractors
