# File Type Detection Module

## Overview

The file type detection module is responsible for identifying file formats and categorizing them into broader content categories (code or text). This enables the document processing pipeline to route files to the appropriate specialized processing paths.

## Content Categories

Files are categorized into two main categories:

1. **Code Files** - Processed with AST-based chunkers:
   - Python (.py)
   - JSON (.json)
   - YAML (.yaml, .yml)
   - XML (.xml) 
   - TOML (.toml)

2. **Text Files** - Processed with docling and Chonky chunker:
   - Markdown (.md, .markdown)
   - PDF (.pdf)
   - CSV (.csv)
   - Plain text (.txt)
   - Office documents (.docx, .xlsx, .pptx)

## Implementation Details

The implementation enhances the document processing pipeline by:

1. Adding content category configuration in `preprocessor_config.yaml`
2. Implementing category detection in `format_detector.py`
3. Incorporating content category metadata in processed documents
4. Providing utility functions to determine appropriate chunking strategies

## Usage

The document processor automatically detects and categorizes files. The content category is included in the document metadata:

```json
{
  "id": "doc_12345",
  "path": "/path/to/file.py",
  "format": "python",
  "content_category": "code",
  "metadata": {
    "file_type": "python",
    "content_category": "code",
    "...": "..."
  }
}
```

## Configuration

File type mappings and categories are defined in `src/config/preprocessor_config.yaml`. To add support for new file formats:

1. Add the file extension to the appropriate format in `file_type_map`
2. Add the format to the appropriate category in `content_categories`

## Integration with ISNE

The file type and content category metadata is preserved throughout the pipeline and passed to the ISNE model, enabling proper handling of mixed file types during training.
