#!/usr/bin/env python3
"""
PDF Text Extraction using HADES Document Processing

This script demonstrates using HADES' own document processing components
to extract text from PDFs and other documents. This functionality would
make an excellent MCP (Model Context Protocol) server for AI assistants.

Future MCP server idea:
- Expose document processing as MCP tools
- Allow AI assistants to read PDFs, Word docs, etc.
- Extract structured content with metadata
- Handle complex documents with tables, images, etc.

Usage:
    python scripts/extract_pdf_text.py <pdf_path> [--output <output_path>]
    
Example:
    python scripts/extract_pdf_text.py docs/paper.pdf --output docs/paper_extracted.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.components.docproc.factory import create_docproc_component


def extract_document_text(
    file_path: Path, 
    output_path: Optional[Path] = None,
    processor_type: str = 'docling'
) -> Dict[str, Any]:
    """
    Extract text from a document using HADES docproc.
    
    Args:
        file_path: Path to the document
        output_path: Optional path to save extracted text
        processor_type: Type of processor to use ('docling' or 'core')
        
    Returns:
        Dictionary with extraction results
    """
    # Create document processor
    processor = create_docproc_component(processor_type)
    
    # Process the document
    print(f"Processing {file_path} with {processor_type} processor...")
    result = processor.process_document(str(file_path))
    
    # Extract the text content
    text_content = ""
    metadata = {}
    
    if hasattr(result, 'content'):
        text_content = result.content
        if hasattr(result, 'metadata'):
            metadata = result.metadata
    elif isinstance(result, dict):
        text_content = result.get('content', result.get('text', str(result)))
        metadata = result.get('metadata', {})
    else:
        text_content = str(result)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Extracted text saved to: {output_path}")
    
    return {
        'content': text_content,
        'metadata': metadata,
        'content_length': len(text_content),
        'processor_used': processor_type
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract text from PDFs and documents using HADES docproc'
    )
    parser.add_argument('file_path', help='Path to the document file')
    parser.add_argument(
        '--output', '-o',
        help='Output path for extracted text'
    )
    parser.add_argument(
        '--processor',
        choices=['docling', 'core'],
        default='docling',
        help='Document processor to use'
    )
    parser.add_argument(
        '--show-metadata',
        action='store_true',
        help='Display document metadata'
    )
    
    args = parser.parse_args()
    
    # Process paths
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    
    try:
        # Extract text
        result = extract_document_text(
            file_path,
            output_path,
            args.processor
        )
        
        print(f"\n✅ Extraction successful!")
        print(f"📄 Content length: {result['content_length']:,} characters")
        
        if args.show_metadata and result['metadata']:
            print(f"\n📋 Metadata:")
            for key, value in result['metadata'].items():
                print(f"   {key}: {value}")
        
        if not output_path:
            print(f"\n📝 Preview (first 500 chars):")
            print("-" * 60)
            print(result['content'][:500])
            print("-" * 60)
            print("\nUse --output to save full content to file")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Future MCP Server Implementation Ideas:

1. Basic document reading tools:
   - read_pdf(file_path) -> text content
   - read_docx(file_path) -> text content
   - extract_tables(file_path) -> structured table data
   
2. Advanced processing:
   - extract_sections(file_path) -> hierarchical document structure
   - extract_metadata(file_path) -> document properties
   - extract_citations(file_path) -> bibliography/references
   
3. Multi-modal extraction:
   - extract_images(file_path) -> image paths/descriptions
   - extract_equations(file_path) -> LaTeX/MathML
   - extract_code_blocks(file_path) -> code snippets with language
   
4. Batch processing:
   - process_directory(dir_path, pattern) -> multiple documents
   - compare_documents(file1, file2) -> similarity/diff
   
This would allow AI assistants to:
- Read research papers and technical documents
- Extract specific information from contracts/legal docs
- Process documentation and manuals
- Analyze code documentation
- Compare multiple documents

Integration with HADES:
- Use existing docproc components
- Leverage chunking for large documents
- Apply embeddings for semantic search
- Store in graph for relationship analysis
"""