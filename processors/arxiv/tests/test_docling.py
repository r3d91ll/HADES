#!/usr/bin/env python3
"""
Test Docling PDF conversion without database operations
"""

import sys
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DocumentConversionInput
from docling.datamodel.pipeline_options import PipelineOptions, TableStructureOptions

def test_docling_conversion(pdf_path: str):
    """Test PDF to Markdown conversion"""
    print(f"Testing Docling with: {pdf_path}")
    
    # Check file exists
    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        return False
    
    try:
        # Configure pipeline options
        pipeline_options = PipelineOptions(
            do_ocr=False,  # Skip OCR for speed
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True
            )
        )
        
        # Create converter
        converter = DocumentConverter(
            pipeline_options=pipeline_options
        )
        
        print("Converting PDF to Markdown...")
        # Create input object
        doc_input = DocumentConversionInput.from_paths(paths=[pdf_path])
        
        # Convert - returns an iterator
        results = converter.convert(doc_input)
        
        # Get the first result
        result = next(results)
        
        # Export to markdown
        markdown = result.output.export_to_markdown()
        
        print(f"\nMarkdown preview (first 500 chars):")
        print("-" * 50)
        print(markdown[:500])
        print("-" * 50)
        
        print(f"\nTotal markdown length: {len(markdown)} characters")
        
        # Document stats
        if hasattr(result.output, 'pages'):
            print(f"Pages: {len(result.output.pages)}")
        if hasattr(result.output, 'tables'):
            print(f"Tables: {len(result.output.tables)}")
        if hasattr(result.output, 'pictures'):
            print(f"Pictures: {len(result.output.pictures)}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_docling.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    success = test_docling_conversion(pdf_path)
    sys.exit(0 if success else 1)