#!/usr/bin/env python3
"""
Test how Docling handles mathematical equations in PDFs.
"""

import os
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DocumentConversionInput
from docling.datamodel.pipeline_options import PipelineOptions, TableStructureOptions
from docling.datamodel.base_models import AssembleOptions

def analyze_equation_handling(pdf_path: str):
    """Analyze how Docling processes equations in a PDF."""
    
    print(f"Analyzing: {pdf_path}")
    print("=" * 70)
    
    # Configure pipeline options
    pipeline_options = PipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True
        )
    )
    
    # Try different AssembleOptions
    assemble_options = AssembleOptions(
        keep_page_images=True,
        images_scale=1.0  # Full resolution
    )
    
    # Create converter
    converter = DocumentConverter(
        pipeline_options=pipeline_options,
        assemble_options=assemble_options
    )
    
    # Convert document
    doc_input = DocumentConversionInput.from_paths(paths=[pdf_path])
    results = converter.convert(doc_input)
    result = next(results)
    
    # Analyze output structure
    print("\nDocument structure:")
    print(f"  Has pictures: {hasattr(result.output, 'pictures') and bool(result.output.pictures)}")
    print(f"  Has tables: {hasattr(result.output, 'tables') and bool(result.output.tables)}")
    print(f"  Has equations: {hasattr(result.output, 'equations') and bool(result.output.equations)}")
    print(f"  Has formulas: {hasattr(result.output, 'formulas') and bool(result.output.formulas)}")
    
    # Check for any math-related attributes
    print("\nChecking for math-related attributes:")
    for attr in dir(result.output):
        if any(term in attr.lower() for term in ['math', 'equation', 'formula', 'latex']):
            print(f"  Found: {attr}")
            value = getattr(result.output, attr, None)
            if value:
                print(f"    Type: {type(value)}")
                print(f"    Value: {str(value)[:100]}...")
    
    # Export to markdown
    markdown = result.output.export_to_markdown()
    
    # Look for equation patterns in markdown
    print("\nMarkdown analysis:")
    print(f"  Total length: {len(markdown)} chars")
    
    # Check for LaTeX patterns
    latex_patterns = [
        (r'$$', 'Display math ($$)'),
        (r'$', 'Inline math ($)'),
        (r'\[', 'Display math (\\[)'),
        (r'\(', 'Inline math (\\()'),
        (r'\begin{equation', 'Equation environment'),
        (r'\begin{align', 'Align environment'),
        ('=', 'Equals signs'),
        ('α', 'Greek letters (α)'),
        ('β', 'Greek letters (β)'),
        ('∞', 'Math symbols (∞)'),
    ]
    
    for pattern, desc in latex_patterns:
        count = markdown.count(pattern)
        if count > 0:
            print(f"  {desc}: {count} occurrences")
    
    # Show a sample around equation references
    if '[6]' in markdown:
        idx = markdown.find('[6]')
        start = max(0, idx - 200)
        end = min(len(markdown), idx + 200)
        print(f"\nSample around reference [6]:")
        print("-" * 70)
        print(markdown[start:end])
        print("-" * 70)
    
    # Check document elements
    if hasattr(result.output, 'elements'):
        print(f"\nDocument elements: {len(result.output.elements)}")
        equation_elements = 0
        for elem in result.output.elements:
            elem_type = getattr(elem, 'type', None) or getattr(elem, '__class__.__name__', 'Unknown')
            if any(term in str(elem_type).lower() for term in ['equation', 'formula', 'math']):
                equation_elements += 1
                print(f"  Found equation element: {elem_type}")
                if hasattr(elem, 'text'):
                    print(f"    Text: {elem.text[:100]}...")
        
        if equation_elements == 0:
            print("  No equation-specific elements found")
    
    return result

if __name__ == "__main__":
    # Test with the physics paper that has equations
    test_pdf = "/bulk-store/arxiv-data/pdf/1712/1712.05056.pdf"
    
    if Path(test_pdf).exists():
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        analyze_equation_handling(test_pdf)
    else:
        print(f"Test PDF not found: {test_pdf}")