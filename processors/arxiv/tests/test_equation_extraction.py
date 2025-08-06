#!/usr/bin/env python3
"""
Test different approaches to extract equations from PDFs.
"""

import os
import subprocess
from pathlib import Path

def test_pdf_to_text_methods(pdf_path: str):
    """Test various PDF extraction methods to see how they handle equations."""
    
    print(f"Testing equation extraction methods for: {pdf_path}")
    print("=" * 70)
    
    # Method 1: Check if pdftotext preserves equations
    print("\n1. Testing pdftotext:")
    print("-" * 40)
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True
        )
        text = result.stdout
        
        # Look for equation content after reference [6]
        if '[6]' in text:
            idx = text.find('[6]')
            sample = text[idx-100:idx+300]
            print("Sample around [6]:")
            print(sample)
        else:
            print("Reference [6] not found")
            
    except Exception as e:
        print(f"pdftotext error: {e}")
    
    # Method 2: Check what Docling sees in the PDF structure
    print("\n\n2. Analyzing PDF structure with Docling:")
    print("-" * 40)
    
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import DocumentConversionInput
    from docling.datamodel.pipeline_options import PipelineOptions
    
    # Try with OCR enabled to see if equations are images
    pipeline_options = PipelineOptions(
        do_ocr=True,  # Enable OCR
        do_table_structure=True
    )
    
    converter = DocumentConverter(pipeline_options=pipeline_options)
    doc_input = DocumentConversionInput.from_paths(paths=[pdf_path])
    results = converter.convert(doc_input)
    result = next(results)
    
    # Check raw elements
    if hasattr(result.output, 'elements'):
        print(f"Total elements: {len(result.output.elements)}")
        
        # Look for elements around reference [6]
        for i, elem in enumerate(result.output.elements):
            if hasattr(elem, 'text') and elem.text and '[6]' in elem.text:
                print(f"\nFound element {i} with [6]:")
                print(f"  Type: {type(elem).__name__}")
                print(f"  Text: {elem.text}")
                
                # Check next few elements
                for j in range(i+1, min(i+5, len(result.output.elements))):
                    next_elem = result.output.elements[j]
                    print(f"\nNext element {j}:")
                    print(f"  Type: {type(next_elem).__name__}")
                    if hasattr(next_elem, 'text'):
                        print(f"  Text: {next_elem.text[:200] if next_elem.text else 'None'}")
                break
    
    # Method 3: Check if equations are stored as images
    print("\n\n3. Checking for equation images:")
    print("-" * 40)
    
    if hasattr(result.output, 'pictures') and result.output.pictures:
        print(f"Found {len(result.output.pictures)} pictures")
        for i, pic in enumerate(result.output.pictures):
            print(f"  Picture {i}: page {getattr(pic, 'page_num', 'unknown')}")
    else:
        print("No pictures found")
    
    # Method 4: Try different export formats
    print("\n\n4. Testing different export formats:")
    print("-" * 40)
    
    # Try JSON export to see raw data
    if hasattr(result.output, 'to_dict'):
        doc_dict = result.output.to_dict()
        # Look for any equation-related keys
        for key in doc_dict:
            if any(term in key.lower() for term in ['equation', 'formula', 'math']):
                print(f"Found key: {key}")
    
    # Check if there's a LaTeX export option
    if hasattr(result.output, 'export_to_latex'):
        print("LaTeX export available!")
        latex = result.output.export_to_latex()
        if '[6]' in latex:
            idx = latex.find('[6]')
            print(f"LaTeX around [6]: {latex[idx-100:idx+200]}")

if __name__ == "__main__":
    test_pdf = "/bulk-store/arxiv-data/pdf/1712/1712.05056.pdf"
    
    if Path(test_pdf).exists():
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        test_pdf_to_text_methods(test_pdf)
    else:
        print(f"Test PDF not found: {test_pdf}")