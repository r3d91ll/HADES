#!/usr/bin/env python3
"""
Test image and graph extraction from PDFs.
"""

import os
from pathlib import Path
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DocumentConversionInput
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.base_models import AssembleOptions

def test_image_extraction(pdf_path: str):
    """Test how Docling handles images and graphs."""
    
    print(f"Testing image extraction for: {pdf_path}")
    print("=" * 70)
    
    # 1. Check with PyMuPDF what images exist
    print("\n1. PyMuPDF Image Analysis:")
    print("-" * 40)
    
    doc = fitz.open(pdf_path)
    total_images = 0
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images()
        if image_list:
            print(f"\nPage {page_num + 1}: Found {len(image_list)} images")
            total_images += len(image_list)
            
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                print(f"  Image {img_idx}: {pix.width}x{pix.height} pixels, {pix.n} channels")
    
    print(f"\nTotal images in PDF: {total_images}")
    doc.close()
    
    # 2. Test Docling extraction
    print("\n\n2. Docling Image Extraction:")
    print("-" * 40)
    
    # Configure for image extraction
    pipeline_options = PipelineOptions(
        do_ocr=False,
        do_table_structure=True
    )
    
    converter = DocumentConverter(
        pipeline_options=pipeline_options,
        assemble_options=AssembleOptions(
            keep_page_images=True,
            images_scale=1.0  # Full resolution
        )
    )
    
    # Convert document
    doc_input = DocumentConversionInput.from_paths(paths=[pdf_path])
    results = converter.convert(doc_input)
    result = next(results)
    
    # Check for pictures
    if hasattr(result.output, 'pictures') and result.output.pictures:
        print(f"Docling found {len(result.output.pictures)} pictures")
        
        for idx, picture in enumerate(result.output.pictures):
            print(f"\nPicture {idx}:")
            print(f"  Has image data: {hasattr(picture, 'image') and picture.image is not None}")
            print(f"  Page: {getattr(picture, 'page_num', 'unknown')}")
            print(f"  BBox: {getattr(picture, 'bbox', 'unknown')}")
            
            if hasattr(picture, 'image') and picture.image is not None:
                img = picture.image
                print(f"  Format: {getattr(img, 'format', 'unknown')}")
                if hasattr(img, 'size'):
                    print(f"  Size: {img.size}")
                if hasattr(img, 'mode'):
                    print(f"  Mode: {img.mode}")
    else:
        print("Docling found no pictures")
    
    # Check for tables (might contain graphs)
    if hasattr(result.output, 'tables') and result.output.tables:
        print(f"\n\nDocling found {len(result.output.tables)} tables")
    
    # 3. Check markdown output
    print("\n\n3. Markdown Output Analysis:")
    print("-" * 40)
    
    markdown = result.output.export_to_markdown()
    
    # Look for image references
    image_patterns = [
        ('![', 'Markdown images'),
        ('<img', 'HTML images'),
        ('Figure', 'Figure references'),
        ('figure', 'figure references'),
        ('Table', 'Table references'),
        ('Graph', 'Graph references')
    ]
    
    for pattern, desc in image_patterns:
        count = markdown.count(pattern)
        if count > 0:
            print(f"  {desc}: {count} occurrences")
    
    # Sample markdown to see how images are handled
    if '![' in markdown:
        idx = markdown.find('![')
        print(f"\nSample image reference in markdown:")
        print(markdown[idx:idx+200])


def find_pdf_with_images():
    """Find PDFs that likely have images/graphs."""
    
    # Common directories for papers with figures
    test_dirs = [
        "/bulk-store/arxiv-data/pdf/1712",  # Recent papers
        "/bulk-store/arxiv-data/pdf/2001",  # 2020 papers
        "/bulk-store/arxiv-data/pdf/cs",    # Computer science
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"\nSearching {test_dir} for PDFs with images...")
            
            pdf_files = list(Path(test_dir).glob("*.pdf"))[:10]  # Check first 10
            
            for pdf_file in pdf_files:
                doc = fitz.open(str(pdf_file))
                
                # Count images
                total_images = 0
                for page in doc:
                    total_images += len(page.get_images())
                
                if total_images > 0:
                    print(f"\nFound {pdf_file.name} with {total_images} images")
                    doc.close()
                    return str(pdf_file)
                
                doc.close()
    
    return None


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # First try the physics paper
    test_pdf = "/bulk-store/arxiv-data/pdf/1712/1712.05056.pdf"
    
    if Path(test_pdf).exists():
        test_image_extraction(test_pdf)
    
    # Try to find a PDF with images
    print("\n\n" + "=" * 70)
    print("Looking for PDFs with images...")
    print("=" * 70)
    
    pdf_with_images = find_pdf_with_images()
    if pdf_with_images:
        print(f"\n\nTesting PDF with images: {pdf_with_images}")
        test_image_extraction(pdf_with_images)