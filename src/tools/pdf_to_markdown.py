#!/usr/bin/env python3
"""
PDF to Markdown Converter Tool

This tool uses our DoclingDocumentProcessor to convert PDF files to markdown format.
It demonstrates using HADES components for standalone tasks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.components.docproc.factory import create_docproc_component
from src.types.components.contracts import DocumentProcessingInput

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    save_images: bool = True
) -> str:
    """
    Convert a PDF file to markdown using DoclingDocumentProcessor.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional output path for markdown file
        save_images: Whether to save extracted images
        
    Returns:
        Markdown content as string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If file is not a PDF
    """
    # Validate input
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if pdf_file.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF, got: {pdf_file.suffix}")
    
    logger.info(f"Converting PDF to markdown: {pdf_path}")
    
    # Create docling processor with appropriate config
    config = {
        "extract_images": save_images,
        "extract_tables": True,
        "extract_equations": True,
        "output_format": "markdown"
    }
    
    try:
        # Create the processor
        processor = create_docproc_component(component_type="docling", config=config)
        
        # Create input
        input_data = DocumentProcessingInput(
            file_path=str(pdf_file),
            metadata={"source": "pdf_to_markdown_tool"}
        )
        
        # Process the document
        logger.info("Processing document with Docling...")
        result = processor.process(input_data)
        
        if result.total_errors > 0:
            logger.error(f"Processing errors: {result.errors}")
            raise RuntimeError(f"Document processing failed with {result.total_errors} errors")
        
        if not result.documents:
            raise RuntimeError("No documents produced from processing")
        
        # Extract markdown content
        document = result.documents[0]
        markdown_content = document.content
        
        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(markdown_content, encoding='utf-8')
            logger.info(f"Markdown saved to: {output_path}")
        
        # Save images if extracted
        if save_images and document.metadata.get("images"):
            image_dir = Path(output_path).parent / "images" if output_path else pdf_file.parent / "images"
            image_dir.mkdir(exist_ok=True)
            
            for idx, image_data in enumerate(document.metadata["images"]):
                image_path = image_dir / f"{pdf_file.stem}_image_{idx}.png"
                # Note: Actual image saving would depend on how Docling returns image data
                logger.info(f"Would save image to: {image_path}")
        
        logger.info("Conversion completed successfully")
        return markdown_content
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


def main():
    """Command-line interface for PDF to Markdown conversion."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using HADES DoclingDocumentProcessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PDF and save to same directory
  python pdf_to_markdown.py paper.pdf
  
  # Convert and save to specific file
  python pdf_to_markdown.py paper.pdf -o converted/paper.md
  
  # Convert without saving images
  python pdf_to_markdown.py paper.pdf --no-images
        """
    )
    
    parser.add_argument(
        "pdf_file",
        help="Path to PDF file to convert"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output markdown file path (default: same as PDF with .md extension)"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't extract and save images from PDF"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output path
    if not args.output:
        pdf_path = Path(args.pdf_file)
        output_path = pdf_path.with_suffix('.md')
    else:
        output_path = args.output
    
    try:
        # Convert the PDF
        markdown_content = convert_pdf_to_markdown(
            args.pdf_file,
            output_path=str(output_path),
            save_images=not args.no_images
        )
        
        # Print summary
        lines = markdown_content.count('\n')
        words = len(markdown_content.split())
        print(f"\n✅ Conversion successful!")
        print(f"📄 Output: {output_path}")
        print(f"📊 Stats: {lines} lines, {words} words")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()