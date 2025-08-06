#!/usr/bin/env python3
"""
Enhanced Docling processor with smart image handling and database-first architecture.

Theory Connection:
This processor implements a "library catalog" approach where documents are 
processed on-demand. It prevents over-extraction of images and maintains
high-fidelity representations in the database that can be reconstructed
without the original files.
"""

import os
import re
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import io

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.pipeline_options import PipelineOptions
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquationBlock:
    """Represents an extracted equation."""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    context: Optional[str] = None


@dataclass
class ImageInfo:
    """Information about an extracted image."""
    filename: str
    page: int
    width: int
    height: int
    file_size: int
    is_valid: bool
    reason: Optional[str] = None


class ImageValidator:
    """Validates and filters images to prevent over-extraction."""
    
    # Image quality thresholds
    MIN_WIDTH = 100  # Minimum width in pixels
    MIN_HEIGHT = 100  # Minimum height in pixels
    MIN_AREA = 15000  # Minimum area (pixels)
    MAX_ASPECT_RATIO = 10.0  # Maximum width/height or height/width ratio
    MIN_FILE_SIZE = 5000  # Minimum file size in bytes (5KB)
    MAX_IMAGES_PER_DOC = 20  # Maximum images to extract per document
    
    @classmethod
    def validate_image(cls, img: Image.Image, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate if an image should be kept.
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        width, height = img.size
        area = width * height
        
        # Check minimum dimensions
        if width < cls.MIN_WIDTH or height < cls.MIN_HEIGHT:
            return False, f"Too small: {width}x{height} pixels"
        
        # Check minimum area
        if area < cls.MIN_AREA:
            return False, f"Area too small: {area} pixels"
        
        # Check aspect ratio (prevent thin strips)
        aspect_ratio = max(width/height, height/width)
        if aspect_ratio > cls.MAX_ASPECT_RATIO:
            return False, f"Bad aspect ratio: {aspect_ratio:.1f}"
        
        # Check file size
        if file_size < cls.MIN_FILE_SIZE:
            return False, f"File too small: {file_size} bytes"
        
        # Check if it's likely a fragmented grid cell
        if cls._is_likely_fragment(img):
            return False, "Likely grid fragment"
        
        return True, None
    
    @classmethod
    def _is_likely_fragment(cls, img: Image.Image) -> bool:
        """
        Detect if image is likely a fragment from a grid/table.
        
        Heuristics:
        - Very uniform color distribution (single color cells)
        - Repetitive patterns
        - Small size with high uniformity
        """
        # Convert to grayscale for analysis
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Check color uniformity
        unique_colors = len(np.unique(img_array))
        total_pixels = img_array.size
        
        # If very few unique colors relative to size, likely a fragment
        color_ratio = unique_colors / min(total_pixels, 256)
        if color_ratio < 0.1:  # Less than 10% color variation
            return True
        
        # Check for border patterns (common in grid cells)
        # Check if edges are mostly the same color
        edge_pixels = np.concatenate([
            img_array[0, :],  # Top edge
            img_array[-1, :],  # Bottom edge
            img_array[:, 0],  # Left edge
            img_array[:, -1]  # Right edge
        ])
        
        edge_std = np.std(edge_pixels)
        if edge_std < 10:  # Very uniform edges
            # Check if center is different (typical of grid cells)
            center = img_array[1:-1, 1:-1]
            if center.size > 0:
                center_mean = np.mean(center)
                edge_mean = np.mean(edge_pixels)
                if abs(center_mean - edge_mean) > 50:
                    return True
        
        return False
    
    @classmethod
    def rank_images(cls, images: List[Dict]) -> List[Dict]:
        """
        Rank images by quality/importance.
        
        Prioritizes:
        1. Larger, more complex images
        2. Images from earlier pages (often more important)
        3. Images with good aspect ratios
        """
        scored_images = []
        
        for img_data in images:
            score = 0
            img = img_data['image']
            width, height = img.size
            
            # Size score (larger is better)
            area = width * height
            score += min(area / 100000, 10)  # Max 10 points for size
            
            # Page position score (earlier is better)
            score += max(10 - img_data['page'], 0)  # Earlier pages get higher scores
            
            # Aspect ratio score (closer to golden ratio is better)
            aspect = width / height if height > 0 else 1
            golden_ratio = 1.618
            aspect_diff = abs(aspect - golden_ratio)
            score += max(5 - aspect_diff, 0)  # Max 5 points for good aspect ratio
            
            # Complexity score (more unique colors is better)
            img_array = np.array(img.convert('L'))
            unique_colors = len(np.unique(img_array))
            score += min(unique_colors / 25, 10)  # Max 10 points for complexity
            
            scored_images.append((score, img_data))
        
        # Sort by score (highest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)
        
        return [img_data for _, img_data in scored_images]


class EnhancedDoclingProcessorV2:
    """
    Enhanced processor with smart image handling and database-centric design.
    """
    
    def __init__(self, output_dir: str = "/bulk-store/arxiv-data/pdf/pre-processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Docling converter
        pipeline_options = PipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            table_structure_options={
                "do_cell_matching": True,
                "mode": "1"  # Changed from "fast" to "1" per API requirements
            }
        )
        
        # DocumentConverter no longer takes pipeline_options in constructor
        self.converter = DocumentConverter()
        self.pipeline_options = pipeline_options
        
        # Image validator
        self.image_validator = ImageValidator()
        
        logger.info("Enhanced Docling Processor V2 initialized")
    
    def process_pdf(self, arxiv_id: str, pdf_path: str, 
                   max_images: int = 20,
                   skip_tiny_images: bool = True) -> Dict:
        """
        Process PDF with smart image handling.
        
        Args:
            arxiv_id: ArXiv ID
            pdf_path: Path to PDF file
            max_images: Maximum number of images to extract
            skip_tiny_images: Whether to skip small/fragmented images
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        stats = {
            'total_images_found': 0,
            'images_validated': 0,
            'images_kept': 0,
            'images_skipped': 0,
            'skip_reasons': {}
        }
        
        try:
            # Process with Docling for text and structure
            # Convert the PDF directly with path
            results = self.converter.convert(pdf_path)
            result = next(results)
            
            # Get base markdown
            markdown_content = result.output.export_to_markdown()
            
            # Extract equations
            equations = self._extract_equations_pymupdf(pdf_path)
            logger.info(f"Extracted {len(equations)} equations from {arxiv_id}")
            
            # Merge equations into markdown
            enhanced_markdown = self._merge_equations_into_markdown(
                markdown_content, equations, pdf_path
            )
            
            # Extract and validate images
            images_data = self._extract_and_validate_images(
                pdf_path, max_images, skip_tiny_images, stats
            )
            
            # Save processed data
            output_data = self._save_processed_data(
                arxiv_id, enhanced_markdown, equations, images_data, stats
            )
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'arxiv_id': arxiv_id,
                'processing_time': processing_time,
                'num_equations': len(equations),
                'num_images': len(images_data),
                'output_path': str(output_data['markdown_path']),
                'stats': stats,
                'metadata': {
                    'enhanced': True,
                    'max_images_enforced': max_images,
                    'image_validation_enabled': skip_tiny_images,
                    'images_skipped': stats['images_skipped'],
                    'skip_reasons': stats['skip_reasons']
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            return {
                'success': False,
                'arxiv_id': arxiv_id,
                'error': str(e),
                'stats': stats
            }
    
    def _extract_and_validate_images(self, pdf_path: str, max_images: int, 
                                    validate: bool, stats: Dict) -> List[Dict]:
        """
        Extract and validate images from PDF.
        """
        extracted_images = []
        
        try:
            doc = fitz.open(pdf_path)
            all_images = []
            
            # First pass: extract all images
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                stats['total_images_found'] += len(image_list)
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PNG bytes
                        img_data = pix.tobytes("png")
                        file_size = len(img_data)
                        
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Store image data
                        all_images.append({
                            'image': img,
                            'page': page_num,
                            'width': pix.width,
                            'height': pix.height,
                            'file_size': file_size,
                            'img_data': img_data
                        })
                        
                        if pix.n - pix.alpha > 3:  # Convert CMYK to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                    except Exception as e:
                        logger.debug(f"Failed to extract image: {e}")
                        continue
            
            doc.close()
            
            # Validate and filter images
            valid_images = []
            
            for img_data in all_images:
                if validate:
                    is_valid, reason = self.image_validator.validate_image(
                        img_data['image'], 
                        img_data['file_size']
                    )
                    
                    stats['images_validated'] += 1
                    
                    if is_valid:
                        valid_images.append(img_data)
                        stats['images_kept'] += 1
                    else:
                        stats['images_skipped'] += 1
                        stats['skip_reasons'][reason] = stats['skip_reasons'].get(reason, 0) + 1
                else:
                    valid_images.append(img_data)
            
            # Rank images by quality/importance
            if len(valid_images) > max_images:
                logger.info(f"Found {len(valid_images)} valid images, limiting to {max_images}")
                ranked_images = self.image_validator.rank_images(valid_images)
                valid_images = ranked_images[:max_images]
                
                # Update stats
                num_skipped = len(ranked_images) - max_images
                stats['images_skipped'] += num_skipped
                stats['skip_reasons']['max_limit_exceeded'] = num_skipped
            
            # Prepare final image data
            for idx, img_data in enumerate(valid_images):
                extracted_images.append({
                    'filename': f"figure_{idx + 1}.png",
                    'image': img_data['image'],
                    'page': img_data['page'],
                    'width': img_data['width'],
                    'height': img_data['height'],
                    'file_size': img_data['file_size'],
                    'img_data': img_data['img_data']
                })
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
        
        return extracted_images
    
    def _extract_equations_pymupdf(self, pdf_path: str) -> List[EquationBlock]:
        """Extract equations using PyMuPDF."""
        equations = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Get page text with detailed layout
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                            
                            # Detect equation patterns
                            if self._is_likely_equation(line_text):
                                bbox = block.get("bbox", (0, 0, 0, 0))
                                equations.append(EquationBlock(
                                    text=line_text.strip(),
                                    page=page_num,
                                    bbox=bbox
                                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting equations: {e}")
        
        return equations
    
    def _is_likely_equation(self, text: str) -> bool:
        """Detect if text is likely an equation."""
        if len(text.strip()) < 3:
            return False
        
        equation_indicators = [
            r'[=<>≤≥≈≠∝]',  # Mathematical relations
            r'[∫∑∏∂∇]',  # Calculus symbols
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
            r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Capital Greek
            r'\^|_|\{|\}',  # LaTeX-like formatting
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'\d+\s*[+\-*/]\s*\d+',  # Basic arithmetic
            r'\([^)]*\)',  # Parenthetical expressions with math
        ]
        
        matches = sum(1 for pattern in equation_indicators if re.search(pattern, text))
        return matches >= 2 or (matches == 1 and len(text) > 10)
    
    def _merge_equations_into_markdown(self, markdown: str, equations: List[EquationBlock], 
                                      pdf_path: str) -> str:
        """Merge extracted equations into markdown."""
        if not equations:
            return markdown
        
        lines = markdown.split('\n')
        enhanced_lines = []
        used_equations = set()
        
        for line in lines:
            enhanced_lines.append(line)
            
            # Check if line might reference an equation
            if any(keyword in line.lower() for keyword in 
                   ['equation', 'formula', 'where', 'given', 'defined', 'follows']):
                
                best_match = self._find_best_equation_match(line, equations, used_equations)
                if best_match:
                    enhanced_lines.append(f"\n$${best_match.text}$$\n")
                    used_equations.add(best_match)
        
        # Add remaining equations at the end if significant
        remaining = [eq for eq in equations if eq not in used_equations]
        if remaining:
            enhanced_lines.append("\n## Additional Equations\n")
            for eq in remaining[:10]:  # Limit to 10 additional equations
                enhanced_lines.append(f"\n$${eq.text}$$\n")
        
        return '\n'.join(enhanced_lines)
    
    def _find_best_equation_match(self, context_line: str, equations: List[EquationBlock], 
                                 used: set) -> Optional[EquationBlock]:
        """Find the best equation match based on context."""
        # Implementation similar to original but with improved matching
        for eq in equations:
            if eq not in used:
                # Check for variable matches
                context_vars = re.findall(r'\b[a-zA-Z]\b', context_line)
                eq_vars = re.findall(r'\b[a-zA-Z]\b', eq.text)
                if any(var in eq_vars for var in context_vars):
                    return eq
        
        # Return next unused equation if no specific match
        for eq in equations:
            if eq not in used:
                return eq
        
        return None
    
    def _save_processed_data(self, arxiv_id: str, markdown: str, 
                            equations: List[EquationBlock], 
                            images: List[Dict], stats: Dict) -> Dict:
        """
        Save processed data to filesystem.
        
        Returns paths to saved files.
        """
        # Create output directory structure
        year_month = arxiv_id[:4] if '.' in arxiv_id else arxiv_id[:2]
        output_subdir = self.output_dir / year_month
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save markdown
        markdown_path = output_subdir / f"{arxiv_id.replace('/', '_')}.md"
        
        # Add image references to markdown if images exist
        if images:
            markdown += "\n\n## Figures\n\n"
            for img_data in images:
                img_ref = f"![{img_data['filename']}](./{arxiv_id.replace('/', '_')}_images/{img_data['filename']})\n"
                markdown += img_ref
        
        markdown_path.write_text(markdown, encoding='utf-8')
        
        # Save images if any
        if images:
            images_dir = output_subdir / f"{arxiv_id.replace('/', '_')}_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for img_data in images:
                img_path = images_dir / img_data['filename']
                img_path.write_bytes(img_data['img_data'])
        
        # Save metadata
        metadata = {
            'arxiv_id': arxiv_id,
            'num_equations': len(equations),
            'num_images': len(images),
            'processing_time': time.time(),
            'enhanced': True,
            'stats': stats,
            'equation_blocks': [
                {'text': eq.text, 'page': eq.page} 
                for eq in equations[:10]  # Save first 10 equations in metadata
            ],
            'images': [
                {
                    'index': idx,
                    'type': 'picture',
                    'filename': img['filename'],
                    'page': img['page'],
                    'size': [img['width'], img['height']],
                    'file_size': img['file_size'],
                    'has_data': True
                }
                for idx, img in enumerate(images)
            ]
        }
        
        metadata_path = output_subdir / f"{arxiv_id.replace('/', '_')}_meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        return {
            'markdown_path': markdown_path,
            'images_dir': images_dir if images else None,
            'metadata_path': metadata_path
        }


def reconstruct_markdown_from_database(arxiv_id: str, db_connection) -> str:
    """
    Reconstruct the markdown document from database storage.
    
    This allows us to verify that documents are properly stored
    and can be retrieved without the original files.
    """
    try:
        # Query the database for the document
        doc = db_connection.collection('base_arxiv').get(arxiv_id)
        
        if not doc:
            raise ValueError(f"Document {arxiv_id} not found in database")
        
        # Get the full text (markdown)
        markdown = doc.get('full_text', '')
        
        # Get metadata
        metadata = {
            'title': doc.get('title', ''),
            'authors': doc.get('authors', []),
            'abstract': doc.get('abstract', ''),
            'categories': doc.get('categories', []),
            'published': doc.get('published_date', ''),
            'pdf_status': doc.get('pdf_status', 'unknown')
        }
        
        # Reconstruct full document with metadata header
        reconstructed = f"""# {metadata['title']}

**Authors:** {', '.join(metadata['authors'])}
**Categories:** {', '.join(metadata['categories'])}
**Published:** {metadata['published']}
**Status:** {metadata['pdf_status']}

## Abstract

{metadata['abstract']}

---

{markdown}
"""
        
        # If embeddings exist, add statistics
        if 'embeddings' in doc:
            emb = doc['embeddings']
            stats = f"""
---
## Processing Statistics

- **Model:** {emb.get('model', 'unknown')}
- **Chunks:** {emb.get('num_chunks', 0)}
- **Total Tokens:** {emb.get('total_tokens', 0)}
- **Processing Time:** {emb.get('processing_time', 0):.2f}s
- **Embedded Date:** {emb.get('embedded_date', 'unknown')}
"""
            reconstructed += stats
        
        return reconstructed
        
    except Exception as e:
        logger.error(f"Error reconstructing markdown for {arxiv_id}: {e}")
        raise


if __name__ == "__main__":
    # Test the processor
    processor = EnhancedDoclingProcessorV2()
    
    # Example usage
    result = processor.process_pdf(
        arxiv_id="2007.00656",
        pdf_path="/path/to/pdf",
        max_images=20,  # Limit to 20 images
        skip_tiny_images=True  # Skip fragments
    )
    
    if result['success']:
        print(f"Successfully processed {result['arxiv_id']}")
        print(f"Images: {result['num_images']} (skipped: {result['stats']['images_skipped']})")
        print(f"Skip reasons: {result['stats']['skip_reasons']}")
    else:
        print(f"Failed: {result['error']}")