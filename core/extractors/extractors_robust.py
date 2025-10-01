"""
Robust PDF Extractor with Fallback
===================================

Wraps Docling with timeout protection and PyMuPDF fallback for problematic PDFs.
Implements the CONVEYANCE dimension with resilience against segfaults.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .docling_extractor import DoclingExtractor

logger = logging.getLogger(__name__)


def _extract_with_docling(pdf_path: str, use_ocr: bool, extract_tables: bool) -> Dict[str, Any]:
    """Extract using Docling (runs in subprocess)."""
    try:
        extractor = DoclingExtractor(
            use_ocr=use_ocr,
            extract_tables=extract_tables,
            use_fallback=False  # Don't use fallback here, we handle it at higher level
        )
        return extractor.extract(pdf_path)
    except Exception as e:
        return {'error': str(e)}


def _extract_with_pymupdf(pdf_path: str) -> Dict[str, Any]:
    """Fallback extraction using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        return None

    try:
        with fitz.open(pdf_path) as doc:
            text_parts = []
            num_pages = len(doc)

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

            full_text = "\n\n".join(text_parts)

            return {
                'full_text': full_text,
                'text': full_text,
                'markdown': full_text,
                'num_pages': num_pages,
                'extractor': 'pymupdf_fallback'
            }
    except Exception as e:
        logger.error(f"PyMuPDF fallback failed: {e}")
        return None


class RobustExtractor:
    """
    Robust PDF extractor with timeout protection and fallback.
    
    This extractor represents the CONVEYANCE dimension with maximum resilience,
    ensuring information can be transformed even when primary tools fail.
    It acts as a boundary object between unstable extraction tools and
    the reliable embedding pipeline.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        extract_tables: bool = True,
        timeout: int = 30,
        use_fallback: bool = True
    ):
        """
        Initialize robust extractor.
        
        Args:
            use_ocr: Whether to use OCR for scanned PDFs
            extract_tables: Whether to extract table structures
            timeout: Maximum seconds to wait for extraction
            use_fallback: Whether to use PyMuPDF fallback on failure
        """
        self.use_ocr = use_ocr
        self.extract_tables = extract_tables
        self.timeout = timeout
        self.use_fallback = use_fallback

        logger.info(f"Initialized RobustExtractor (timeout: {timeout}s, fallback: {use_fallback})")

    def extract(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract text and structures from PDF with timeout protection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted data or None if extraction fails
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None

        # Try Docling with timeout
        logger.debug(f"Attempting Docling extraction: {pdf_file.name}")

        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context('spawn')
        )

        try:
            future = executor.submit(
                _extract_with_docling,
                pdf_path,
                self.use_ocr,
                self.extract_tables
            )

            try:
                result = future.result(timeout=self.timeout)

                if result and 'error' not in result:
                    logger.debug(f"Docling extraction successful: {pdf_file.name}")
                    return result
                else:
                    logger.warning(f"Docling extraction failed: {pdf_file.name}")

            except TimeoutError:
                logger.warning(f"Docling timeout after {self.timeout}s: {pdf_file.name}")
                # Cancel the future and shutdown executor
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

                # Clean up any remaining child processes
                for child in mp.active_children():
                    logger.debug(f"Terminating child process: {child.pid}")
                    child.terminate()
                    child.join(timeout=2)
                    if child.is_alive():
                        logger.warning(f"Force killing child process: {child.pid}")
                        child.kill()
                        child.join(timeout=1)

                # Additional cleanup of executor processes
                try:
                    # Try public shutdown API first
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    # Fall back to private attribute access as last resort
                    try:
                        if hasattr(executor, '_processes'):
                            for process in executor._processes.values():
                                if process and process.is_alive():
                                    process.terminate()
                                    process.join(timeout=2)
                                    if process.is_alive():
                                        process.kill()
                                        process.join(timeout=1)
                    except (AttributeError, Exception) as e:
                        logger.debug(f"Could not access executor processes: {e}")

            except Exception as e:
                logger.error(f"Docling crashed: {pdf_file.name} - {e}")
                executor.shutdown(wait=False, cancel_futures=True)

                # Clean up child processes on any exception
                for child in mp.active_children():
                    child.terminate()
                    child.join(timeout=2)

        finally:
            # Ensure executor is shut down
            executor.shutdown(wait=False)
            # Clear executor reference
            executor = None

        # Try fallback if enabled
        if self.use_fallback and PYMUPDF_AVAILABLE:
            logger.info(f"Using PyMuPDF fallback: {pdf_file.name}")
            return _extract_with_pymupdf(pdf_path)

        return None
