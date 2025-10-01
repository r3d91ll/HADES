"""
Docling PDF Extractor
=====================

Extracts text and structures from PDFs using Docling v2.
Represents the CONVEYANCE dimension in Information Reconstructionism -
transforming information from PDF format to structured text and metadata.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    # DocumentConversionInput was removed in newer versions
    from docling.datamodel.pipeline_options import PipelineOptions, TableStructureOptions
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    logging.warning(f"Docling not available: {e}")

logger = logging.getLogger(__name__)


class DoclingExtractor:
    """
    Extract text and structures from PDFs using Docling.
    
    This extractor serves as a boundary object between raw PDFs and
    the embedding pipeline, preserving semantic structures while
    transforming the document format.
    """

    def __init__(self, config=None):
        """
        Initialize Docling extractor.

        Args:
            config: ExtractorConfig object or None for defaults
        """
        # Handle both old-style params and new config object
        if config is None:
            # Default values
            self.use_ocr = False
            self.extract_tables = True
            self.use_fallback = False
        elif hasattr(config, 'ocr_enabled'):
            # It's an ExtractorConfig
            self.use_ocr = config.ocr_enabled
            self.extract_tables = config.extract_tables
            self.use_fallback = False
        else:
            # Old-style direct params (for backwards compatibility)
            self.use_ocr = config
            self.extract_tables = True
            self.use_fallback = False

        if DOCLING_AVAILABLE:
            # Configure pipeline options - Docling v2 doesn't accept None for table_structure_options
            if self.extract_tables:
                self.pipeline_options = PipelineOptions(
                    do_ocr=self.use_ocr,
                    do_table_structure=True,
                    table_structure_options=TableStructureOptions(
                        do_cell_matching=True
                    )
                )
            else:
                self.pipeline_options = PipelineOptions(
                    do_ocr=self.use_ocr,
                    do_table_structure=False
                    # Don't include table_structure_options when not extracting tables
                )

            # Initialize converter
            self.converter = DocumentConverter(
                pipeline_options=self.pipeline_options
            )
            logger.info(f"Initialized Docling extractor (OCR: {self.use_ocr}, Tables: {self.extract_tables})")
        else:
            self.converter = None
            logger.warning("Docling not available - using fallback text extraction")

    def extract(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and optional structured content from a file path (PDF or plain text).
        
        This method accepts a path to a document and returns a normalized extraction result. Behavior:
        - If the path does not exist, raises FileNotFoundError.
        - For text files (suffix .txt, .text, .md): reads the file as UTF-8 and returns an early-text result:
          a dictionary containing 'text' (file contents), empty lists for 'tables', 'equations', 'images', 'figures',
          and 'metadata' with page_count=1, extractor='text_reader', pdf_path, processing_time=0.0 and version='text_reader'.
          Reading errors or empty text raise RuntimeError.
        - For PDF files: performs a basic PDF header check and then attempts extraction using Docling if available,
          otherwise uses the configured fallback extractor. The chosen extractor's result is returned (typically containing
          full_text/markdown, structures, and metadata). In all successful non-text cases the method injects processing_time
          (seconds) and pdf_path into the returned metadata.
        - Pre-validation failures (empty file, invalid header, unreadable file) raise RuntimeError and are not retried with fallback.
        - Processing-time errors will attempt the fallback extractor if use_fallback is True; otherwise a RuntimeError is raised.
        
        Returns:
            Dict[str, Any]: Extraction result. For text-reader paths the result contains a 'text' key and metadata described above.
            For PDF extraction, the exact keys come from the underlying extractor but will include metadata with at least
            'extractor', 'num_pages' (when available), 'processing_time', and 'pdf_path'.
        
        Raises:
            FileNotFoundError: If the provided path does not exist.
            RuntimeError: For empty files, invalid PDF header, unreadable files, text read failures, extraction failures,
                          or when Docling is unavailable and fallback is disabled.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Pre-validation: Check if file is readable and not empty
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            # More specific error message based on file type
            if pdf_path.suffix in ['.txt', '.text', '.md']:
                raise RuntimeError(f"Text file is empty: {pdf_path}")
            else:
                raise RuntimeError(f"File is empty: {pdf_path}")

        # Check file type and handle appropriately
        pdf_path_obj = Path(pdf_path)

        # For text files, handle directly (useful for testing)
        if pdf_path_obj.suffix in ['.txt', '.text', '.md']:
            try:
                with open(pdf_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                if not text.strip():
                    raise RuntimeError(f"Text file is empty: {pdf_path}")

                return {
                    'text': text,
                    'tables': [],
                    'equations': [],
                    'images': [],
                    'figures': [],
                    'metadata': {
                        'page_count': 1,
                        'extractor': 'text_reader',
                        'pdf_path': str(pdf_path),
                        'processing_time': 0.0
                    },
                    'version': 'text_reader'
                }
            except (IOError, UnicodeDecodeError) as e:
                raise RuntimeError(f"Cannot read text file: {pdf_path}, error: {e}")

        # Check basic PDF header for actual PDF files
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    raise RuntimeError(f"Invalid PDF header: {pdf_path}")
        except IOError as e:
            raise RuntimeError(f"Cannot read PDF file: {pdf_path}, error: {e}")

        start_time = datetime.now()

        try:
            if DOCLING_AVAILABLE and self.converter:
                result = self._extract_with_docling(pdf_path)
                # Add processing time and pdf_path to metadata
                duration = (datetime.now() - start_time).total_seconds()
                result['metadata']['processing_time'] = duration
                result['metadata']['pdf_path'] = str(pdf_path)
                return result
            else:
                if self.use_fallback:
                    logger.warning("Docling not available, using fallback")
                    result = self._extract_fallback(pdf_path)
                    # Add processing time and pdf_path to metadata
                    duration = (datetime.now() - start_time).total_seconds()
                    result['metadata']['processing_time'] = duration
                    result['metadata']['pdf_path'] = str(pdf_path)
                    return result
                else:
                    raise RuntimeError("Docling not available and fallback disabled")
        except Exception as e:
            # Compute duration before handling exception
            duration = (datetime.now() - start_time).total_seconds()

            # Don't use fallback for pre-validation errors (they should fail fast)
            error_msg = str(e)
            if any(x in error_msg for x in ["empty", "Invalid PDF header", "Cannot read PDF file"]):
                logger.error(f"Pre-validation failed for {pdf_path}: {e}")
                raise RuntimeError(f"Pre-validation failed for {pdf_path}: {e}") from e

            # Use fallback for processing errors if enabled
            if self.use_fallback:
                logger.warning(f"Extraction failed, attempting fallback: {e}")
                result = self._extract_fallback(pdf_path)
                # Add processing time and pdf_path to metadata
                result['metadata']['processing_time'] = duration
                result['metadata']['pdf_path'] = str(pdf_path)
                return result
            else:
                # Re-raise with more context
                logger.error(f"Extraction failed for {pdf_path}: {e}")
                raise RuntimeError(f"Extraction failed for {pdf_path}: {e}") from e

    def _extract_with_docling(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract using Docling library v2 with enhanced error handling."""
        try:
            # Docling v2 uses convert_single for single PDFs
            # Note: Timeout protection disabled due to signal conflicts with Docling's internal processes
            result = self.converter.convert_single(str(pdf_path))

            # Check conversion status
            if hasattr(result, 'status'):
                # ConversionStatus.SUCCESS has value '4', ConversionStatus.PARTIAL_SUCCESS has different value
                # Check by name instead of value
                status_name = result.status.name if hasattr(result.status, 'name') else str(result.status)
                if status_name not in ['SUCCESS', 'PARTIAL_SUCCESS']:
                    error_msg = f"Docling conversion failed with status: {status_name}"
                    logger.error(error_msg)
                    if self.use_fallback:
                        logger.info("Attempting fallback extraction")
                        return self._extract_fallback(pdf_path)
                    else:
                        raise RuntimeError(error_msg)

            # Extract from result.output (Docling v2 structure)
            full_text = ""
            structures: Dict[str, Any] = {}
            num_pages = None

            if hasattr(result, 'output'):
                output = result.output

                # Export to markdown
                if hasattr(output, 'export_to_markdown'):
                    full_text = output.export_to_markdown()
                elif hasattr(output, 'text'):
                    full_text = output.text
                else:
                    full_text = str(output)

                # Extract tables if present
                if self.extract_tables and hasattr(output, 'tables'):
                    structures['tables'] = []
                    for table in output.tables:
                        table_data = {
                            'caption': getattr(table, 'caption', ''),
                            'content': str(table),
                            'rows': getattr(table, 'rows', []),
                            'headers': getattr(table, 'headers', [])
                        }
                        structures['tables'].append(table_data)

                # Extract equations if present
                if hasattr(output, 'equations'):
                    structures['equations'] = []
                    for eq in output.equations:
                        eq_data = {
                            'latex': getattr(eq, 'latex', str(eq)),
                            'label': getattr(eq, 'label', None)
                        }
                        structures['equations'].append(eq_data)

                # Extract figures if present
                if hasattr(output, 'figures'):
                    structures['images'] = []
                    for fig in output.figures:
                        fig_data = {
                            'caption': getattr(fig, 'caption', ''),
                            'label': getattr(fig, 'label', None),
                            'page': getattr(fig, 'page', None)
                        }
                        structures['images'].append(fig_data)

            # Get number of pages if available
            if hasattr(result, 'pages'):
                num_pages = len(result.pages)

            if not full_text:
                error_msg = "Docling extracted no text from PDF"
                logger.error(error_msg)
                if self.use_fallback:
                    logger.info("Attempting fallback extraction")
                    return self._extract_fallback(pdf_path)
                else:
                    raise RuntimeError(error_msg)

        except (TimeoutError, RuntimeError) as e:
            # Handle timeout and runtime errors explicitly
            logger.warning(f"Docling processing issue for {pdf_path}: {e}")
            if self.use_fallback:
                logger.info("Attempting fallback extraction")
                return self._extract_fallback(pdf_path)
            else:
                raise

        except Exception as e:
            # Handle assertion failures and other crashes
            error_msg = str(e)
            if "Assertion" in error_msg or "assertion" in error_msg:
                logger.error(f"Docling assertion failure for {pdf_path}: {e}")
            elif "qpdf" in error_msg or "cmap" in error_msg:
                logger.error(f"Docling PDF parsing crash for {pdf_path}: {e}")
            else:
                logger.error(f"Docling unexpected error for {pdf_path}: {e}")

            if self.use_fallback:
                logger.info("Attempting fallback extraction")
                return self._extract_fallback(pdf_path)
            else:
                raise RuntimeError(f"Docling conversion failed: {e}") from e

        return {
            'full_text': full_text,
            'markdown': full_text,  # Also provide as 'markdown' key
            'structures': structures,
            'metadata': {
                'extractor': 'docling_v2',
                'num_pages': num_pages,
                'processing_time': None  # Docling doesn't provide this
            }
        }

    def _extract_fallback(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Fallback extraction using PyMuPDF when Docling is not available.
        
        This provides basic text extraction for testing purposes.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            try:
                text_parts = []
                num_pages = len(doc)

                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

                full_text = "\n\n".join(text_parts)

                # Also create markdown version
                markdown_text = full_text.replace("--- Page", "\n## Page")

                return {
                    'full_text': full_text,
                    'markdown': markdown_text,  # Add markdown key
                    'structures': {},
                    'metadata': {
                        'extractor': 'pymupdf_fallback',
                        'num_pages': num_pages,
                        'warning': 'Using fallback extractor - structures not extracted'
                    }
                }
            finally:
                # Always close the document, even if an exception occurs
                doc.close()
        except ImportError:
            # If even PyMuPDF is not available, return minimal stub
            logger.warning("PyMuPDF not available for fallback extraction")
            return {
                'full_text': f"[Stub text extraction from {pdf_path.name}]",
                'markdown': f"[Stub text extraction from {pdf_path.name}]",
                'structures': {},
                'metadata': {
                    'extractor': 'stub',
                    'warning': 'No PDF extraction libraries available'
                }
            }
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return {
                'full_text': '',
                'markdown': '',
                'structures': {},
                'metadata': {
                    'extractor': 'fallback_failed',
                    'error': str(e)
                }
            }

    def extract_batch(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract from multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of extraction results
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.extract(pdf_path)
                result['pdf_path'] = str(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path}: {e}")
                results.append({
                    'pdf_path': str(pdf_path),
                    'full_text': None,
                    'structures': {},
                    'metadata': {'error': str(e)}
                })

        return results
