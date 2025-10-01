"""
LaTeX Source Extractor
======================

Extracts text and structures from ArXiv LaTeX sources.
Provides perfect equation extraction and structure preservation.

This extractor represents the highest fidelity CONVEYANCE dimension -
extracting information directly from the author's source with no OCR errors
or formatting ambiguities.
"""

import gzip
import logging
import re
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class LaTeXExtractor:
    """
    Extract content from LaTeX source files.
    
    This provides perfect extraction of equations, tables, and structure
    directly from the LaTeX source, avoiding all PDF parsing issues.
    
    In Actor-Network Theory terms, this is the most direct translation
    from author intent to machine representation.
    """

    def __init__(self, use_pandoc: bool = False):
        """
        Initialize LaTeX extractor.
        
        Args:
            use_pandoc: Whether to use pandoc for LaTeX to markdown conversion
        """
        self.use_pandoc = use_pandoc
        # Track pandoc conversion stats
        self.pandoc_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }
        logger.info(f"Initialized LaTeX extractor (pandoc: {use_pandoc})")

    def extract(self, latex_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from LaTeX source archive.
        
        Args:
            latex_path: Path to .gz file containing LaTeX source
            
        Returns:
            Dictionary containing:
            - full_text: Complete text content (LaTeX or markdown)
            - structures: Extracted structures (equations, tables, citations)
            - metadata: Processing metadata
        """
        latex_path = Path(latex_path)

        if not latex_path.exists():
            raise FileNotFoundError(f"LaTeX source not found: {latex_path}")

        start_time = datetime.now()

        try:
            # First check if it's a tar.gz or just a plain gzipped file
            with gzip.open(latex_path, 'rb') as gz:
                # Try to read first few bytes to detect format
                initial_bytes = gz.read(512)
                gz.seek(0)  # Reset for actual reading

                # Check if it's a tar archive (tar magic bytes)
                # Check if it's a tar archive (tar magic bytes at offset 257)
                # In a tar file, 'ustar' appears at bytes 257-261 of the header
                is_tar = len(initial_bytes) >= 262 and initial_bytes[257:262] == b'ustar'

                # If not tar, check if it looks like LaTeX
                if not is_tar:
                    try:
                        # Try to decode as text
                        text_sample = initial_bytes.decode('utf-8', errors='ignore')
                        # Check for common LaTeX patterns
                        is_latex = ('\\documentclass' in text_sample or
                                  '\\begin{document}' in text_sample or
                                  '\\section' in text_sample or
                                  '\\usepackage' in text_sample)
                    except:
                        is_latex = False
                else:
                    is_latex = False

            # Handle based on file type
            if is_tar or not is_latex:
                # Extract from gzipped tar archive (original code path)
                return self._extract_from_tar_gz(latex_path, start_time)
            else:
                # Handle plain gzipped .tex file
                return self._extract_from_plain_gz(latex_path, start_time)

        except Exception as e:
            logger.error(f"LaTeX extraction failed for {latex_path}: {e}")
            return self._empty_result(latex_path, str(e))

    def _extract_from_tar_gz(self, latex_path: Path, start_time: datetime) -> Dict[str, Any]:
        """Extract from tar.gz archive (multiple files)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Extract the tar.gz file
                with gzip.open(latex_path, 'rb') as gz:
                    with tarfile.open(fileobj=gz, mode='r') as tar:
                        tar.extractall(temp_path)
            except tarfile.TarError as e:
                # Not a valid tar file, might be plain gzipped
                logger.debug(f"Not a tar archive, trying plain gzip: {e}")
                return self._extract_from_plain_gz(latex_path, start_time)

            # Find main .tex file (usually the largest)
            tex_files = list(temp_path.glob("**/*.tex"))
            if not tex_files:
                logger.warning(f"No .tex files found in {latex_path}")
                return self._empty_result(latex_path, "No .tex files found")

            # Sort by size and pick largest as main file
            main_tex = max(tex_files, key=lambda f: f.stat().st_size)
            logger.info(f"Processing main LaTeX file: {main_tex.name}")

            # Read LaTeX content
            try:
                with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                    latex_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read LaTeX file: {e}")
                return self._empty_result(latex_path, str(e))

            # Extract structured content
            structures = {
                'equations': self._extract_equations(latex_content),
                'tables': self._extract_tables(latex_content),
                'citations': self._extract_citations(latex_content),
                'sections': self._extract_sections(latex_content)
            }

            # Check for bibliography file
            bbl_files = list(temp_path.glob("**/*.bbl"))
            if bbl_files:
                try:
                    with open(bbl_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        structures['bibliography'] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read bibliography: {e}")

            # Convert to markdown if requested
            pandoc_error = None
            if self.use_pandoc:
                full_text, pandoc_error = self._convert_to_markdown(main_tex)
                if not full_text:
                    full_text = latex_content  # Fallback to raw LaTeX
            else:
                full_text = latex_content

            duration = (datetime.now() - start_time).total_seconds()

            metadata = {
                'extractor': 'latex',
                'extraction_type': 'tar.gz',
                'main_file': main_tex.name,
                'num_tex_files': len(tex_files),
                'has_bibliography': len(bbl_files) > 0,
                'processing_time': duration,
                'latex_path': str(latex_path)
            }

            # Add Pandoc error if conversion failed
            if pandoc_error:
                metadata['pandoc_error'] = pandoc_error

            return {
                'full_text': full_text,
                'latex_source': latex_content,  # Always include raw LaTeX
                'structures': structures,
                'metadata': metadata
            }

    def _extract_from_plain_gz(self, latex_path: Path, start_time: datetime) -> Dict[str, Any]:
        """
        Extract from plain gzipped .tex file (single file).
        
        Some ArXiv papers store LaTeX as a single gzipped .tex file
        rather than a tar.gz archive.
        """
        try:
            # Read the gzipped LaTeX directly
            with gzip.open(latex_path, 'rt', encoding='utf-8', errors='ignore') as gz:
                latex_content = gz.read()

            if not latex_content:
                logger.warning(f"Empty LaTeX content from {latex_path}")
                return self._empty_result(latex_path, "Empty LaTeX file")

            # Extract structured content
            structures = {
                'equations': self._extract_equations(latex_content),
                'tables': self._extract_tables(latex_content),
                'citations': self._extract_citations(latex_content),
                'sections': self._extract_sections(latex_content)
            }

            # For plain gzipped files, we need to write to temp for Pandoc
            pandoc_error = None
            full_text = latex_content

            if self.use_pandoc:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as temp_tex:
                    temp_tex.write(latex_content)
                    temp_tex_path = Path(temp_tex.name)

                try:
                    full_text_converted, pandoc_error = self._convert_to_markdown(temp_tex_path)
                    if full_text_converted:
                        full_text = full_text_converted
                finally:
                    # Clean up temp file
                    temp_tex_path.unlink(missing_ok=True)

            duration = (datetime.now() - start_time).total_seconds()

            metadata = {
                'extractor': 'latex',
                'extraction_type': 'plain.gz',
                'main_file': latex_path.name,
                'num_tex_files': 1,
                'has_bibliography': False,  # No separate bbl in plain gz
                'processing_time': duration,
                'latex_path': str(latex_path)
            }

            # Add Pandoc error if conversion failed
            if pandoc_error:
                metadata['pandoc_error'] = pandoc_error

            logger.info(f"Successfully extracted from plain gzipped LaTeX: {latex_path.name}")

            return {
                'full_text': full_text,
                'latex_source': latex_content,
                'structures': structures,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Failed to extract plain gzipped LaTeX from {latex_path}: {e}")
            return self._empty_result(latex_path, str(e))

    def _extract_equations(self, latex: str) -> List[Dict[str, Any]]:
        """Extract all equations from LaTeX source."""
        equations = []

        # Display equations: \begin{equation}...\end{equation}
        display_eqs = re.findall(
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            latex,
            re.DOTALL
        )
        for eq in display_eqs:
            # Check for label
            label_match = re.search(r'\\label\{([^}]+)\}', eq)
            label = label_match.group(1) if label_match else None
            # Clean equation text
            eq_text = re.sub(r'\\label\{[^}]+\}', '', eq).strip()
            equations.append({
                'type': 'display',
                'latex': eq_text,
                'label': label
            })

        # Numbered equations: \begin{equation*}...\end{equation*}
        display_star_eqs = re.findall(
            r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}',
            latex,
            re.DOTALL
        )
        for eq in display_star_eqs:
            equations.append({
                'type': 'display_unnumbered',
                'latex': eq.strip(),
                'label': None
            })

        # Align environments
        align_eqs = re.findall(
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            latex,
            re.DOTALL
        )
        for eq in align_eqs:
            label_match = re.search(r'\\label\{([^}]+)\}', eq)
            label = label_match.group(1) if label_match else None
            eq_text = re.sub(r'\\label\{[^}]+\}', '', eq).strip()
            equations.append({
                'type': 'align',
                'latex': eq_text,
                'label': label
            })

        # Inline math: $...$ (limit to reasonable length to avoid false positives)
        inline_math = re.findall(r'\$([^$]{2,200})\$', latex)
        for eq in inline_math[:100]:  # Limit to first 100 to avoid noise
            equations.append({
                'type': 'inline',
                'latex': eq,
                'label': None
            })

        logger.info(f"Extracted {len(equations)} equations")
        return equations

    def _extract_tables(self, latex: str) -> List[Dict[str, Any]]:
        """Extract all tables from LaTeX source."""
        tables = []

        # Find table environments
        table_pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
        table_matches = re.findall(table_pattern, latex, re.DOTALL)

        for table_content in table_matches:
            # Extract caption
            caption_match = re.search(r'\\caption\{((?:[^{}]|\{[^}]*\})*)\}', table_content)
            caption = caption_match.group(1) if caption_match else None

            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', table_content)
            label = label_match.group(1) if label_match else None

            # Extract tabular content
            tabular_match = re.search(
                r'\\begin\{tabular\}(.*?)\\end\{tabular\}',
                table_content,
                re.DOTALL
            )

            if not tabular_match:
                # Try array environment (sometimes used for tables)
                tabular_match = re.search(
                    r'\\begin\{array\}(.*?)\\end\{array\}',
                    table_content,
                    re.DOTALL
                )

            tabular = tabular_match.group(0) if tabular_match else table_content

            # Parse column specification
            col_spec_match = re.search(r'\\begin\{(?:tabular|array)\}\{([^}]+)\}', tabular)
            col_spec = col_spec_match.group(1) if col_spec_match else None

            tables.append({
                'caption': caption,
                'label': label,
                'latex': tabular,
                'column_spec': col_spec,
                'full_content': table_content
            })

        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def _extract_citations(self, latex: str) -> List[str]:
        """Extract all citation keys from LaTeX source."""
        citations = []

        # Find \cite commands (including variants like \citep, \citet)
        cite_pattern = r'\\cite[pt]?\{([^}]+)\}'
        for match in re.finditer(cite_pattern, latex):
            # Split multiple citations
            cite_keys = match.group(1).split(',')
            citations.extend([key.strip() for key in cite_keys])

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for cite in citations:
            if cite not in seen:
                seen.add(cite)
                unique_citations.append(cite)

        logger.info(f"Extracted {len(unique_citations)} unique citations")
        return unique_citations

    def _extract_sections(self, latex: str) -> List[Dict[str, Any]]:
        """Extract section structure from LaTeX source."""
        sections = []

        # Find all section commands
        section_pattern = r'\\(section|subsection|subsubsection|paragraph)\{([^}]+)\}'
        for match in re.finditer(section_pattern, latex):
            sections.append({
                'level': match.group(1),
                'title': match.group(2),
                'position': match.start()
            })

        logger.info(f"Extracted {len(sections)} sections")
        return sections

    def _convert_to_markdown(self, tex_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert LaTeX to markdown using pandoc if available.
        
        Returns:
            Tuple of (markdown_text, error_message)
        """
        self.pandoc_stats['attempts'] += 1
        try:
            import subprocess
            result = subprocess.run(
                ['pandoc', '-f', 'latex', '-t', 'markdown', str(tex_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.pandoc_stats['successes'] += 1
                logger.info(f"Successfully converted LaTeX to markdown (success rate: {self.pandoc_stats['successes']}/{self.pandoc_stats['attempts']})")
                return result.stdout, None
            else:
                self.pandoc_stats['failures'] += 1
                # Log at debug level to reduce noise - these are common with academic LaTeX
                # Extract just the error type for cleaner logging
                error_lines = result.stderr.strip().split('\n')
                error_message = result.stderr.strip()  # Keep full error for database

                if error_lines:
                    # Get first line of error which usually has the key info
                    error_summary = error_lines[0]
                    if "expecting" in result.stderr:
                        # This is a LaTeX structure mismatch - very common
                        logger.debug(f"Pandoc: LaTeX structure issue - {error_summary}")
                    else:
                        logger.debug(f"Pandoc conversion issue: {error_summary}")
                # Log stats periodically
                if self.pandoc_stats['attempts'] % 10 == 0:
                    success_rate = (self.pandoc_stats['successes'] / self.pandoc_stats['attempts']) * 100
                    logger.info(f"Pandoc conversion rate: {success_rate:.1f}% ({self.pandoc_stats['successes']}/{self.pandoc_stats['attempts']})")
                return None, error_message
        except Exception as e:
            logger.warning(f"Pandoc not available or failed: {e}")
            return None, str(e)

    def _empty_result(self, latex_path: Path, error: str) -> Dict[str, Any]:
        """Return empty result structure for failed extraction."""
        return {
            'full_text': '',
            'latex_source': '',
            'structures': {
                'equations': [],
                'tables': [],
                'citations': [],
                'sections': []
            },
            'metadata': {
                'extractor': 'latex',
                'error': error,
                'latex_path': str(latex_path)
            }
        }

    def extract_batch(self, latex_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract from multiple LaTeX sources.
        
        Args:
            latex_paths: List of LaTeX archive paths
            
        Returns:
            List of extraction results
        """
        results = []
        for latex_path in latex_paths:
            try:
                result = self.extract(latex_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {latex_path}: {e}")
                results.append(self._empty_result(Path(latex_path), str(e)))

        return results
