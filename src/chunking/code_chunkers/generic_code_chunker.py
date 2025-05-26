"""
Generic code chunker for HADES-PathRAG.

This module provides a generic code chunker that can handle various code formats
by applying simple line-based chunking with code-specific boundary detection.
"""

import logging
import hashlib
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path

from ..base import BaseChunker

logger = logging.getLogger(__name__)

# mypy: disable-error-code="unreachable"

class GenericCodeChunker(BaseChunker):
    """Generic code chunker that works with various code formats.
    
    This chunker divides code into logical chunks by detecting common code boundaries
    such as function definitions, class definitions, and import blocks, while also
    respecting a maximum chunk size.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the generic code chunker.
        
        Args:
            config: Configuration for the chunker including max_chunk_size, 
                   min_chunk_size, and overlap
        """
        super().__init__(name="generic_code", config=config or {})
        
        # Set defaults for configuration
        self.max_chunk_size = self.config.get("max_chunk_size", 400)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.overlap = self.config.get("overlap", 20)
        self.respect_boundaries = self.config.get("respect_boundaries", True)
        
        logger.info(f"Initialized GenericCodeChunker with max_size={self.max_chunk_size}, min_size={self.min_chunk_size}")

    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        """Chunk code content using generic code boundaries.
        
        Args:
            text: Text to chunk (alternative to content)
            content: Content to chunk (alternative to text)
            doc_id: Document ID
            path: Path to the original document
            doc_type: Document type
            max_tokens: Maximum tokens per chunk (overrides config if provided)
            output_format: Output format (json or dict)
            
        Returns:
            List of chunks
        """
        # Handle different content types and prepare variables for chunking
        code_content = None
        doc_id = None
        path = 'unknown'
        doc_type = 'code'
        
        # Process based on content type
        # Early return for unsupported content types
        if not isinstance(content, (dict, str)):
            # This is a guard clause that mypy should recognize
            logger.warning(f"Unsupported content type: {type(content)}")
            # This explicit return prevents further processing
            return []  # This line is reachable
            
        # Extract content based on type
        if isinstance(content, dict):
            code_content = content.get('text', '')
            doc_id = content.get('id', kwargs.get('doc_id'))
            path = content.get('path', kwargs.get('path', 'unknown'))
            doc_type = content.get('type', kwargs.get('doc_type', 'code'))
        else:  # content is str
            code_content = content
            doc_id = kwargs.get('doc_id')
            path = kwargs.get('path', 'unknown')
            doc_type = kwargs.get('doc_type', 'code')
        
        # Legacy support for text parameter
        text = kwargs.get('text')
        if code_content is None and text is not None:
            code_content = text
            
        # Handle empty content case
        if code_content is None:
            logger.error("No content or text provided for chunking")
            return []
            
        # Use max_tokens if provided, otherwise use max_chunk_size from config
        # Get size configuration
        max_size = self.max_chunk_size
        max_tokens = kwargs.get('max_tokens')
        if max_tokens:
            # Approximate tokens to characters (rough estimate)
            max_size = max_tokens * 4
        
        # Generate document ID if not provided
        if doc_id is None:
            content_hash = hashlib.md5(code_content.encode()).hexdigest()[:8]
            doc_id = f"code_{content_hash}"
            
        # Split content into lines
        lines = code_content.split("\n")
        
        # Find potential chunk boundaries (functions, classes, blocks)
        boundaries = self._detect_chunk_boundaries(lines)
        
        # Create chunks based on boundaries and size constraints
        chunks = self._create_chunks_from_boundaries(lines, boundaries, path, doc_id, doc_type)
        
        return chunks
        
    def _detect_chunk_boundaries(self, lines: List[str]) -> List[int]:
        """Find potential code boundaries for chunking.
        
        Looks for common code structure elements like function definitions,
        class definitions, import blocks, etc.
        
        Args:
            lines: List of code lines
            
        Returns:
            List of line indices that represent potential chunk boundaries
        """
        boundaries = [0]  # Always include the start of the file
        
        # Common code block starters
        code_block_patterns = [
            # Function/method definitions
            ("def ", True),
            # Class definitions
            ("class ", True),
            # Import statements
            ("import ", False),
            ("from ", False),
            # Module-level constants (ALL_CAPS)
            # If line is ALL_CAPS and contains =
            (lambda line: line.strip() and line.strip()[0].isupper() and "=" in line and 
                         line.strip().split("=")[0].strip().isupper(), False),
            # Block comments and docstrings
            ('"""', False),
            ("'''", False),
            # Control structures that often define logical blocks
            (" if ", False),
            (" for ", False),
            (" while ", False),
            (" try:", False),
            (" except ", False),
            # Blank lines after non-blank lines can be soft boundaries
            (lambda idx, lines: idx > 0 and lines[idx].strip() == "" and lines[idx-1].strip() != "", False)
        ]
        
        for i, line in enumerate(lines):
            if i == 0:
                continue  # Skip the first line as it's already a boundary
                
            line_stripped = line.strip()
            
            # Check for patterns
            for pattern, is_hard_boundary in code_block_patterns:
                if callable(pattern):
                    try:
                        # Safely inspect the callable's parameters
                        sig = inspect.signature(pattern)
                        param_count = len(sig.parameters)
                        
                        if param_count == 1:
                            is_match = pattern(line_stripped)
                        else:
                            is_match = pattern(i, lines)
                    except Exception as e:
                        logger.warning(f"Error inspecting callable pattern: {e}")
                        is_match = False
                elif isinstance(pattern, str):
                    is_match = pattern in line_stripped
                else:
                    is_match = False
                        
                if is_match:
                    boundaries.append(i)
                    break
        
        # Add the end of the file
        boundaries.append(len(lines))
        
        return sorted(set(boundaries))  # Remove duplicates and ensure order
    
    def _create_chunk_from_lines(self, lines: List[str], start_idx: int, end_idx: int, path: str, doc_id: Optional[str], doc_type: str) -> Dict[str, Any]:
        """Create a chunk from a range of lines.
        
        Args:
            lines: List of code lines
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            path: Path to the original document
            doc_id: Document ID
            doc_type: Document type
            
        Returns:
            Chunk dictionary with metadata
        """
        # Get section content
        section_lines = lines[start_idx:end_idx+1] if end_idx >= start_idx else []
        section_text = "\n".join(section_lines)
        
        # Generate a chunk ID
        chunk_id = f"{doc_id or 'chunk'}_{start_idx}_{end_idx}"
        
        # Create the chunk dictionary
        return {
            "id": chunk_id,
            "doc_id": doc_id,
            "text": section_text,
            "metadata": {
                "start_line": start_idx,
                "end_line": end_idx,
                "line_count": len(section_lines),
                "char_count": len(section_text),
                "source_path": path,
                "doc_type": doc_type,
                "chunk_type": "code"
            }
        }
    
    def _create_chunks_from_boundaries(self, lines: List[str], boundaries: List[int], path: str = "unknown", doc_id: Optional[str] = None, doc_type: str = "code") -> List[Dict[str, Any]]:
        """Create code chunks based on detected boundaries and size constraints.
        
        Args:
            lines: List of code lines
            boundaries: List of line indices for chunk boundaries
            path: Path to the original document
            doc_id: Document ID
            doc_type: Document type
            
        Returns:
            List of chunk dictionaries
        """
        # If no boundaries found, create a single chunk
        if len(boundaries) <= 1:
                    # Create a single chunk for all lines
            start_idx, end_idx = 0, len(lines) - 1 if lines else 0
            # Call the correct method to create a chunk
            chunk = self._create_chunk_from_lines(lines, start_idx, end_idx, path, doc_id, doc_type)
            return [chunk]
        
        # Create chunks based on boundaries
        chunks: List[Dict[str, Any]] = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Get the content between boundaries
            section_lines = lines[start_idx:end_idx]
            section_text = "\n".join(section_lines)
            
            # If the section is too large, split it further
            if len(section_text) > self.max_chunk_size and len(section_lines) > 1:
                # Simple approach: Split into roughly equal parts
                mid_point = len(section_lines) // 2
                
                # Create first chunk
                first_half = "\n".join(section_lines[:mid_point])
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": first_half,
                    "metadata": {
                        "source": path,
                        "start_line": start_idx + 1,  # 1-indexed for user readability
                        "end_line": start_idx + mid_point,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
                
                # Create second chunk
                second_half = "\n".join(section_lines[mid_point:])
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": second_half,
                    "metadata": {
                        "source": path,
                        "start_line": start_idx + mid_point + 1,  # 1-indexed
                        "end_line": end_idx,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
            else:
                # Section fits within max_size, create a single chunk
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": section_text,
                    "metadata": {
                        "source": path,
                        "start_line": start_idx + 1,  # 1-indexed for user readability
                        "end_line": end_idx,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
        
        return chunks
