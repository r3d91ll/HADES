"""
YAML code chunker for HADES-PathRAG.

This module provides a specialized chunker for YAML files that breaks down
YAML content into meaningful semantic chunks based on structure rather than
arbitrary text boundaries.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import hashlib
from pathlib import Path
from collections import defaultdict

from src.chunking.base import BaseChunker
from src.docproc.adapters.yaml_adapter import YAMLAdapter

logger = logging.getLogger(__name__)


class YAMLCodeChunker(BaseChunker):
    """
    Chunker specialized for YAML files.
    
    This chunker parses YAML files to extract meaningful data structures
    and creates chunks that preserve the semantic structure of the YAML content.
    Each chunk contains a complete YAML element with its path and relationships
    to other elements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the YAML code chunker.
        
        Args:
            config: Optional configuration for the chunker
        """
        super().__init__(name="yaml_code", config=config)  # type: ignore[arg-type]
        
        # Configure chunker behavior
        self.include_comments = self.config.get("include_comments", False)
        self.include_source = self.config.get("include_source", True)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)  # Minimum characters
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)  # Maximum characters
        
        # Create YAML adapter for parsing
        self.yaml_adapter = YAMLAdapter(
            create_symbol_table=True,
            options={
                "extract_comments": self.include_comments,
            }
        )
        
        logger.info(f"Initialized YAMLCodeChunker with config: {self.config}")
    
    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:  # type: ignore[override]
        """
        Chunk YAML content into semantically meaningful chunks.
        
        Args:
            content: YAML content to chunk (either a string or a dict with 'text' and 'metadata')
            **kwargs: Optional additional arguments, including 'metadata' about the source file
            
        Returns:
            List of chunks, each representing a YAML element
        """
        # Handle different content types
        if isinstance(content, dict):
            text = content.get('text', '')
            metadata = content.get('metadata', {})
        else:
            text = content
            metadata = kwargs.get('metadata', {})
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided to YAMLCodeChunker")
            return []
        
        # Process YAML using the adapter
        metadata = metadata or {}
        file_path = metadata.get("file_path", "unknown.yaml")
        
        try:
            # Parse YAML
            logger.info(f"Processing YAML from {file_path}, text length: {len(text)}")
            processed = self.yaml_adapter.process_text(text)
            
            # Log the structure of processed output for debugging
            logger.info(f"Processed YAML result keys: {list(processed.keys()) if processed else 'None'}")
            
            if not processed or "error" in processed and processed["error"]:
                logger.warning(f"Error processing YAML: {processed.get('error', 'Unknown error')}")
                return self._fallback_chunking(text, metadata)
            
            # Extract chunks from processed YAML
            chunks = self._extract_yaml_chunks(processed, metadata)
            
            # Handle case where no valid chunks were extracted
            if not chunks:
                logger.warning(f"No valid YAML chunks extracted from {file_path}, falling back to text chunking")
                return self._fallback_chunking(text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing YAML with adapter: {e}")
            return self._fallback_chunking(text, metadata)
    
    def _extract_yaml_chunks(self, processed: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract chunks from the processed YAML structure.
        
        Args:
            processed: Processed YAML data from the adapter
            metadata: Metadata about the source file
            
        Returns:
            List of chunks representing YAML elements
        """
        chunks = []
        
        # Extract the symbol table
        symbol_table = processed.get("symbol_table", {})
        relationships = processed.get("relationships", [])
        original_content = processed.get("original_content", "")
        
        # Create a mapping of element IDs to their relationships
        element_relationships: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rel in relationships:
            source_id = rel["source"]
            target_id = rel["target"]
            rel_type = rel["type"]
            element_relationships[source_id].append({"target": target_id, "type": rel_type})
        
        # Convert elements to chunks
        file_path = metadata.get("file_path", "unknown.yaml")
        file_name = Path(file_path).name
        
        for element_id, element_info in symbol_table.items():
            # Get element text - in a real implementation, this would extract
            # the relevant lines from the original content based on line numbers
            element_text = self._get_element_text(
                original_content, 
                element_info.get("line_start", 0), 
                element_info.get("line_end", 0)
            )
            
            if not element_text and self.include_source:
                # If we couldn't extract the specific text, use a representation
                element_text = f"{element_info.get('key', '')}: {element_info.get('value_preview', '')}"
            
            # Skip elements that are too small
            if len(element_text) < self.min_chunk_size and not element_info.get("children"):
                continue
                
            # Create the chunk
            chunk = {
                "id": element_id,
                "text": element_text,
                "metadata": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "element_type": "yaml_element",
                    "element_key": element_info.get("key", ""),
                    "element_path": element_info.get("path", ""),
                    "value_type": element_info.get("value_type", ""),
                    "line_start": element_info.get("line_start", 0),
                    "line_end": element_info.get("line_end", 0),
                    "parent": element_info.get("parent"),
                    "children": element_info.get("children", []),
                    "chunk_type": "yaml_element"
                }
            }
            
            # Add relationships
            if element_id in element_relationships:
                chunk["metadata"]["relationships"] = [
                    {"target": target, "type": rel_type}
                    for target, rel_type in element_relationships[element_id]
                ]
            
            chunks.append(chunk)
        
        return chunks
    
    def _get_element_text(self, original_content: str, line_start: int, line_end: int) -> str:
        """
        Extract the text for a specific element from the original content.
        
        Args:
            original_content: Original YAML content
            line_start: Starting line number (1-based)
            line_end: Ending line number (1-based)
            
        Returns:
            Extracted text for the element
        """
        if not original_content or line_start <= 0 or line_end <= 0:
            return ""
            
        try:
            lines = original_content.split("\n")
            # Adjust for 0-based indexing
            line_start_idx = max(0, line_start - 1)
            line_end_idx = min(len(lines), line_end)
            
            return "\n".join(lines[line_start_idx:line_end_idx])
        except Exception as e:
            logger.error(f"Error extracting element text: {e}")
            return ""
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback to simple text-based chunking when structured parsing fails.
        
        Args:
            text: Original YAML text
            metadata: Metadata about the source file
            
        Returns:
            List of text-based chunks
        """
        logger.info("Using fallback chunking for YAML")
        
        # Simple chunking by top-level keys
        chunks: List[Dict[str, Any]] = []
        file_path = metadata.get("file_path", "unknown.yaml")
        file_name = Path(file_path).name
        
        # Split by lines and group by top-level indentation
        lines = text.split("\n")
        current_chunk: List[str] = []
        current_key = ""
        
        for i, line in enumerate(lines):
            # Check if this is a top-level key (no indentation)
            stripped = line.lstrip()
            if stripped and not line.startswith(" ") and ":" in stripped:
                # Save the previous chunk if it exists
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_id = f"yaml_chunk_{len(chunks)}"
                        chunks.append({
                            "id": chunk_id,
                            "content": chunk_text,  # Using 'content' to match BaseChunker contract
                            "text": chunk_text,     # Keep 'text' for backward compatibility
                            "metadata": {
                                "file_path": file_path,
                                "file_name": file_name,
                                "element_type": "yaml_chunk",
                                "element_key": current_key,
                                "line_start": i - len(current_chunk) + 1,
                                "line_end": i,
                                "chunk_type": "yaml_fallback"
                            }
                        })
                
                # Start a new chunk
                current_key = stripped.split(":")[0].strip()
                current_chunk = [line]
            else:
                # Continue the current chunk
                current_chunk.append(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = f"yaml_chunk_{len(chunks)}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,  # Using 'content' to match BaseChunker contract
                    "text": chunk_text,     # Keep 'text' for backward compatibility
                    "metadata": {
                        "file_path": file_path,
                        "file_name": file_name,
                        "element_type": "yaml_chunk",
                        "element_key": current_key,
                        "line_start": len(lines) - len(current_chunk) + 1,
                        "line_end": len(lines),
                        "chunk_type": "yaml_fallback"
                    }
                })
        
        return chunks
