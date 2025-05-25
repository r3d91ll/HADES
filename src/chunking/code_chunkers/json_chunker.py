"""
JSON code chunker for HADES-PathRAG.

This module provides a specialized chunker for JSON files that breaks down
JSON content into meaningful semantic chunks based on structure rather than
arbitrary text boundaries.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import hashlib
from pathlib import Path
from collections import defaultdict

from src.chunking.base import BaseChunker
from src.docproc.adapters.json_adapter import JSONAdapter

logger = logging.getLogger(__name__)


class JSONCodeChunker(BaseChunker):
    """
    Chunker specialized for JSON files.
    
    This chunker parses JSON files to extract meaningful data structures
    and creates chunks that preserve the semantic structure of the JSON content.
    Each chunk contains a complete JSON element with its path and relationships
    to other elements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON code chunker.
        
        Args:
            config: Optional configuration for the chunker
        """
        super().__init__(name="json_code", config=config)
        
        # Configure chunker behavior
        self.include_source = self.config.get("include_source", True)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)  # Minimum characters
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)  # Maximum characters
        
        # Create JSON adapter for parsing
        self.json_adapter = JSONAdapter(
            create_symbol_table=True,
            options={}
        )
        
        logger.info(f"Initialized JSONCodeChunker with config: {self.config}")
    
    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk JSON content into semantically meaningful chunks.
        
        Args:
            content: JSON content to chunk (either a string or a dict with 'text' and 'metadata')
            **kwargs: Optional additional arguments, including 'metadata' about the source file
            
        Returns:
            List of chunks, each representing a JSON element
        """
        # Handle different content types
        if isinstance(content, dict):
            text = content.get('text', '')
            metadata = content.get('metadata', {})
        else:
            text = content
            metadata = kwargs.get('metadata', {})
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided to JSONCodeChunker")
            return []
        
        # Process JSON using the adapter
        metadata = metadata or {}
        file_path = metadata.get("file_path", "unknown.json")
        
        try:
            # Parse JSON
            logger.info(f"Processing JSON from {file_path}, text length: {len(text)}")
            processed = self.json_adapter.process_text(text)
            
            # Log the structure of processed output for debugging
            logger.info(f"Processed JSON result keys: {list(processed.keys()) if processed else 'None'}")
            
            if not processed or "error" in processed and processed["error"]:
                logger.warning(f"Error processing JSON: {processed.get('error', 'Unknown error')}")
                return self._fallback_chunking(text, metadata)
            
            # Extract chunks from processed JSON
            chunks = self._extract_json_chunks(processed, metadata)
            
            # Handle case where no valid chunks were extracted
            if not chunks:
                logger.warning(f"No valid JSON chunks extracted from {file_path}, falling back to text chunking")
                return self._fallback_chunking(text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing JSON with adapter: {e}")
            return self._fallback_chunking(text, metadata)
    
    def _extract_json_chunks(self, processed: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract chunks from the processed JSON structure.
        
        Args:
            processed: Processed JSON data from the adapter
            metadata: Metadata about the source file
            
        Returns:
            List of chunks representing JSON elements
        """
        chunks = []
        
        # Extract the symbol table
        symbol_table = processed.get("symbol_table", {})
        relationships = processed.get("relationships", [])
        original_content = processed.get("original_content", "")
        
        # Create a mapping of element IDs to their relationships
        element_relationships = defaultdict(list)
        for rel in relationships:
            source_id = rel["source"]
            target_id = rel["target"]
            rel_type = rel["type"]
            element_relationships[source_id].append((target_id, rel_type))
        
        # Convert elements to chunks
        file_path = metadata.get("file_path", "unknown.json")
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
                    "element_type": "json_element",
                    "element_key": element_info.get("key", ""),
                    "element_path": element_info.get("path", ""),
                    "value_type": element_info.get("value_type", ""),
                    "line_start": element_info.get("line_start", 0),
                    "line_end": element_info.get("line_end", 0),
                    "parent": element_info.get("parent"),
                    "children": element_info.get("children", []),
                    "chunk_type": "json_element"
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
            original_content: Original JSON content
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
            text: Original JSON text
            metadata: Metadata about the source file
            
        Returns:
            List of text-based chunks
        """
        logger.info("Using fallback chunking for JSON")
        
        chunks = []
        file_path = metadata.get("file_path", "unknown.json")
        file_name = Path(file_path).name
        
        try:
            # For fallback, try to parse the JSON and chunk it by top-level keys
            import json
            data = json.loads(text)
            
            if isinstance(data, dict):
                # For dictionary, create one chunk per top-level key
                for key, value in data.items():
                    # Serialize this portion of the JSON
                    chunk_text = json.dumps({key: value}, indent=2)
                    
                    # Skip if too small
                    if len(chunk_text) < self.min_chunk_size:
                        continue
                        
                    # Create chunk
                    chunk_id = f"json_chunk_{key}"
                    chunks.append({
                        "id": chunk_id,
                        "content": chunk_text,  # Using 'content' to match BaseChunker contract
                        "text": chunk_text,     # Keep 'text' for backward compatibility
                        "metadata": {
                            "file_path": file_path,
                            "file_name": file_name,
                            "element_type": "json_chunk",
                            "element_key": key,
                            "value_type": "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "primitive",
                            "chunk_type": "json_fallback"
                        }
                    })
            elif isinstance(data, list):
                # For list, create chunks of list items (may need to group for small items)
                chunk_size = 5  # Number of list items per chunk
                for i in range(0, len(data), chunk_size):
                    end_idx = min(i + chunk_size, len(data))
                    chunk_data = data[i:end_idx]
                    
                    # Serialize this portion of the JSON
                    chunk_text = json.dumps(chunk_data, indent=2)
                    
                    # Skip if too small
                    if len(chunk_text) < self.min_chunk_size:
                        continue
                        
                    # Create chunk
                    chunk_id = f"json_chunk_list_{i}_{end_idx}"
                    chunks.append({
                        "id": chunk_id,
                        "content": chunk_text,  # Using 'content' to match BaseChunker contract
                        "text": chunk_text,     # Keep 'text' for backward compatibility
                        "metadata": {
                            "file_path": file_path,
                            "file_name": file_name,
                            "element_type": "json_chunk",
                            "element_key": f"items_{i}_{end_idx}",
                            "value_type": "array",
                            "chunk_type": "json_fallback"
                        }
                    })
            else:
                # For primitive values, just create one chunk
                chunk_id = "json_chunk_root"
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "file_name": file_name,
                        "element_type": "json_chunk",
                        "element_key": "root",
                        "value_type": "primitive",
                        "chunk_type": "json_fallback"
                    }
                })
                
        except Exception as e:
            logger.error(f"Error in fallback chunking: {e}")
            # If all else fails, just return the whole document as one chunk
            chunk_id = "json_chunk_full"
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "element_type": "json_chunk",
                    "element_key": "full_document",
                    "chunk_type": "json_fallback_full"
                }
            })
            
        return chunks
