"""
Python code chunker for HADES-PathRAG.

This module provides a specialized chunker for Python code that breaks down
code files into meaningful semantic chunks based on code structure (functions,
classes, methods) rather than arbitrary text boundaries.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import hashlib
from pathlib import Path

from src.chunking.base import BaseChunker
from src.docproc.adapters.python_adapter import PythonAdapter
from src.docproc.models.python_code import (
    ModuleElement, ClassElement, FunctionElement, MethodElement, 
    CodeRelationship, RelationshipType
)

logger = logging.getLogger(__name__)


class PythonCodeChunker(BaseChunker):
    """
    Chunker specialized for Python code files.
    
    This chunker uses AST parsing to extract meaningful code entities (functions,
    classes, methods) and creates chunks that preserve the semantic structure of
    the code. Each chunk contains a complete code entity with its docstring and
    relationships to other entities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Python code chunker.
        
        Args:
            config: Optional configuration for the chunker
        """
        super().__init__(name="python_code", config=config)
        
        # Configure chunker behavior
        self.include_imports = self.config.get("include_imports", False)
        self.include_docstrings = self.config.get("include_docstrings", True)
        self.include_source = self.config.get("include_source", True)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)  # Minimum characters
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)  # Maximum characters
        
        # Create Python adapter for parsing
        self.python_adapter = PythonAdapter(
            create_symbol_table=True, 
            options={
                "extract_docstrings": True,
                "analyze_imports": True,
                "analyze_calls": True,
                "extract_type_hints": True,
                "compute_complexity": True
            }
        )
        
        logger.info(f"Initialized PythonCodeChunker with config: {self.config}")
    
    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Chunk Python code into semantically meaningful chunks.
        
        Args:
            content: Python source code to chunk (either a string or a dict with 'text' and 'metadata')
            **kwargs: Optional additional arguments, including 'metadata' about the source file
            
        Returns:
            List of chunks, each representing a code entity
        """
        # Handle different content types
        if isinstance(content, dict):
            text = content.get('text', '')
            metadata = content.get('metadata', {})
        else:
            text = content
            metadata = kwargs.get('metadata', {})
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided to PythonCodeChunker")
            return []
        
        # Process Python code using the adapter
        metadata = metadata or {}
        file_path = metadata.get("file_path", "unknown.py")
        
        try:
            # Parse Python code
            logger.info(f"Processing Python code from {file_path}, text length: {len(text)}")
            processed = self.python_adapter.process_text(text)
            
            # Log the structure of processed output for debugging
            logger.info(f"Processed Python code result keys: {list(processed.keys()) if processed else 'None'}")
            
            if not processed or "error" in processed and processed["error"]:
                logger.warning(f"Error processing Python code: {processed.get('error', 'Unknown error')}")
                return self._fallback_chunking(text, metadata)
            
            # Extract chunks from processed code
            chunks = self._extract_code_chunks(processed, metadata)
            
            # Handle case where no valid chunks were extracted
            if not chunks:
                logger.warning(f"No valid code chunks extracted from {file_path}, falling back to text chunking")
                return self._fallback_chunking(text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking Python code: {e}")
            return self._fallback_chunking(text, metadata)
    
    def _extract_code_chunks(self, processed: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract code chunks from processed Python code.
        
        Args:
            processed: Processed Python code from adapter
            metadata: Metadata about the source file
            
        Returns:
            List of code chunks
        """
        chunks = []
        file_path = metadata.get("file_path", "unknown.py")
        file_id = metadata.get("file_id", hashlib.md5(file_path.encode()).hexdigest())
        
        # Extract module-level docstring as a chunk if it exists
        if "symbol_table" in processed and processed["symbol_table"]["docstring"]:
            module_chunk = {
                "chunk_id": f"{file_id}_module",
                "type": "module",
                "text": processed["symbol_table"]["docstring"],
                "embedding": None,  # Will be populated by embedding component
                "metadata": {
                    "source": file_path,
                    "line_range": [1, 1],  # Usually at the top
                    "type": "module_docstring",
                    "name": Path(file_path).stem,
                    "references": []
                }
            }
            chunks.append(module_chunk)
        
        # Process each entity in the symbol table
        if "symbol_table" in processed:
            logger.info(f"Symbol table keys: {list(processed['symbol_table'].keys())}")
            
            if "elements" in processed["symbol_table"]:
                logger.info(f"Found {len(processed['symbol_table']['elements'])} code elements to process")
                for i, element in enumerate(processed["symbol_table"]["elements"]):
                    logger.info(f"Processing element {i+1}/{len(processed['symbol_table']['elements'])}: {element.get('name', 'unnamed')} ({element.get('type', 'unknown')})")
                    chunk = self._process_code_element(element, file_id, file_path)
                    if chunk:
                        chunks.append(chunk)
                    else:
                        logger.warning(f"Failed to create chunk for element: {element.get('name', 'unnamed')}")
            else:
                logger.warning(f"No 'elements' key found in symbol table for {file_path}")
        else:
            logger.warning(f"No 'symbol_table' key found in processed result for {file_path}")
        
        # Process relationships to create references
        if "relationships" in processed and processed["relationships"]:
            self._process_relationships(chunks, processed["relationships"])
        
        return chunks
    
    def _process_code_element(self, element: Dict[str, Any], file_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a code element into a chunk.
        
        Args:
            element: Code element dictionary
            file_id: File identifier
            file_path: Path to the source file
            
        Returns:
            Chunk dictionary or None if element should be skipped
        """
        # Skip imports if configured to do so
        if element["type"] == "import" and not self.include_imports:
            return None
        
        # Get element content
        content = element.get("content", "")
        docstring = element.get("docstring", "")
        
        # Combine content and docstring based on configuration
        if self.include_docstrings and docstring:
            text = f"{docstring}\n\n{content}" if self.include_source else docstring
        else:
            text = content if self.include_source else ""
        
        # Skip if resulting text is too small
        if len(text) < self.min_chunk_size:
            return None
        
        # Truncate if too large
        if len(text) > self.max_chunk_size:
            text = text[:self.max_chunk_size]
        
        # Create chunk ID based on element type and name
        element_type = element["type"]
        element_name = element["name"]
        qualified_name = element.get("qualified_name", element_name)
        
        chunk_id = f"{file_id}_{element_type}_{qualified_name.replace('.', '_')}"
        
        # Get line range
        line_range = element.get("line_range", [0, 0])
        
        # Create chunk
        chunk = {
            "chunk_id": chunk_id,
            "type": element_type,
            "text": text,
            "embedding": None,  # Will be populated by embedding component
            "metadata": {
                "source": file_path,
                "line_range": line_range,
                "type": element_type,
                "name": element_name,
                "qualified_name": qualified_name,
                "references": []
            }
        }
        
        # Add additional metadata based on element type
        if element_type == "function":
            chunk["metadata"]["parameters"] = element.get("parameters", [])
            chunk["metadata"]["returns"] = element.get("returns", None)
            chunk["metadata"]["decorators"] = element.get("decorators", [])
            chunk["metadata"]["complexity"] = element.get("complexity", 0)
        
        elif element_type == "class":
            chunk["metadata"]["base_classes"] = element.get("base_classes", [])
            chunk["metadata"]["decorators"] = element.get("decorators", [])
            
            # Process methods separately
            if "elements" in element:
                for method in element["elements"]:
                    method_chunk = self._process_code_element(
                        method, 
                        file_id, 
                        file_path
                    )
                    if method_chunk:
                        # Add reference to parent class
                        method_chunk["metadata"]["parent_class"] = qualified_name
                        method_chunk["metadata"]["references"].append({
                            "type": "CONTAINED_BY",
                            "target": chunk_id,
                            "weight": 1.0
                        })
        
        elif element_type == "method":
            chunk["metadata"]["parameters"] = element.get("parameters", [])
            chunk["metadata"]["returns"] = element.get("returns", None)
            chunk["metadata"]["decorators"] = element.get("decorators", [])
            chunk["metadata"]["is_static"] = element.get("is_static", False)
            chunk["metadata"]["is_class_method"] = element.get("is_class_method", False)
            chunk["metadata"]["parent_class"] = element.get("parent_class", "")
            chunk["metadata"]["complexity"] = element.get("complexity", 0)
        
        return chunk
    
    def _process_relationships(self, chunks: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        """
        Process relationships between code elements and add them to chunks.
        
        Args:
            chunks: List of chunks to update
            relationships: List of relationships between code elements
        """
        # Create a map of qualified names to chunk IDs
        chunk_map = {}
        for chunk in chunks:
            qualified_name = chunk["metadata"].get("qualified_name")
            if qualified_name:
                chunk_map[qualified_name] = chunk["chunk_id"]
        
        # Process each relationship
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            weight = rel.get("weight", 0.5)
            
            # Skip if source or target is missing
            if not source or not target:
                continue
            
            # Find source and target chunks
            source_id = chunk_map.get(source)
            target_id = chunk_map.get(target)
            
            if source_id and target_id:
                # Find source chunk
                for chunk in chunks:
                    if chunk["chunk_id"] == source_id:
                        # Add reference to the chunk
                        chunk["metadata"]["references"].append({
                            "type": rel_type,
                            "target": target_id,
                            "weight": weight
                        })
                        break
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback to simple text chunking if code parsing fails.
        
        Args:
            text: Text to chunk
            metadata: Metadata about the source
            
        Returns:
            List of simple text chunks
        """
        # Create a single chunk with the entire file
        file_path = metadata.get("file_path", "unknown.py")
        file_id = metadata.get("file_id", hashlib.md5(file_path.encode()).hexdigest())
        
        return [{
            "chunk_id": f"{file_id}_text",
            "type": "text",
            "text": text,
            "embedding": None,
            "metadata": {
                "source": file_path,
                "line_range": [1, text.count("\n") + 1],
                "type": "text",
                "name": Path(file_path).stem,
                "references": []
            }
        }]


# Register the chunker
from src.chunking.registry import register_chunker
register_chunker("python_code", PythonCodeChunker)
