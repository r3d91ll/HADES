from __future__ import annotations

"""
YAML adapter for the document processing system.

This module provides an adapter for processing YAML files,
focusing on extracting structure, keys, and relationships between components.
"""

import logging
import yaml
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TypedDict, cast
from collections import defaultdict

from .base import BaseAdapter

# Set up logging
logger = logging.getLogger(__name__)


class YAMLNodeInfo(TypedDict):
    """TypedDict for YAML node information."""
    key: str
    path: str
    line_start: int
    line_end: int
    value_type: str
    value_preview: Optional[str]
    children: List[str]
    parent: Optional[str]


class YAMLAdapter(BaseAdapter):
    """
    Adapter for processing YAML files.
    
    This adapter parses YAML files and extracts their structure, including
    nested objects, lists, and scalar values. It builds a symbol table and
    creates relationships between YAML elements.
    """
    
    def __init__(self, create_symbol_table: bool = True, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the YAML adapter.
        
        Args:
            create_symbol_table: Whether to create a symbol table
            options: Additional options for the adapter
        """
        super().__init__(format_type="yaml")
        self.options = options or {}
        self.create_symbol_table = create_symbol_table
        
    def process(self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a YAML file.
        
        Args:
            file_path: Path to the YAML file
            options: Additional processing options
            
        Returns:
            Processed YAML document
        """
        options = options or {}
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path_obj.exists():
            raise FileNotFoundError(f"YAML file not found: {path_obj}")
            
        # Read the file content
        text = path_obj.read_text(encoding="utf-8", errors="replace")
        
        # Process the YAML content
        result = self.process_text(text)
        
        # Add file metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["path"] = str(path_obj.absolute())
        result["metadata"]["filename"] = path_obj.name
        result["metadata"]["extension"] = path_obj.suffix
        result["metadata"]["language"] = "yaml"
        result["metadata"]["file_type"] = "yaml"
        result["metadata"]["content_category"] = "code"
        
        # Set format information
        result["format"] = "yaml"
        result["content_category"] = "code"
        
        # Generate ID if not present
        if "id" not in result:
            file_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            result["id"] = f"yaml_{file_hash}_{path_obj.stem}"
            
        return result
    
    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a YAML document.
        
        Args:
            document: Processed YAML document
            
        Returns:
            Extracted metadata
        """
        metadata = document.get("metadata", {})
        
        # Add any YAML-specific metadata extraction here
        if "symbol_table" in document:
            metadata["key_count"] = len(document["symbol_table"])
            
        return metadata
    
    def extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from a YAML document.
        
        Args:
            document: Processed YAML document
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract top-level keys as entities
        if "symbol_table" in document:
            for symbol_id, symbol_info in document["symbol_table"].items():
                if symbol_info.get("path", "").count("/") <= 1:  # Only top-level or direct children
                    entities.append({
                        "id": symbol_id,
                        "type": "yaml_key",
                        "name": symbol_info.get("key", ""),
                        "value_type": symbol_info.get("value_type", ""),
                        "path": symbol_info.get("path", ""),
                    })
                    
        return entities
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process YAML text content.
        
        Args:
            text: YAML content to process
            
        Returns:
            Processed YAML information
        """
        if not text or not text.strip():
            return {"error": "Empty YAML content"}
            
        try:
            # Parse the YAML content
            yaml_data = yaml.safe_load(text)
            
            # Create basic result structure
            result = {
                "content_type": "yaml",
                "content_hash": hashlib.md5(text.encode()).hexdigest(),
                "symbol_table": {},
                "relationships": [],
                "metadata": {
                    "line_count": len(text.split("\n")),
                    "char_count": len(text),
                    "root_elements": 1,  # YAML always has one root element
                },
                "original_content": text,
            }
            
            # Extract structure if requested
            if self.create_symbol_table:
                # Create line-to-position mapping for more accurate line numbers
                line_positions = self._create_line_mapping(text)
                
                # Process the YAML structure
                elements, relationships = self._process_yaml_structure(
                    yaml_data, 
                    line_positions, 
                    text
                )
                
                result["symbol_table"] = elements
                result["relationships"] = relationships
                
                # Add structure statistics
                result["metadata"]["element_count"] = len(elements)
                result["metadata"]["relationship_count"] = len(relationships)
                
            return result
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            return {"error": f"YAML parsing error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error processing YAML: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _create_line_mapping(self, text: str) -> Dict[int, int]:
        """
        Create a mapping of line numbers to positions in the text.
        
        Args:
            text: The YAML text
            
        Returns:
            Dictionary mapping line numbers to character positions
        """
        positions = {}
        pos = 0
        for i, line in enumerate(text.split("\n")):
            positions[i+1] = pos
            pos += len(line) + 1  # +1 for the newline
        return positions
    
    def _process_yaml_structure(
        self, 
        data: Any, 
        line_positions: Dict[int, int], 
        original_text: str,
        parent_path: str = "",
        parent_id: Optional[str] = None
    ) -> Tuple[Dict[str, YAMLNodeInfo], List[Dict[str, Any]]]:
        """
        Process the YAML structure recursively.
        
        Args:
            data: YAML data to process
            line_positions: Mapping of line numbers to positions
            original_text: Original YAML text
            parent_path: Path of the parent element
            parent_id: ID of the parent element
            
        Returns:
            Tuple of (elements, relationships)
        """
        elements = {}
        relationships = []
        
        # Process based on data type
        if isinstance(data, dict):
            for key, value in data.items():
                # Create path for this element
                current_path = f"{parent_path}.{key}" if parent_path else key
                element_id = f"yaml_element_{current_path}"
                
                # Get line numbers (estimated)
                # In a real implementation, this would be more precise by using a YAML parser with line info
                line_start = 1  # Default
                line_end = 1
                
                # Create element info
                element_info = {
                    "key": key,
                    "path": current_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "value_type": self._get_type_name(value),
                    "value_preview": self._get_value_preview(value),
                    "children": [],
                    "parent": parent_id
                }
                
                elements[element_id] = element_info
                
                # Create relationship to parent if exists
                if parent_id:
                    relationships.append({
                        "source": parent_id,
                        "target": element_id,
                        "type": "CONTAINS",
                        "metadata": {}
                    })
                
                # Process children recursively
                if isinstance(value, (dict, list)):
                    child_elements, child_relationships = self._process_yaml_structure(
                        value, line_positions, original_text, current_path, element_id
                    )
                    
                    # Update with child information
                    elements.update(child_elements)
                    relationships.extend(child_relationships)
                    
                    # Add child IDs to parent
                    elements[element_id]["children"] = list(child_elements.keys())
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                # Create path for this element
                current_path = f"{parent_path}[{i}]"
                element_id = f"yaml_element_{current_path}"
                
                # Get line numbers (estimated)
                line_start = 1  # Default
                line_end = 1
                
                # Create element info
                element_info = {
                    "key": f"[{i}]",
                    "path": current_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "value_type": self._get_type_name(item),
                    "value_preview": self._get_value_preview(item),
                    "children": [],
                    "parent": parent_id
                }
                
                elements[element_id] = element_info
                
                # Create relationship to parent if exists
                if parent_id:
                    relationships.append({
                        "source": parent_id,
                        "target": element_id,
                        "type": "CONTAINS",
                        "metadata": {}
                    })
                
                # Process children recursively
                if isinstance(item, (dict, list)):
                    child_elements, child_relationships = self._process_yaml_structure(
                        item, line_positions, original_text, current_path, element_id
                    )
                    
                    # Update with child information
                    elements.update(child_elements)
                    relationships.extend(child_relationships)
                    
                    # Add child IDs to parent
                    elements[element_id]["children"] = list(child_elements.keys())
        
        return elements, relationships
    
    def _get_type_name(self, value: Any) -> str:
        """Get the type name of a value."""
        if value is None:
            return "null"
        elif isinstance(value, dict):
            return "mapping"
        elif isinstance(value, list):
            return "sequence"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        else:
            return type(value).__name__
    
    def _get_value_preview(self, value: Any) -> Optional[str]:
        """Get a preview of a value for display."""
        if value is None:
            return "null"
        elif isinstance(value, (dict, list)):
            return None  # No preview for complex types
        elif isinstance(value, str):
            if len(value) > 50:
                return value[:47] + "..."
            return value
        else:
            return str(value)
