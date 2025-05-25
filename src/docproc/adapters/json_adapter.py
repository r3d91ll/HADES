from __future__ import annotations

"""
JSON adapter for the document processing system.

This module provides an adapter for processing JSON files,
focusing on extracting structure, keys, and relationships between components.
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TypedDict, cast
from collections import defaultdict

from .base import BaseAdapter

# Set up logging
logger = logging.getLogger(__name__)


class JSONNodeInfo(TypedDict):
    """TypedDict for JSON node information."""
    key: str
    path: str
    line_start: int
    line_end: int
    value_type: str
    value_preview: Optional[str]
    children: List[str]
    parent: Optional[str]


class JSONAdapter(BaseAdapter):
    """
    Adapter for processing JSON files.
    
    This adapter parses JSON files and extracts their structure, including
    nested objects, arrays, and primitive values. It builds a symbol table and
    creates relationships between JSON elements.
    """
    
    def __init__(self, create_symbol_table: bool = True, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON adapter.
        
        Args:
            create_symbol_table: Whether to create a symbol table
            options: Additional options for the adapter
        """
        super().__init__(name="json", options=options or {})
        self.create_symbol_table = create_symbol_table
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process JSON text content.
        
        Args:
            text: JSON content to process
            
        Returns:
            Processed JSON information
        """
        if not text or not text.strip():
            return {"error": "Empty JSON content"}
            
        try:
            # Parse the JSON content
            json_data = json.loads(text)
            
            # Create basic result structure
            result = {
                "content_type": "json",
                "content_hash": hashlib.md5(text.encode()).hexdigest(),
                "symbol_table": {},
                "relationships": [],
                "metadata": {
                    "line_count": len(text.split("\n")),
                    "char_count": len(text),
                    "root_elements": 1,  # JSON always has one root element
                },
                "original_content": text,
            }
            
            # Extract structure if requested
            if self.create_symbol_table:
                # Create line-to-position mapping for more accurate line numbers
                line_positions = self._create_line_mapping(text)
                
                # Process the JSON structure
                elements, relationships = self._process_json_structure(
                    json_data, 
                    line_positions, 
                    text
                )
                
                result["symbol_table"] = elements
                result["relationships"] = relationships
                
                # Add structure statistics
                result["metadata"]["element_count"] = len(elements)
                result["metadata"]["relationship_count"] = len(relationships)
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return {"error": f"JSON parsing error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error processing JSON: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _create_line_mapping(self, text: str) -> Dict[int, int]:
        """
        Create a mapping of line numbers to positions in the text.
        
        Args:
            text: The JSON text
            
        Returns:
            Dictionary mapping line numbers to character positions
        """
        positions = {}
        pos = 0
        for i, line in enumerate(text.split("\n")):
            positions[i+1] = pos
            pos += len(line) + 1  # +1 for the newline
        return positions
    
    def _process_json_structure(
        self, 
        data: Any, 
        line_positions: Dict[int, int], 
        original_text: str,
        parent_path: str = "",
        parent_id: Optional[str] = None
    ) -> Tuple[Dict[str, JSONNodeInfo], List[Dict[str, Any]]]:
        """
        Process the JSON structure recursively.
        
        Args:
            data: JSON data to process
            line_positions: Mapping of line numbers to positions
            original_text: Original JSON text
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
                element_id = f"json_element_{current_path}"
                
                # Get line numbers (estimated)
                # In a real implementation, this would use a JSON parser with line info
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
                    child_elements, child_relationships = self._process_json_structure(
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
                element_id = f"json_element_{current_path}"
                
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
                    child_elements, child_relationships = self._process_json_structure(
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
            return "object"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, float):
            return "number"
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
