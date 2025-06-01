from __future__ import annotations

from pathlib import Path

"""
JSON adapter for the document processing system.

This module provides an adapter for processing JSON files,
focusing on extracting structure, keys, and relationships between components.
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TypedDict, cast, Collection, MutableMapping
from collections import defaultdict

from pathlib import Path

from .base import BaseAdapter

from pathlib import Path

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
    
    def __init__(self, create_symbol_table: bool = True, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the JSON adapter.
        
        Args:
            create_symbol_table: Whether to create a symbol table
            options: Additional options for the adapter
        """
        super().__init__(format_type="json")  # type: ignore[arg-type]
        self.options = options or {}
        self.create_symbol_table = create_symbol_table
        
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore[override]
        """
        Process JSON text content.
        
        Args:
            text: JSON content to process
            
        Returns:
            Processed JSON information
        """
        if not text or not text.strip():
            return cast(Dict[str, Any], {"error": "Empty JSON content"})
            
        try:
            # Parse the JSON content
            json_data = json.loads(text)
            
            # Create basic result structure
            result: Dict[str, Any] = {
                "content_type": "json",
                "content_hash": hashlib.md5(text.encode()).hexdigest(),
                "symbol_table": {},
                "relationships": [],  # List[Dict[str, Any]]
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
                # Convert elements to the correct type to avoid incompatibility
                # elements is originally Dict[str, JSONNodeInfo] but we need Dict[str, Dict[str, Any]]
                elements_dict: Dict[str, Dict[str, Any]] = {}
                for k, v in elements.items():
                    elements_dict[k] = dict(v)  # Convert JSONNodeInfo to plain dict
                
                result["symbol_table"] = elements_dict
                result["relationships"] = cast(Any, relationships)

                # Add structure statistics - safely with proper types
                if "metadata" not in result:
                    result["metadata"] = {}

                # Create a fully typed metadata dictionary to avoid Collection issues
                metadata_dict: Dict[str, Any] = {}
                if isinstance(result["metadata"], dict):
                    for key, value in result["metadata"].items():
                        metadata_dict[key] = value

                # Now safely set counts
                metadata_dict["element_count"] = len(elements_dict)
                metadata_dict["relationship_count"] = len(relationships)

                # Update the result with the properly typed metadata
                result["metadata"] = metadata_dict

            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return cast(Dict[str, Any], {"error": f"JSON parsing error: {str(e)}"})
        except Exception as e:
            logger.error(f"Unexpected error processing JSON: {e}")
            return cast(Dict[str, Any], {"error": f"Processing error: {str(e)}"})
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore[override]
        """
        Extract metadata from a JSON document.
        
        Args:
            content: JSON content as string or dict
            options: Additional extraction options
            
        Returns:
            Extracted metadata
        """
        document = content if isinstance(content, dict) else {}
        metadata: Dict[str, Any] = {}
        
        # Extract basic metadata
        if "metadata" in document:
            metadata.update(document["metadata"])
            
        return metadata  # type: ignore[return-value]
    
    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:  # type: ignore[override]
        """
        Extract entities from a JSON document.
        
        Args:
            content: JSON content as string or dict
            options: Additional extraction options
            
        Returns:
            List of extracted entities
        """
        document = content if isinstance(content, dict) else {}
        entities: List[Dict[str, Any]] = []
        
        # Extract top-level keys as entities
        if "symbol_table" in document:
            for symbol_id, symbol_info in document["symbol_table"].items() :
                if symbol_info.get("path", "").count("/") <= 1:  # Only top-level or direct children
                    entity = {
                        "id": symbol_id,
                        "type": "json_key",
                        "name": symbol_info.get("key", ""),
                        "value": symbol_info.get("value_preview", ""),
                        "path": symbol_info.get("path", ""),
                    }
                    entities.append(entity)
                    
        return entities
    
    def _create_line_mapping(self, text: str) -> Dict[int, int]:
        """
        Create a mapping of line numbers to positions in the text.
        
        Args:
            text: The JSON text
            
        Returns:
            Dictionary mapping line numbers to character positions
        """
        positions: Dict[str, Any] = {}
        pos = 0
        for i, line in enumerate(text.split("\n")) :
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
        elements: Dict[str, Any] = {}
        relationships: List[Any] = []
        
        # Process based on data type
        if isinstance(data, dict) :
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
                
                elements = cast(Dict[str, Any], dict(elements))

                
                elements[element_id] = element_info  # elements is now properly typed as Dict
                
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
                    elements = cast(Dict[str, Any], dict(elements))

                    elements[element_id]["children"] = list(child_elements.keys())  # elements is now properly typed as Dict
        
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
                
                elements = cast(Dict[str, Any], dict(elements))

                
                elements[element_id] = element_info  # elements is now properly typed as Dict
                
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
                    elements = cast(Dict[str, Any], dict(elements))

                    elements[element_id]["children"] = list(child_elements.keys())  # elements is now properly typed as Dict
        
        # Convert elements and relationships to the correct types
        typed_elements: Dict[str, JSONNodeInfo] = {}
        for key, value in elements.items():
            typed_elements[key] = cast(JSONNodeInfo, value)
            
        # Return with proper types
        return typed_elements, relationships
    
    def _get_type_name(self, value: Any) -> str:
        """Get the type name of a value."""
        if value is None:
            return "null"
        elif isinstance(value, dict) :
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
        elif isinstance(value, (dict, list)) :
            return None  # No preview for complex types
        elif isinstance(value, str):
            if len(value) > 50:
                return value[:47] + "..."
            return value
        else:
            return str(value)