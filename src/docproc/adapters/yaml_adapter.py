from __future__ import annotations

"""
YAML adapter for the document processing system.

This module provides an adapter for processing YAML files,
focusing on extracting structure, keys, and relationships between components.
"""

import logging
import yaml
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TypedDict, cast, Collection, MutableMapping
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
    
    def __init__(self, create_symbol_table: bool = True, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the YAML adapter.
        
        Args:
            create_symbol_table: Whether to create a symbol table
            options: Additional options for the adapter
        """
        super().__init__(format_type="yaml")  # type: ignore[arg-type]
        self.options = options or {}
        self.create_symbol_table = create_symbol_table
        
    def process(self, file_path: Union[str, Path], options: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:  # type: ignore[override]
        """Process a YAML file or content.
        
        Args:
            file_path: Path to the YAML file
            options: Additional processing options (can contain content as an option)
            
        Returns:
            Processed YAML document
        """

        # Process options
        options_dict: Dict[str, Any] = {}
        content = None
        
        if options is None:
            pass
        elif isinstance(options, str):
            content = options
        elif isinstance(options, dict):
            options_dict = options
            content = options.get('content')
        
        # Handle the case where content is provided directly
        if content is not None and isinstance(content, str):
            text = content
            path_obj = Path(str(file_path)) if file_path else Path("")
        else:
            # Process from file path
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
            
        return result  # Return type is already Dict[str, Any]
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore[override]
        document = content if isinstance(content, dict) else {}
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
            
        return metadata  # type: ignore[return-value]
    
    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:  # type: ignore[override]
        document = content if isinstance(content, dict) else {}
        """
        Extract entities from a YAML document.
        
        Args:
            document: Processed YAML document
            
        Returns:
            List of extracted entities
        """
        entities: List[Dict[str, Any]] = []
        
        # Track entities by key
        entities_by_key: Dict[str, Dict[str, Any]] = {}
        
        # If we have nodes, convert them to entities
        if "symbol_table" in document:
            for symbol_id, symbol_info in document["symbol_table"].items() :
                if symbol_info.get("path", "").count("/") <= 1:  # Only top-level or direct children
                    entity = {
                        "id": symbol_id,
                        "type": "yaml_key",
                        "name": symbol_info.get("key", ""),
                        "value_type": symbol_info.get("value_type", ""),
                        "path": symbol_info.get("path", ""),
                    }
                    entities.append(entity)
                    entities_by_key[symbol_id] = entity
        
        return entities
        
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore[override]
        """
        Process YAML text content.
        
        Args:
            text: YAML content to process
            
        Returns:
            Processed YAML information
        """
        if not text or not text.strip() :
            return cast(Dict[str, Any], {"error": "Empty YAML content"})
            
        try:
            # Parse the YAML content
            yaml_data = yaml.safe_load(text)
            
            # Create basic result structure
            result: Dict[str, Any] = {
                "content_type": "yaml",
                "content_hash": hashlib.md5(text.encode()).hexdigest(),
                "symbol_table": {},
                "relationships": [],  # List[Dict[str, Any]]
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
                # Convert elements to the correct type to avoid incompatibility
                # elements is originally Dict[str, YAMLNodeInfo] but we need Dict[str, Dict[str, Any]]
                elements_dict: Dict[str, Dict[str, Any]] = {}
                for k, v in elements.items():
                    elements_dict[k] = dict(v)  # Convert YAMLNodeInfo to plain dict
                
                result["symbol_table"] = elements_dict
                result["relationships"] = cast(Any, relationships)
                
                # Add structure statistics - safely with proper types
                if "metadata" not in result:
                    result["metadata"] = {}
                    
                # Create a fully typed metadata dictionary to avoid Collection issues
                metadata_dict: Dict[str, Any] = {}
                if isinstance(result["metadata"], dict):
                    for k, v in result["metadata"].items():
                        metadata_dict[k] = v
                
                # Set values on our properly typed dictionary
                metadata_dict["element_count"] = len(elements_dict)
                metadata_dict["relationship_count"] = len(relationships)
                
                # Update result with the properly typed metadata
                result["metadata"] = metadata_dict
                
            return result
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            return cast(Dict[str, Any], {"error": f"YAML parsing error: {str(e)}"})
        except Exception as e:
            logger.error(f"Unexpected error processing YAML: {e}")
            return cast(Dict[str, Any], {"error": f"Processing error: {str(e)}"})
    
    def _create_line_mapping(self, text: str) -> Dict[int, int]:
        """
        Create a mapping of line numbers to positions in the text.
        
        Args:
            text: The YAML text
            
        Returns:
            Dictionary mapping line numbers to character positions
        """
        positions: Dict[str, Any] = {}
        pos = 0
        for i, line in enumerate(text.split("\n")) :
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
        node_map: Dict[str, YAMLNodeInfo] = {}
        relationships: List[Dict[str, Any]] = []
        
        # Process based on data type
        if isinstance(data, dict) :
            for key, value in data.items():
                # Create path for this element
                current_path = f"{parent_path}.{key}" if parent_path else key
                element_id = f"yaml_element_{current_path}"
                
                # Get line numbers (estimated)
                # In a real implementation, this would be more precise by using a YAML parser with line info
                line_start = 1  # Default
                line_end = 1
                
                # Create element info
                # Create a properly typed YAMLNodeInfo with all required fields
                element_info: Dict[str, Any] = {
                    "key": key,
                    "path": current_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "value_type": self._get_type_name(value),
                    "value_preview": self._get_value_preview(value),
                    "children": [],
                    "parent": parent_id
                }
                
                # Cast to YAMLNodeInfo before assignment to match expected type
                node_map[element_id] = cast(YAMLNodeInfo, element_info)
                
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
                    # Update with properly typed dictionary
                    for k, v in child_elements.items():
                        node_map[k] = v  # v is already of type Dict[str, Any], compatible with YAMLNodeInfo
                    relationships.extend(child_relationships)
                    
                    # Add child IDs to parent
                    node_map[element_id]["children"] = list(child_elements.keys())
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                # Create path for this element
                current_path = f"{parent_path}[{i}]"
                element_id = f"yaml_element_{current_path}"
                
                # Get line numbers (estimated)
                line_start = 1  # Default
                line_end = 1
                
                # Create element info
                # Create a properly typed YAMLNodeInfo with all required fields
                list_element_info: Dict[str, Any] = {
                    "key": f"[{i}]",
                    "path": current_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "value_type": self._get_type_name(item),
                    "value_preview": self._get_value_preview(item),
                    "children": [],
                    "parent": parent_id
                }
                
                # Cast to YAMLNodeInfo before assignment to match expected type
                node_map[element_id] = cast(YAMLNodeInfo, list_element_info)
                
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
                    # Update with properly typed dictionary
                    for k, v in child_elements.items():
                        node_map[k] = v  # v is already of type Dict[str, Any], compatible with YAMLNodeInfo
                    relationships.extend(child_relationships)
                    
                    # Add child IDs to parent
                    node_map[element_id]["children"] = list(child_elements.keys())
        
        # Update the node map with root info by converting to compatible types
        nodes: Dict[str, YAMLNodeInfo] = {}
        for key, value in node_map.items():
            nodes[key] = cast(YAMLNodeInfo, value)
        
        # Return the node map and relationships list with proper types
        return nodes, relationships
    
    def _get_type_name(self, value: Any) -> str:
        """Get the type name of a value."""
        if value is None:
            return "null"
        elif isinstance(value, dict) :
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
        elif isinstance(value, (dict, list)) :
            return None  # No preview for complex types
        elif isinstance(value, str):
            if len(value) > 50:
                return value[:47] + "..."
            return value
        else:
            return str(value)