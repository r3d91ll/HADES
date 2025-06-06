from __future__ import annotations

"""
YAML adapter for the document processing system.

This module provides an adapter for processing YAML files,
focusing on extracting structure, keys, and relationships between components.
"""

import logging
import yaml
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, cast, Collection, MutableMapping
from collections import defaultdict

from src.types.docproc.enums import ContentCategory

from src.types.docproc.adapter import ExtractorOptions, MetadataExtractionConfig, EntityExtractionConfig
from src.types.docproc.document import ProcessedDocument, DocumentEntity, DocumentMetadata
from src.types.docproc.formats.yaml import YAMLNodeInfo, YAMLDocumentInfo, YAMLValidationResult
from .base import BaseAdapter

# Set up logging
logger = logging.getLogger(__name__)


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
        
    def process(self, file_path: Union[str, Path], options: Optional[ExtractorOptions] = None) -> ProcessedDocument:  # type: ignore[override]
        """Process a YAML file or content.
        
        Args:
            file_path: Path to the YAML file
            options: Additional processing options (can contain content as an option)
            
        Returns:
            Processed YAML document
        """

        # Convert to plain dict for easier handling
        options_dict: Dict[str, Any] = {}
        if options is not None:
            # Create a copy of options as a regular dict
            options_dict = dict(options)  
        
        content = None
        # Handle ExtractorOptions properly
        if options is not None:
            content = options.get('content')
        
        # Handle the case where content is provided directly
        if content is not None and isinstance(content, str):
            text = content
            path_obj = Path(str(file_path)) if file_path else Path("")
        else:
            # Process from file path
            path_obj = Path(file_path) if isinstance(file_path, str) else file_path
            
            if not path_obj.exists():
                # Return a properly typed error document
                error_doc: ProcessedDocument = {
                    "id": f"yaml-notfound-{str(uuid.uuid4())[:8]}",
                    "content": "",
                    "raw_content": "",
                    "content_type": "text/yaml",
                    "format": "yaml",
                    "metadata": {},
                    "entities": [],
                    "sections": [],
                    "error": f"File not found: {file_path}",
                    "content_category": ContentCategory.DATA
                }
                return error_doc
                
            # Read the file content
            text = path_obj.read_text(encoding="utf-8", errors="replace")
        
        # Process the YAML content
        result = self.process_text(text)
        
        # Add file metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["path"] = str(path_obj.absolute())
        result["metadata"]["filename"] = path_obj.name
        # Create a properly typed metadata dictionary
        metadata_dict: DocumentMetadata = {
            "language": "yaml",
            "file_type": "yaml",
            "content_category": ContentCategory.DATA,
            "custom": {
                "extension": path_obj.suffix  # Store extension in custom field
            }
        }
        result["metadata"] = metadata_dict
        
        # Set format information
        result["format"] = "yaml"
        result["content_category"] = ContentCategory.DATA  # Use DATA for YAML
        
        # Generate ID if not present - only if needed
        if not result["id"]:
            file_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
            result["id"] = f"yaml_{file_hash}_{path_obj.stem}"
            
        return result  
    
    def extract_metadata(self, content: Union[str, ProcessedDocument], options: Optional[MetadataExtractionConfig] = None) -> DocumentMetadata:  # type: ignore[override]
        """
        Extract metadata from a YAML document.
        
        Args:
            content: YAML content as string or processed document
            options: Optional extraction configuration
            
        Returns:
            Extracted document metadata
        """
        # Convert content to appropriate type
        # Initialize with an empty dict to avoid None issues
        document: Dict[str, Any] = {}
        
        if isinstance(content, str):
            try:
                # Try to parse YAML string to extract metadata
                yaml_data = yaml.safe_load(content)
                if isinstance(yaml_data, dict):
                    # Cast yaml_data to Dict[str, Any] to match document type
                    document = cast(Dict[str, Any], yaml_data)
            except Exception as e:
                logger.warning(f"Failed to parse YAML for metadata extraction: {e}")
        elif isinstance(content, dict):
            # Cast ProcessedDocument to Dict[str, Any]
            document = cast(Dict[str, Any], content)
        metadata = document.get("metadata", {})
        
        # Add any YAML-specific metadata extraction here
        if "symbol_table" in document:
            metadata["key_count"] = len(document["symbol_table"])
        
        # Ensure we have the minimum required fields for DocumentMetadata
        if "title" not in metadata and "filename" in metadata:
            metadata["title"] = metadata["filename"]
            
        return cast(DocumentMetadata, metadata)
    
    def extract_entities(self, content: Union[str, ProcessedDocument], options: Optional[EntityExtractionConfig] = None) -> List[DocumentEntity]:  # type: ignore[override]
        """
        Extract entities from a YAML document.
        
        Args:
            content: YAML content as string or processed document
            options: Optional extraction configuration
            
        Returns:
            List of extracted entities
        """
        # Convert content to appropriate type
        document: Dict[str, Any] = {}
        if isinstance(content, str):
            try:
                # Try to parse YAML string to extract entities
                yaml_data = yaml.safe_load(content)
                if isinstance(yaml_data, dict):
                    # Cast yaml_data to Dict[str, Any] to match document type
                    document = cast(Dict[str, Any], yaml_data)
            except Exception as e:
                logger.warning(f"Failed to parse YAML for entity extraction: {e}")
        elif isinstance(content, dict):
            # Cast ProcessedDocument to Dict[str, Any]
            document = cast(Dict[str, Any], content)
        entities: List[DocumentEntity] = []
        
        # Track entities by key
        entities_by_key: Dict[str, DocumentEntity] = {}
        
        # If we have nodes, convert them to entities
        if "symbol_table" in document:
            for symbol_id, symbol_info in document["symbol_table"].items():
                if symbol_info.get("path", "").count("/") <= 1:  # Only top-level or direct children
                    # Create a DocumentEntity with required fields
                    entity: DocumentEntity = {
                        "type": "yaml_key",
                        "text": symbol_info.get("key", ""),
                        "start": symbol_info.get("line_start", 0),
                        "end": symbol_info.get("line_end", 0),
                        "metadata": {
                            "id": symbol_id,  # Store ID in metadata
                            "value_type": symbol_info.get("value_type", ""),
                            "path": symbol_info.get("path", ""),
                        }
                    }
                    entities.append(entity)
                    entities_by_key[symbol_id] = entity
        
        return entities  
    
    def process_text(self, text: str, options: Optional[ExtractorOptions] = None) -> ProcessedDocument:  # type: ignore[override]
        """
        Process YAML text content.
        
        Args:
            text: YAML content to process
            
        Returns:
            Processed YAML information
        """
        if not text or not text.strip():
            # Return a properly typed error document
            error_doc: ProcessedDocument = {
                "id": f"yaml-notfound-{str(uuid.uuid4())[:8]}",
                "content": "",
                "raw_content": "",
                "content_type": "text/yaml",
                "format": "yaml",
                "metadata": {},
                "entities": [],
                "sections": [],
                "error": "Empty YAML content",
                "content_category": ContentCategory.DATA
            }
            return error_doc
            
        try:
            # Parse the YAML content
            yaml_data = yaml.safe_load(text)
            
            # Start with basic document structure
            doc_id = f"yaml-{str(uuid.uuid4())[:8]}"
            if options is not None and "id" in options:
                doc_id = str(options.get("id", doc_id))
                
            document: ProcessedDocument = {
                "id": doc_id,
                "content": text,
                "content_type": "text/yaml",
                "format": "yaml",
                "raw_content": text,  # Add required raw_content field
                "metadata": {},
                "entities": [],
                "sections": [],  # Add required sections field
                "error": None,  # Add required error field
                "content_category": ContentCategory.DATA  # YAML is data format
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
                
                # Add custom field to metadata if it doesn't exist
                if "custom" not in document["metadata"]:
                    document["metadata"]["custom"] = {}
                    
                # Add to custom metadata
                document["metadata"]["custom"]["symbol_table"] = elements_dict
                document["metadata"]["custom"]["relationships"] = cast(Any, relationships)
                
                # Add structure statistics
                document["metadata"]["custom"]["element_count"] = len(elements_dict)
                document["metadata"]["custom"]["relationship_count"] = len(relationships)
                
            # Return the properly typed document
            return document
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            doc_id = f"yaml-error-{str(uuid.uuid4())[:8]}"
            if options is not None and "id" in options:
                doc_id = str(options.get("id", doc_id))
                
            # Use yaml_error_doc instead of error_doc to avoid redefinition
            yaml_error_doc: ProcessedDocument = {
                "id": doc_id,
                "content": text,
                "raw_content": text,
                "content_type": "text/yaml",
                "format": "yaml",
                "metadata": {},
                "entities": [],
                "sections": [],
                "error": f"YAML parsing error: {str(e)}",
                "content_category": ContentCategory.DATA
            }
            return yaml_error_doc
        except Exception as e:
            logger.error(f"Unexpected error processing YAML: {e}")
            # Use a different variable name to avoid redefinition error
            general_error_doc: ProcessedDocument = {
                "id": doc_id,  # Reuse the doc_id generated above
                "content": text,
                "raw_content": text,
                "content_type": "text/yaml",
                "format": "yaml",
                "metadata": {},
                "entities": [],
                "sections": [],
                "error": f"Processing error: {str(e)}",
                "content_category": ContentCategory.DATA
            }
            return general_error_doc
    
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
        node_map: Dict[str, YAMLNodeInfo] = {}
        relationships: List[Dict[str, Any]] = []
        
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