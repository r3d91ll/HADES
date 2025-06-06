from __future__ import annotations

"""
JSON adapter for the document processing system.

This module provides an adapter for processing JSON files,
focusing on extracting structure, keys, and relationships between components.
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Set, cast, Collection, MutableMapping
from pathlib import Path
from src.types.docproc.enums import ContentCategory
from collections import defaultdict

from src.types.docproc.adapter import ExtractorOptions, MetadataExtractionConfig, EntityExtractionConfig
from src.types.docproc.document import ProcessedDocument, DocumentEntity, DocumentMetadata
from src.types.docproc.formats.json import JSONNodeInfo, JSONQueryResult
from .base import BaseAdapter

# Set up logging
logger = logging.getLogger(__name__)


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
        super().__init__(format_type="json")
        self.options = options or {}
        self.create_symbol_table = create_symbol_table
        
    def process(self, file_path: Union[str, Path], options: Optional[ExtractorOptions] = None) -> ProcessedDocument:
        """
        Process a JSON file and convert to standardized format.
        
        Args:
            file_path: Path to the JSON file to process
            options: Optional processing options
            
        Returns:
            Processed document with JSON analysis
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If there's an error reading or parsing the file
        """
        try:
            # Convert to Path object if string
            path_obj = Path(file_path) if isinstance(file_path, str) else file_path
            
            # Check if file exists
            if not path_obj.exists():
                raise FileNotFoundError(f"JSON file not found: {path_obj}")
                
            # Read file content
            with open(path_obj, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Process the text content
            return self.process_text(text, options)
                
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            raise ValueError(f"Error processing JSON file: {str(e)}")
            
    def process_text(self, text: str, options: Optional[ExtractorOptions] = None) -> ProcessedDocument:
        """
        Process JSON text content without an associated file.
        
        Args:
            text: JSON content to process
            options: Optional processing options
            
        Returns:
            Processed document with JSON structure analysis
        """
        if not text or not text.strip():
            raise ValueError("Empty JSON content")
            
        try:
            # Parse the JSON content
            json_data = json.loads(text)
            
            # Generate a unique document ID
            doc_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            doc_id = f"json_{doc_hash}"
            
            # Create basic result structure as a ProcessedDocument
            result: ProcessedDocument = {
                "id": doc_id,
                "content": text,
                "raw_content": text,
                "content_type": "application/json",
                "format": "json",
                "content_category": ContentCategory.DATA,
                "metadata": {
                    "title": f"JSON Document {doc_hash}",
                    "custom": {
                        "source": "memory",
                        "line_count": len(text.split("\n")),
                        "char_count": len(text),
                        "root_elements": 1,  # JSON always has one root element
                        "symbol_table": {},
                        "relationships": []
                    }
                },
                "entities": [],
                "sections": [],
                "error": None
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
                
                result["metadata"]["custom"]["symbol_table"] = elements_dict
                result["metadata"]["custom"]["relationships"] = cast(Any, relationships)
                
                # Add structure statistics to custom metadata
                result["metadata"]["custom"]["element_count"] = len(elements_dict) # Use elements_dict here
                result["metadata"]["custom"]["relationship_count"] = len(relationships)


            # Extract entities and add them to the result
            if options and options.get("extract_entities", True):
                entity_config = options.get("entity_extraction_config") if options else None
                # Cast to proper type for extract_entities
                entity_config_typed = cast(EntityExtractionConfig, entity_config) if entity_config else None
                result["entities"] = self.extract_entities(result, entity_config_typed)
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise ValueError(f"JSON parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing JSON: {e}")
            raise ValueError(f"Processing error: {str(e)}")
    
    def extract_metadata(self, content: Union[str, ProcessedDocument], options: Optional[MetadataExtractionConfig] = None) -> DocumentMetadata:
        """
        Extract metadata from a JSON document.
        
        Args:
            content: JSON content as string or dict
            options: Additional extraction options
            
        Returns:
            Extracted metadata
        """
        extracted_metadata: Dict[str, Any] = {}
        custom_metadata: Dict[str, Any] = {}

        if isinstance(content, str):
            try:
                json_data = json.loads(content)
                if isinstance(json_data, dict):
                    # Extract any top-level keys that might be considered metadata
                    # For example, if the JSON itself is a metadata object
                    for key, value in json_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            # Simple heuristic: add simple type top-level keys to custom
                            custom_metadata[key] = value 
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON string in extract_metadata")
        elif isinstance(content, dict): # It's a ProcessedDocument (TypedDict)
            # Start with a copy of the existing metadata if available
            existing_meta = content.get("metadata")
            if existing_meta:
                extracted_metadata.update(existing_meta)
                # Preserve existing custom metadata
                if "custom" in existing_meta and isinstance(existing_meta["custom"], dict):
                    custom_metadata.update(existing_meta["custom"])

        # Basic default metadata structure
        # Title is often derived or set based on context (e.g., filename)
        # For standalone metadata extraction, it might be missing or come from the content itself.
        final_metadata: DocumentMetadata = {
            "title": extracted_metadata.get("title", "Untitled JSON Document"),
            # Add other required DocumentMetadata fields with defaults if necessary
            # "id": extracted_metadata.get("id", str(uuid.uuid4())), # Example if ID were needed here
        }

        # Merge extracted standard metadata fields
        for key in DocumentMetadata.__annotations__.keys():
            if key in extracted_metadata and key != "custom":
                final_metadata[key] = extracted_metadata[key] # type: ignore
        
        # Add all custom_metadata collected
        if custom_metadata:
            if "custom" not in final_metadata or not isinstance(final_metadata.get("custom"), dict):
                final_metadata["custom"] = {}
            final_metadata["custom"].update(custom_metadata)

        return final_metadata
    
    def extract_entities(self, content: Union[str, ProcessedDocument], options: Optional[EntityExtractionConfig] = None) -> List[DocumentEntity]:
        """
        Extract entities from a JSON document.
        
        Args:
            content: JSON content as string or dict
            options: Additional extraction options
            
        Returns:
            List of extracted entities
        """
        entities: List[DocumentEntity] = []
        symbol_table: Optional[Dict[str, JSONNodeInfo]] = None
        json_text_content: Optional[str] = None

        if isinstance(content, str):
            json_text_content = content
            # If content is a string, we might parse it. For entity extraction based on
            # a symbol table, this string content would typically need to be processed first
            # (e.g., by process_text) to generate that symbol table.
            # For now, if we only have a string and no pre-existing symbol_table from options,
            # this method might return an empty list or rely on simpler string-based entity extraction rules (not implemented here).
            try:
                # Example: try to parse it to see if it's simple data we can work with
                # This is a placeholder for more sophisticated string-based entity extraction logic if needed.
                # For instance, one might use regex or other NLP techniques if options specify so.
                # json.loads(content) # Not used directly here unless we build a symbol table on the fly
                pass
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON string in extract_entities: {content[:100]}...")
                return entities # Return empty if unparsable and no other way to get entities

        elif isinstance(content, dict): # Assumed to be ProcessedDocument
            doc_metadata = content.get("metadata")
            if doc_metadata and isinstance(doc_metadata, dict):
                custom_meta = doc_metadata.get("custom")
                if custom_meta and isinstance(custom_meta, dict):
                    symbol_table = custom_meta.get("symbol_table")
            
            # Try to get raw_content or content for text spans
            raw_content = content.get("raw_content")
            if isinstance(raw_content, str):
                json_text_content = raw_content
            else:
                doc_content = content.get("content")
                if isinstance(doc_content, str):
                    json_text_content = doc_content
            
            # Fallback: if entities are already populated, return them (e.g. by a previous step)
            # This is a common pattern if an upstream process already did entity extraction.
            # However, this adapter's role is often to *create* them from the symbol table.
            # existing_entities = content.get("entities")
            # if existing_entities is not None:
            #    return existing_entities

        # If options provide a way to get entities (e.g., regex patterns, specific keys to extract)
        # that logic would go here. For now, we primarily use the symbol_table.

        if symbol_table and isinstance(symbol_table, dict):
            for node_id, node_info_any in symbol_table.items():
                if isinstance(node_info_any, dict):
                    node_info = cast(JSONNodeInfo, node_info_any)
                    
                    entity_label = node_info.get("type", "json_element")
                    # Use 'name' if available (e.g., for object keys), otherwise the node_id (often a path)
                    entity_identifier_text = node_info.get("name", node_id)
                    
                    start_char = node_info.get("start_char", 0)
                    end_char = node_info.get("end_char", 0)
                    path = node_info.get("path", node_id)

                    # Determine the best text for the entity
                    final_entity_text = entity_identifier_text # Default to identifier
                    if json_text_content and isinstance(start_char, int) and isinstance(end_char, int) and 0 <= start_char < end_char <= len(json_text_content):
                        # If valid span and text available, use the span from raw content
                        final_entity_text = json_text_content[start_char:end_char]
                    elif node_info.get("value_preview") is not None:
                        # Fallback to value_preview if span is not good
                        final_entity_text = str(node_info["value_preview"])

                    entity: DocumentEntity = {
                        "text": str(final_entity_text),  # Ensure it's a string
                        "type": entity_label,  # Use 'type' instead of 'label'
                        "start": start_char or 0,  # Use 'start' instead of 'start_char'
                        "end": end_char or 0,  # Use 'end' instead of 'end_char'
                        "confidence": 1.0,  # Default confidence
                        "metadata": {"path": path, "id": node_id}
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
        elements: Dict[str, Any] = {}
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