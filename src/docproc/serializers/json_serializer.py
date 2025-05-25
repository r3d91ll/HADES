"""
JSON Serializer for Document Processing Results

This module provides functions to standardize the serialization of document processing
results to JSON format, ensuring consistency throughout the processing pipeline.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to serializable formats.
    
    Args:
        obj: Any Python object
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        # Handle custom objects
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    else:
        # Convert anything else to string
        return str(obj)


def serialize_to_json(
    processing_result: Dict[str, Any],
    include_metadata: bool = True,
    include_timestamp: bool = True,
    include_version: bool = True,
    version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Serialize document processing results to a standardized JSON structure.
    
    Args:
        processing_result: The raw result from document processing
        include_metadata: Whether to include metadata in the output
        include_timestamp: Whether to include a timestamp
        include_version: Whether to include version information
        version: The version string to use (if include_version is True)
        
    Returns:
        A dictionary with standardized structure ready for JSON serialization
    """
    # Create the result dictionary with carefully ordered keys
    # Start with core identifying fields
    result = {}
    
    # First add ID, source, and format (critical metadata fields)
    if "id" in processing_result:
        result["id"] = processing_result["id"]
    
    if "source" in processing_result:
        result["source"] = processing_result["source"]
    
    # Include format information if available
    if "format" in processing_result:
        result["format"] = processing_result["format"]
        
    # Include content type information if available
    if "content_type" in processing_result:
        result["content_type"] = processing_result["content_type"]
    
    # Add version information (early in the document)
    if include_version:
        result["version"] = version
        
    # Add timestamp (early in the document)
    if include_timestamp:
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Include document metadata if requested - BEFORE content
    if include_metadata:
        metadata = {}
        
        # Copy existing metadata if present
        original_metadata = processing_result.get("metadata", {})
        if isinstance(original_metadata, dict):
            metadata.update(_make_json_serializable(original_metadata))
        
        # Add processing statistics if present
        if "processing_time" in processing_result:
            metadata["processing_time"] = processing_result["processing_time"]
            
        if metadata:
            result["metadata"] = metadata
    
    # Handle entities specially to ensure standardized format - BEFORE content
    entities = processing_result.get("entities", [])
    if entities:
        result["entities"] = _make_json_serializable(entities)
    
    # Add any remaining fields that don't fit the standard structure
    # but exclude already processed fields and content (which will be added last)
    processed_keys = {"id", "source", "content", "format", "entities", "metadata", "version", "timestamp", "processing_time"}
    for key, value in processing_result.items():
        if key not in processed_keys:
            result[key] = _make_json_serializable(value)
    
    # Add content field LAST to ensure it appears after metadata
    # This is critical for proper downstream processing (e.g., chunkers)
    result["content"] = processing_result.get("content", "")
    
    # Include content type if available (next to content)
    if "content_type" in processing_result:
        result["content_type"] = processing_result["content_type"]
    
    return result


def save_to_json_file(
    processing_result: Dict[str, Any], 
    output_path: Union[str, Path],
    pretty_print: bool = True,
    include_metadata: bool = True,
    include_timestamp: bool = True,
    include_version: bool = True,
    version: str = "1.0.0"
) -> str:
    """
    Serialize document processing results and save to a JSON file.
    
    Args:
        processing_result: The raw result from document processing
        output_path: Path where the JSON file will be saved
        pretty_print: Whether to format the JSON with indentation
        include_metadata: Whether to include metadata 
        include_timestamp: Whether to include a timestamp
        include_version: Whether to include version information
        version: The version string to use (if include_version is True)
        
    Returns:
        The absolute path to the saved JSON file
    """
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
        
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize result
    serialized_result = serialize_to_json(
        processing_result,
        include_metadata=include_metadata,
        include_timestamp=include_timestamp,
        include_version=include_version,
        version=version
    )
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(serialized_result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(serialized_result, f, ensure_ascii=False)
            
    return str(output_path.absolute())
