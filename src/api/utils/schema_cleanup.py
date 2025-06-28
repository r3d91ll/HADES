"""
Schema cleanup utilities for FastAPI-MCP integration.

This module provides utilities to clean up JSON schemas that violate
JSON Schema draft 2020-12 specification, specifically the issue where
both 'anyOf' and top-level 'type' fields are present.
"""

import logging
from typing import Dict, Any, Union, Optional

# Import json with error handling for circular imports
try:
    import json
except ImportError:
    # Fallback for debugging if json import fails
    class MockJson:
        def dumps(self, obj: Any, indent: Optional[int] = None) -> str:
            return str(obj)
    json = MockJson()  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def clean_json_schema(schema: Union[Dict[str, Any], Any]) -> Union[Dict[str, Any], Any]:
    """
    Clean a JSON schema by removing problematic field combinations.
    
    Specifically handles the issue where FastAPI-MCP generates schemas with both
    'anyOf' and top-level 'type' fields, which violates JSON Schema draft 2020-12.
    
    Args:
        schema: The JSON schema to clean (can be dict or any other type)
        
    Returns:
        Cleaned schema with problematic fields removed
        
    Example:
        >>> schema = {
        ...     "anyOf": [{"type": "string"}, {"type": "null"}],
        ...     "type": "string",
        ...     "title": "database_name",
        ...     "description": "Database name"
        ... }
        >>> cleaned = clean_json_schema(schema)
        >>> print(cleaned)
        {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "title": "database_name", 
            "description": "Database name"
        }
    """
    # Only process dictionaries
    if not isinstance(schema, dict):
        return schema
    
    # Create a copy to avoid mutating the original
    cleaned_schema = schema.copy()
    
    # Recursively clean nested schemas first
    for key, value in cleaned_schema.items():
        if isinstance(value, dict):
            cleaned_schema[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned_schema[key] = [clean_json_schema(item) for item in value]
    
    # Fix JSON Schema 2020-12 violations:
    # Remove conflicting type fields with composition operators
    composition_keys = ['anyOf', 'oneOf', 'allOf']
    has_composition = any(key in cleaned_schema for key in composition_keys)
    
    if has_composition and 'type' in cleaned_schema:
        composition_type = next((key for key in composition_keys if key in cleaned_schema), None)
        title = cleaned_schema.get('title', 'unnamed')
        logger.info(f"🔧 Removing conflicting 'type' field from schema with '{composition_type}': {title}")
        logger.debug(f"Schema before fix: {json.dumps(cleaned_schema, indent=2)}")
        del cleaned_schema['type']
        logger.debug(f"Schema after fix: {json.dumps(cleaned_schema, indent=2)}")
    
    # Validate 'type' field format (must be string or array of strings)
    if 'type' in cleaned_schema:
        type_value = cleaned_schema['type']
        if not isinstance(type_value, (str, list)):
            logger.info(f"🔧 Removing invalid 'type' field (not string or array): {type_value}")
            del cleaned_schema['type']
        elif isinstance(type_value, list):
            # Ensure all items in type array are strings
            if not all(isinstance(t, str) for t in type_value):
                logger.info(f"🔧 Removing invalid 'type' array (contains non-strings): {type_value}")
                del cleaned_schema['type']
            elif len(type_value) == 0:
                logger.info(f"🔧 Removing empty 'type' array")
                del cleaned_schema['type']
    
    # Fix additional JSON Schema 2020-12 violations
    
    # 1. Fix invalid 'additionalProperties' values
    if 'additionalProperties' in cleaned_schema:
        add_props = cleaned_schema['additionalProperties']
        if not isinstance(add_props, (bool, dict)):
            logger.info(f"🔧 Removing invalid 'additionalProperties': {type(add_props)} (must be boolean or object)")
            del cleaned_schema['additionalProperties']
    
    # 2. Fix invalid 'required' field
    if 'required' in cleaned_schema:
        required_value = cleaned_schema['required']
        if not isinstance(required_value, list):
            logger.info(f"🔧 Removing invalid 'required': {type(required_value)} (must be array)")
            del cleaned_schema['required']
        elif not all(isinstance(item, str) for item in required_value):
            logger.info(f"🔧 Removing invalid 'required' array (contains non-strings)")
            del cleaned_schema['required']
        elif len(required_value) == 0:
            logger.info(f"🔧 Removing empty 'required' array")
            del cleaned_schema['required']
    
    # 3. Fix invalid 'properties' field
    if 'properties' in cleaned_schema:
        if not isinstance(cleaned_schema['properties'], dict):
            logger.info(f"🔧 Removing invalid 'properties': {type(cleaned_schema['properties'])} (must be object)")
            del cleaned_schema['properties']
    
    # 4. Fix invalid 'items' field
    if 'items' in cleaned_schema:
        items_value = cleaned_schema['items']
        if not isinstance(items_value, (dict, list, bool)):
            logger.info(f"🔧 Removing invalid 'items': {type(items_value)} (must be object, array, or boolean)")
            del cleaned_schema['items']
    
    # 5. Fix $ref fields that might have invalid values
    if '$ref' in cleaned_schema:
        ref_value = cleaned_schema['$ref']
        if not isinstance(ref_value, str):
            logger.info(f"🔧 Removing invalid '$ref': {type(ref_value)} (must be string)")
            del cleaned_schema['$ref']
        elif not ref_value.strip():
            logger.info(f"🔧 Removing empty '$ref' field")
            del cleaned_schema['$ref']
    
    # 6. Remove deprecated fields that are not valid in JSON Schema 2020-12
    deprecated_fields = ['definitions', 'id']  # 'definitions' -> '$defs', 'id' -> '$id'
    for field in deprecated_fields:
        if field in cleaned_schema:
            logger.info(f"🔧 Removing deprecated field '{field}' (not valid in JSON Schema 2020-12)")
            del cleaned_schema[field]
    
    # 7. Fix invalid enum values
    if 'enum' in cleaned_schema:
        enum_value = cleaned_schema['enum']
        if not isinstance(enum_value, list):
            logger.info(f"🔧 Removing invalid 'enum': {type(enum_value)} (must be array)")
            del cleaned_schema['enum']
        elif len(enum_value) == 0:
            logger.info(f"🔧 Removing empty 'enum' array")
            del cleaned_schema['enum']
    
    # 8. Fix duplicate values in required array
    if 'required' in cleaned_schema and isinstance(cleaned_schema['required'], list):
        original_required = cleaned_schema['required']
        unique_required = list(dict.fromkeys(original_required))  # Preserves order, removes duplicates
        if len(unique_required) != len(original_required):
            logger.info(f"🔧 Removing duplicate values from 'required' array: {original_required} -> {unique_required}")
            cleaned_schema['required'] = unique_required
    
    # 9. Simplify anyOf patterns that could cause client issues
    if 'anyOf' in cleaned_schema:
        any_of = cleaned_schema['anyOf']
        if isinstance(any_of, list) and len(any_of) == 2:
            # Check for common pattern: anyOf with null
            non_null_schemas = [schema for schema in any_of if not (isinstance(schema, dict) and schema.get('type') == 'null')]
            null_schemas = [schema for schema in any_of if isinstance(schema, dict) and schema.get('type') == 'null']
            
            if len(null_schemas) == 1 and len(non_null_schemas) == 1:
                non_null_schema = non_null_schemas[0]
                if isinstance(non_null_schema, dict) and 'type' in non_null_schema:
                    # Convert anyOf[{type: X}, {type: null}] to nullable pattern
                    logger.info(f"🔧 Simplifying anyOf null pattern for better client compatibility")
                    # Remove anyOf and flatten to the main type
                    cleaned_schema.update(non_null_schema)
                    if 'anyOf' in cleaned_schema:
                        del cleaned_schema['anyOf']
                    # Note: this makes the field effectively non-nullable for better compatibility
    
    return cleaned_schema


def clean_openapi_spec(openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean an entire OpenAPI specification by fixing problematic schemas.
    
    This function walks through the OpenAPI spec and cleans all schema definitions
    in components/schemas and any inline schemas in paths.
    
    Args:
        openapi_spec: The OpenAPI specification dictionary
        
    Returns:
        Cleaned OpenAPI specification
    """
    if not isinstance(openapi_spec, dict):
        return openapi_spec
    
    logger.info("Starting to clean OpenAPI specification")
    
    # Create a copy to avoid mutating the original
    cleaned_spec = {}
    
    for key, value in openapi_spec.items():
        if key == 'components' and isinstance(value, dict):
            # Clean component schemas
            logger.debug("Cleaning component schemas")
            cleaned_components = value.copy()
            if 'schemas' in cleaned_components:
                cleaned_schemas = {}
                for schema_name, schema_def in cleaned_components['schemas'].items():
                    logger.debug(f"Cleaning component schema: {schema_name}")
                    cleaned_schemas[schema_name] = clean_json_schema(schema_def)
                cleaned_components['schemas'] = cleaned_schemas
                logger.info(f"Cleaned {len(cleaned_schemas)} component schemas")
            cleaned_spec[key] = cleaned_components
        elif key == 'paths' and isinstance(value, dict):
            # Clean inline schemas in paths - this is critical for tool generation
            logger.debug("Cleaning path schemas")
            cleaned_paths = {}
            for path, path_def in value.items():
                logger.debug(f"Cleaning path: {path}")
                cleaned_paths[path] = clean_json_schema(path_def)
            cleaned_spec[key] = cleaned_paths
            logger.info(f"Cleaned schemas in {len(cleaned_paths)} paths")
        else:
            # Recursively clean other parts
            cleaned_spec[key] = clean_json_schema(value)
    
    logger.info("Completed cleaning OpenAPI specification")
    return cleaned_spec


def clean_tool_schemas(tools: Union[Dict[str, Any], list]) -> Union[Dict[str, Any], list]:
    """
    Clean tool schemas for MCP integration.
    
    This function specifically targets tool definitions that might have schema issues.
    
    Args:
        tools: Tool definitions (can be dict or list)
        
    Returns:
        Cleaned tool definitions
    """
    if isinstance(tools, dict):
        cleaned_dict = {}
        for tool_name, tool_def in tools.items():
            if isinstance(tool_def, dict) and 'parameters' in tool_def:
                # Clean the parameters schema
                cleaned_tool = tool_def.copy()
                cleaned_tool['parameters'] = clean_json_schema(tool_def['parameters'])
                cleaned_dict[tool_name] = cleaned_tool
            else:
                cleaned_dict[tool_name] = clean_json_schema(tool_def)
        return cleaned_dict
    elif isinstance(tools, list):
        cleaned_list: list[dict[str, Any]] = []
        for tool_def in tools:
            if isinstance(tool_def, dict) and 'parameters' in tool_def:
                # Clean the parameters schema
                cleaned_tool = tool_def.copy()
                cleaned_tool['parameters'] = clean_json_schema(tool_def['parameters'])
                cleaned_list.append(cleaned_tool)
            else:
                cleaned_list.append(clean_json_schema(tool_def))
        return cleaned_list
    else:
        return tools