"""
Custom FastAPI-MCP wrapper with schema cleanup.

This module provides a wrapper around FastApiMCP that automatically
cleans up problematic JSON schemas before they are converted to MCP tools.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

try:
    from fastapi_mcp import FastApiMCP
    from fastapi_mcp.openapi.convert import convert_openapi_to_mcp_tools
    from fastapi.openapi.utils import get_openapi
    FASTAPI_MCP_AVAILABLE = True
except ImportError:
    """
    # FastApiMCP = None  # type: ignore
    # convert_openapi_to_mcp_tools = None  # type: ignore
    """
    get_openapi = None  # type: ignore
    FASTAPI_MCP_AVAILABLE = False

from .schema_cleanup import clean_openapi_spec, clean_json_schema

logger = logging.getLogger(__name__)


if FASTAPI_MCP_AVAILABLE:
    class CleanedFastApiMCP(FastApiMCP):
        """
        A FastAPI-MCP wrapper that automatically cleans problematic schemas.
        
        This class extends FastApiMCP to apply schema cleanup before converting
        OpenAPI schemas to MCP tools, fixing issues with conflicting 'anyOf' and 'type' fields.
        """
        
        def setup_server(self) -> None:
            """Set up the MCP server with cleaned schemas."""
            logger.info("Setting up FastAPI-MCP server with schema cleanup")
            
            # Generate and clean the OpenAPI schema
            openapi_schema = get_openapi(
                title=self.fastapi.title,
                version=self.fastapi.version,
                openapi_version=self.fastapi.openapi_version,
                description=self.fastapi.description,
                routes=self.fastapi.routes,
            )
            
            logger.debug("Cleaning OpenAPI schema to fix JSON Schema violations")
            cleaned_openapi_schema = clean_openapi_spec(openapi_schema)
            
            # Log the number of schemas cleaned (for debugging)
            original_schemas = self._count_schemas(openapi_schema)
            cleaned_schemas = self._count_schemas(cleaned_openapi_schema)
            if original_schemas != cleaned_schemas:
                logger.info(f"Schema cleanup applied to {original_schemas - cleaned_schemas} problematic schemas")
            
            # Temporarily replace the FastAPI app's openapi method to return cleaned schema
            original_openapi = self.fastapi.openapi
            self.fastapi.openapi = lambda: cleaned_openapi_schema
            
            try:
                # Call parent setup with cleaned schema
                super().setup_server()
                logger.info(f"Generated {len(self.tools)} MCP tools from cleaned OpenAPI schema")
                
                # Additional cleanup: fix any remaining schema issues in the generated tools
                self._clean_generated_tools()
                
            finally:
                # Restore original openapi method
                self.fastapi.openapi = original_openapi
        
        def _count_schemas(self, openapi_schema: Dict[str, Any]) -> int:
            """
            Count the number of schemas in an OpenAPI specification.
            
            Used for logging purposes to track how many schemas were processed.
            
            Args:
                openapi_schema: The OpenAPI schema dictionary
                
            Returns:
                Number of schema definitions found
            """
            count = 0
            
            # Count schemas in components
            if 'components' in openapi_schema and 'schemas' in openapi_schema['components']:
                count += len(openapi_schema['components']['schemas'])
            
            # Count inline schemas in paths (simplified counting)
            if 'paths' in openapi_schema:
                for path_def in openapi_schema['paths'].values():
                    if isinstance(path_def, dict):
                        count += self._count_inline_schemas(path_def)
            
            return count
        
        def _count_inline_schemas(self, obj: Any) -> int:
            """
            Recursively count inline schema definitions.
            
            Args:
                obj: Object to search for schema definitions
                
            Returns:
                Number of inline schema definitions found
            """
            count = 0
            
            if isinstance(obj, dict):
                # Check if this looks like a schema definition
                if 'type' in obj or 'anyOf' in obj or 'properties' in obj:
                    count += 1
                
                # Recursively search nested objects
                for value in obj.values():
                    count += self._count_inline_schemas(value)
            elif isinstance(obj, list):
                for item in obj:
                    count += self._count_inline_schemas(item)
            
            return count
        
        def _clean_generated_tools(self) -> None:
            """
            Clean the generated MCP tools to fix any remaining schema issues.
            
            This method applies schema cleanup to the inputSchema of each generated tool
            to ensure they are valid JSON Schema 2020-12 compliant.
            """
            if not hasattr(self, 'tools') or not self.tools:
                logger.debug("No tools to clean")
                return
            
            logger.info(f"Applying additional schema cleanup to {len(self.tools)} generated MCP tools")
            
            tools_cleaned = 0
            problematic_tools = []
            
            for i, tool in enumerate(self.tools):
                try:
                    tool_name = getattr(tool, 'name', f'tool_{i}')
                    
                    # Log tool details for debugging
                    logger.debug(f"Processing tool {i}: {tool_name}")
                    
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        # Clean the input schema
                        original_schema = tool.inputSchema
                        cleaned_schema = clean_json_schema(original_schema)
                        
                        # Check if any changes were made
                        if cleaned_schema != original_schema:
                            tool.inputSchema = cleaned_schema
                            tools_cleaned += 1
                            logger.info(f"🔧 Cleaned schema for tool '{tool_name}' (index {i})")
                            logger.debug(f"Original schema: {json.dumps(original_schema, indent=2)}")
                            logger.debug(f"Cleaned schema: {json.dumps(cleaned_schema, indent=2)}")
                        
                        # Validate the cleaned schema for JSON Schema 2020-12 compliance
                        validation_issues = self._validate_json_schema_2020_12(cleaned_schema)
                        if validation_issues:
                            problematic_tools.append({
                                'index': i,
                                'name': tool_name,
                                'issues': validation_issues,
                                'schema': cleaned_schema
                            })
                            logger.warning(f"⚠️  Tool {i} '{tool_name}' still has schema issues: {validation_issues}")
                            
                    else:
                        logger.debug(f"Tool {i} '{tool_name}' has no inputSchema")
                        
                except Exception as e:
                    logger.error(f"Error processing tool {i}: {e}")
                    problematic_tools.append({
                        'index': i,
                        'name': f'tool_{i}',
                        'issues': [f'Processing error: {e}'],
                        'schema': None
                    })
            
            # Report results
            if tools_cleaned > 0:
                logger.info(f"🔧 Applied schema fixes to {tools_cleaned} tools")
            else:
                logger.info("✓ All generated tools already have clean schemas")
                
            if problematic_tools:
                logger.error(f"❌ Found {len(problematic_tools)} tools with remaining schema issues:")
                for tool_info in problematic_tools:
                    logger.error(f"  - Tool {tool_info['index']} '{tool_info['name']}': {tool_info['issues']}")
                    if tool_info['schema']:
                        logger.error(f"    Schema: {json.dumps(tool_info['schema'], indent=4)}")
            else:
                logger.info("✅ All tools passed JSON Schema 2020-12 validation")
        
        def _validate_json_schema_2020_12(self, schema: Dict[str, Any]) -> List[str]:
            """
            Validate a JSON schema against JSON Schema 2020-12 requirements.
            
            Args:
                schema: The JSON schema to validate
                
            Returns:
                List of validation issues found (empty if valid)
            """
            issues = []
            
            if not isinstance(schema, dict):
                return issues
            
            # Check for composition + type conflicts
            composition_keys = ['anyOf', 'oneOf', 'allOf']
            has_composition = any(key in schema for key in composition_keys)
            
            if has_composition and 'type' in schema:
                composition_type = next((key for key in composition_keys if key in schema), None)
                issues.append(f"Has both '{composition_type}' and 'type' fields")
            
            # Check type field validity
            if 'type' in schema:
                type_value = schema['type']
                if not isinstance(type_value, (str, list)):
                    issues.append(f"Invalid 'type' field: {type(type_value)} (must be string or array)")
                elif isinstance(type_value, list):
                    if not all(isinstance(t, str) for t in type_value):
                        issues.append(f"Invalid 'type' array: contains non-strings")
                    if len(type_value) == 0:
                        issues.append("Empty 'type' array")
            
            # Check for invalid additional properties
            if 'additionalProperties' in schema:
                add_props = schema['additionalProperties']
                if not isinstance(add_props, (bool, dict)):
                    issues.append(f"Invalid 'additionalProperties': {type(add_props)} (must be boolean or object)")
            
            # Check properties structure
            if 'properties' in schema:
                if not isinstance(schema['properties'], dict):
                    issues.append(f"Invalid 'properties': {type(schema['properties'])} (must be object)")
                else:
                    # Recursively validate property schemas
                    for prop_name, prop_schema in schema['properties'].items():
                        if isinstance(prop_schema, dict):
                            prop_issues = self._validate_json_schema_2020_12(prop_schema)
                            for issue in prop_issues:
                                issues.append(f"Property '{prop_name}': {issue}")
            
            # Check required field
            if 'required' in schema:
                if not isinstance(schema['required'], list):
                    issues.append(f"Invalid 'required': {type(schema['required'])} (must be array)")
                elif not all(isinstance(item, str) for item in schema['required']):
                    issues.append("Invalid 'required' array: contains non-strings")
            
            return issues
else:
    # Create a dummy class when FastAPI-MCP is not available
    class CleanedFastApiMCP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("FastAPI-MCP is not available")