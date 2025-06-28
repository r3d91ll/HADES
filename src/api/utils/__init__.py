"""
Utility modules for the HADES API.

This package contains utility functions and classes for API operations.
"""

from .schema_cleanup import clean_json_schema, clean_openapi_spec, clean_tool_schemas

try:
    from .fastapi_mcp_wrapper import CleanedFastApiMCP, FASTAPI_MCP_AVAILABLE
    CLEANED_FASTAPI_MCP_AVAILABLE = FASTAPI_MCP_AVAILABLE
except ImportError:
    CleanedFastApiMCP = None  # type: ignore[assignment]
    CLEANED_FASTAPI_MCP_AVAILABLE = False

__all__ = [
    "clean_json_schema",
    "clean_openapi_spec", 
    "clean_tool_schemas",
    "CleanedFastApiMCP",
    "CLEANED_FASTAPI_MCP_AVAILABLE",
]