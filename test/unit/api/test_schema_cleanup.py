"""
Tests for schema cleanup utilities.

This module tests the schema cleanup functionality used to fix
FastAPI-MCP JSON schema issues.
"""

import pytest
from src.api.utils.schema_cleanup import clean_json_schema, clean_openapi_spec, clean_tool_schemas


class TestSchemaCleanup:
    """Test cases for schema cleanup functions."""
    
    def test_clean_json_schema_removes_conflicting_type(self):
        """Test that conflicting type field is removed when anyOf is present."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "type": "string",
            "title": "database_name",
            "description": "Database name"
        }
        
        cleaned = clean_json_schema(schema)
        
        assert "anyOf" in cleaned
        assert "type" not in cleaned
        assert cleaned["title"] == "database_name"
        assert cleaned["description"] == "Database name"
    
    def test_clean_json_schema_preserves_type_without_anyof(self):
        """Test that type field is preserved when anyOf is not present."""
        schema = {
            "type": "string",
            "title": "simple_field",
            "description": "A simple string field"
        }
        
        cleaned = clean_json_schema(schema)
        
        assert cleaned["type"] == "string"
        assert cleaned["title"] == "simple_field"
        assert cleaned["description"] == "A simple string field"
    
    def test_clean_json_schema_preserves_anyof_without_type(self):
        """Test that anyOf field is preserved when type is not present."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "title": "optional_field",
            "description": "An optional field"
        }
        
        cleaned = clean_json_schema(schema)
        
        assert cleaned["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert cleaned["title"] == "optional_field"
        assert cleaned["description"] == "An optional field"
    
    def test_clean_json_schema_handles_nested_schemas(self):
        """Test that nested schemas are cleaned recursively."""
        schema = {
            "type": "object",
            "properties": {
                "field1": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "type": "string",
                    "title": "field1"
                },
                "field2": {
                    "type": "integer",
                    "title": "field2"
                }
            }
        }
        
        cleaned = clean_json_schema(schema)
        
        assert cleaned["type"] == "object"
        assert "anyOf" in cleaned["properties"]["field1"]
        assert "type" not in cleaned["properties"]["field1"]
        assert cleaned["properties"]["field2"]["type"] == "integer"
    
    def test_clean_json_schema_handles_arrays(self):
        """Test that arrays of schemas are cleaned recursively."""
        schema = {
            "items": [
                {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "type": "string"
                },
                {
                    "type": "integer"
                }
            ]
        }
        
        cleaned = clean_json_schema(schema)
        
        assert "anyOf" in cleaned["items"][0]
        assert "type" not in cleaned["items"][0]
        assert cleaned["items"][1]["type"] == "integer"
    
    def test_clean_json_schema_handles_non_dict(self):
        """Test that non-dictionary values are returned unchanged."""
        values = [
            "string",
            123,
            True,
            None,
            ["list", "items"]
        ]
        
        for value in values:
            assert clean_json_schema(value) == value
    
    def test_clean_openapi_spec(self):
        """Test cleaning of complete OpenAPI specification."""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "components": {
                "schemas": {
                    "TestModel": {
                        "type": "object",
                        "properties": {
                            "optional_field": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "type": "string",
                                "title": "optional_field"
                            }
                        }
                    }
                }
            },
            "paths": {
                "/test": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "anyOf": [{"type": "object"}, {"type": "null"}],
                                        "type": "object"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        cleaned = clean_openapi_spec(openapi_spec)
        
        # Check that component schema was cleaned
        optional_field = cleaned["components"]["schemas"]["TestModel"]["properties"]["optional_field"]
        assert "anyOf" in optional_field
        assert "type" not in optional_field
        
        # Check that inline schema was cleaned
        inline_schema = cleaned["paths"]["/test"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert "anyOf" in inline_schema
        assert "type" not in inline_schema
    
    def test_clean_tool_schemas_dict(self):
        """Test cleaning of tool schemas in dictionary format."""
        tools = {
            "tool1": {
                "parameters": {
                    "anyOf": [{"type": "object"}, {"type": "null"}],
                    "type": "object",
                    "properties": {}
                }
            },
            "tool2": {
                "description": "A simple tool"
            }
        }
        
        cleaned = clean_tool_schemas(tools)
        
        assert "anyOf" in cleaned["tool1"]["parameters"]
        assert "type" not in cleaned["tool1"]["parameters"]
        assert cleaned["tool2"]["description"] == "A simple tool"
    
    def test_clean_tool_schemas_list(self):
        """Test cleaning of tool schemas in list format."""
        tools = [
            {
                "name": "tool1",
                "parameters": {
                    "anyOf": [{"type": "object"}, {"type": "null"}],
                    "type": "object"
                }
            },
            {
                "name": "tool2",
                "description": "A simple tool"
            }
        ]
        
        cleaned = clean_tool_schemas(tools)
        
        assert "anyOf" in cleaned[0]["parameters"]
        assert "type" not in cleaned[0]["parameters"]
        assert cleaned[1]["description"] == "A simple tool"