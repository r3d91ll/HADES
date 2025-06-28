#!/usr/bin/env python3
"""Advanced test script for JSON Schema cleanup functionality."""

import sys
import json
sys.path.insert(0, 'src')

from api.utils.schema_cleanup import clean_json_schema

# Test additional JSON Schema 2020-12 violations
advanced_test_schemas = [
    # Test 1: Empty required array
    {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': [],  # Empty array should be removed
        'title': 'empty_required'
    },
    # Test 2: Invalid additionalProperties
    {
        'type': 'object',
        'additionalProperties': 'invalid',  # Should be boolean or object
        'title': 'invalid_additional_props'
    },
    # Test 3: Deprecated 'definitions' field
    {
        'type': 'object',
        'definitions': {  # Should be '$defs' in 2020-12
            'Person': {'type': 'object'}
        },
        'title': 'deprecated_definitions'
    },
    # Test 4: Invalid enum
    {
        'enum': 'not_an_array',  # Should be array
        'title': 'invalid_enum'
    },
    # Test 5: Complex nested violations
    {
        'type': 'object',
        'properties': {
            'nested': {
                'anyOf': [{'type': 'string'}],
                'type': 'string',  # Conflict in nested schema
                'required': 'not_array'  # Invalid required
            }
        },
        'title': 'complex_nested'
    }
]

print('Testing advanced JSON Schema cleanup...')
for i, schema in enumerate(advanced_test_schemas, 1):
    print(f'Test {i}:')
    print(f'Original: {json.dumps(schema, indent=2)}')
    cleaned = clean_json_schema(schema)
    print(f'Cleaned:  {json.dumps(cleaned, indent=2)}')
    changed = schema != cleaned
    print(f'Changed:  {changed}')
    print()