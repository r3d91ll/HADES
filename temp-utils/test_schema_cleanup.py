#!/usr/bin/env python3
"""Test script for JSON Schema cleanup functionality."""

import sys
import json
sys.path.insert(0, 'src')

from api.utils.schema_cleanup import clean_json_schema

# Test various JSON Schema 2020-12 violations
test_schemas = [
    # Test 1: anyOf + type conflict
    {
        'anyOf': [{'type': 'string'}, {'type': 'null'}],
        'type': 'string',
        'title': 'test_field'
    },
    # Test 2: oneOf + type conflict  
    {
        'oneOf': [{'type': 'integer'}, {'type': 'string'}],
        'type': 'integer',
        'title': 'test_field2'
    },
    # Test 3: Invalid type value
    {
        'type': 123,  # Should be string
        'title': 'test_field3'
    },
    # Test 4: Valid schema (should remain unchanged)
    {
        'type': 'string',
        'title': 'valid_field'
    }
]

print('Testing JSON Schema cleanup...')
for i, schema in enumerate(test_schemas, 1):
    print(f'Test {i}:')
    print(f'Original: {json.dumps(schema, indent=2)}')
    cleaned = clean_json_schema(schema)
    print(f'Cleaned:  {json.dumps(cleaned, indent=2)}')
    changed = schema != cleaned
    print(f'Changed:  {changed}')
    print()