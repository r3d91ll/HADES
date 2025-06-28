#!/usr/bin/env python3
"""Test the specific fixes for HADES schema issues."""

import sys
import json
sys.path.insert(0, 'src')

from api.utils.schema_cleanup import clean_json_schema

# Test specific issues found in HADES tools
specific_test_schemas = [
    # Test 1: Duplicate required fields (like configure_engine)
    {
        'type': 'object',
        'properties': {
            'engine_type': {'type': 'string'},
            'config_updates': {'type': 'object'}
        },
        'required': ['engine_type', 'engine_type', 'config_updates'],  # Duplicate!
        'title': 'duplicate_required'
    },
    # Test 2: anyOf with null pattern (like start_experiment)
    {
        'type': 'object',
        'properties': {
            'optional_field': {
                'anyOf': [
                    {'type': 'object', 'additionalProperties': True},
                    {'type': 'null'}
                ],
                'title': 'optional_field'
            }
        },
        'title': 'anyof_null_pattern'
    },
    # Test 3: Complex anyOf that should not be simplified
    {
        'type': 'object', 
        'properties': {
            'complex_field': {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'integer'},
                    {'type': 'boolean'}
                ],
                'title': 'complex_field'
            }
        },
        'title': 'complex_anyof'
    }
]

print('Testing specific HADES schema fixes...')
for i, schema in enumerate(specific_test_schemas, 1):
    print(f'Test {i}:')
    print(f'Original: {json.dumps(schema, indent=2)}')
    cleaned = clean_json_schema(schema)
    print(f'Cleaned:  {json.dumps(cleaned, indent=2)}')
    changed = schema != cleaned
    print(f'Changed:  {changed}')
    print()