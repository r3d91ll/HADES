#!/usr/bin/env python3
"""
Comprehensive script to fix specific type issues in the schema files.

This script addresses:
1. field_validator not defined errors
2. Untyped decorator errors
3. Unreachable statement errors
4. Incompatible return type errors
5. Missing type annotations
"""
import os
import re
import sys
from pathlib import Path

# Regular expressions for finding and fixing issues
MISSING_IMPORT_PATTERN = r'from pydantic import (.*?)(?=\n)'
FIELD_VALIDATOR_PATTERN = r'@field_validator\(["\']([^"\']+)["\']\)'
VALIDATOR_PATTERN = r'@validator\(["\']([^"\']+)["\']\)'
MODEL_VALIDATOR_PATTERN = r'@model_validator\(mode=["\']([^"\']+)["\']\)'
TYPED_MODEL_VALIDATOR_PATTERN = r'@typed_model_validator\(mode=["\']([^"\']+)["\']\)'
UNREACHABLE_RETURN_PATTERN = r'(\s+return\s+[^;]+?)\s+return\s+'
INCOMPATIBLE_RETURN_PATTERN = r'def\s+(\w+)\([^)]*\)\s*->\s*[\'"]?(\w+)[\'"]?'
MISSING_CLASSMETHOD_PATTERN = r'@[^@]*validator[^@]*\n(\s+)def\s+(\w+)\((self|cls)'

def fix_specific_files():
    """Fix specific files with known issues."""
    # Fix isne/documents.py
    documents_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/isne/documents.py'
    with open(documents_path, 'r') as f:
        content = f.read()
    
    # Fix imports
    if 'field_validator' not in content.split('from pydantic import')[1].split('\n')[0]:
        content = re.sub(
            MISSING_IMPORT_PATTERN,
            r'from pydantic import \1, field_validator',
            content,
            count=1
        )
    
    # Replace @field_validator with @validator (since it's already imported)
    content = content.replace('@field_validator(', '@validator(')
    
    # Write changes
    with open(documents_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {documents_path}")
    
    # Fix pipeline/config.py
    config_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/config.py'
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Fix imports
    if 'field_validator' not in content.split('from pydantic import')[1].split('\n')[0]:
        content = re.sub(
            MISSING_IMPORT_PATTERN,
            r'from pydantic import \1, field_validator',
            content,
            count=1
        )
    
    # Write changes
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {config_path}")
    
    # Fix pipeline/jobs.py
    jobs_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/jobs.py'
    with open(jobs_path, 'r') as f:
        content = f.read()
    
    # Fix unreachable statements by removing duplicated returns
    content = re.sub(UNREACHABLE_RETURN_PATTERN, r'\1', content)
    
    # Write changes
    with open(jobs_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {jobs_path}")
    
    # Fix validation.py no-any-return error
    validation_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/common/validation.py'
    with open(validation_path, 'r') as f:
        content = f.read()
    
    # Fix missing type annotations in the validate_or_raise function
    if 'def validate_or_raise(' in content:
        content = content.replace(
            'def validate_or_raise(',
            'def validate_or_raise[T](',
            1
        )
    
    # Write changes
    with open(validation_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {validation_path}")
    
    # Fix documents/base.py
    base_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/base.py'
    with open(base_path, 'r') as f:
        content = f.read()
    
    # Fix unreachable statements by removing duplicated returns
    content = re.sub(UNREACHABLE_RETURN_PATTERN, r'\1', content)
    
    # Write changes
    with open(base_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {base_path}")
    
    # Fix pipeline/base.py
    pipeline_base_path = '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/base.py'
    with open(pipeline_base_path, 'r') as f:
        content = f.read()
    
    # Fix unreachable statements by removing duplicated returns
    content = re.sub(UNREACHABLE_RETURN_PATTERN, r'\1', content)
    
    # Write changes
    with open(pipeline_base_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed {pipeline_base_path}")
    
    # Fix typed_model_validator errors in multiple files
    fix_typed_model_validator('/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/dataset.py')
    fix_typed_model_validator('/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/base.py')
    fix_typed_model_validator('/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/queue.py')
    fix_typed_model_validator('/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/config.py')
    fix_typed_model_validator('/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/text.py')

def fix_typed_model_validator(file_path):
    """Fix issues with typed_model_validator decorator."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace typed_model_validator with validator for class methods
    content = re.sub(
        TYPED_MODEL_VALIDATOR_PATTERN,
        r'@validator(\'\', check_fields=False)',
        content
    )
    
    # Add missing @classmethod decorators
    content = re.sub(
        MISSING_CLASSMETHOD_PATTERN,
        r'@validator\n\1@classmethod\n\1def \2(\3',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed typed_model_validator in {file_path}")

if __name__ == "__main__":
    fix_specific_files()
    print("Completed fixing schema type issues")
