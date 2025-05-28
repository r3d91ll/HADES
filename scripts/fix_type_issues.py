#!/usr/bin/env python3
"""
Utility script to fix type issues in the schema files.

This script addresses the following common issues:
1. Untyped decorator errors by properly setting up validator decorators
2. Replacing field_validator with validator where needed
3. Adding missing @classmethod decorators
4. Adding proper type annotations to validator functions
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Regular expressions to match common patterns
FIELD_VALIDATOR_PATTERN = r'@field_validator\((.*?)\)'
VALIDATOR_PATTERN = r'@validator\((.*?)\)'
MODEL_VALIDATOR_PATTERN = r'@model_validator\((.*?)\)'
TYPED_FIELD_VALIDATOR_PATTERN = r'@typed_field_validator\((.*?)\)'
TYPED_MODEL_VALIDATOR_PATTERN = r'@typed_model_validator\((.*?)\)'
MISSING_CLASSMETHOD_PATTERN = r'@(validator|field_validator|model_validator|typed_field_validator|typed_model_validator).*?\n(\s+)def\s+(\w+)\('
MISSING_TYPE_ANNOTATION_PATTERN = r'def\s+(\w+)\((cls|self),\s*v(,|\)).*?:'
REDUNDANT_CAST_PATTERN = r'return\s+cast\(.*?,\s*v\)'

def fix_file(file_path: str) -> bool:
    """
    Fix type issues in a single file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        bool: True if changes were made, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Check if we need to add validator import
    if 'from pydantic import' in content and 'validator' not in content and '@validator' in content:
        content = content.replace(
            'from pydantic import Field',
            'from pydantic import Field, validator'
        )
    
    # Fix field_validator without import
    if '@field_validator' in content and 'field_validator' not in content.split('@field_validator')[0]:
        if 'from pydantic import' in content:
            content = content.replace(
                'from pydantic import',
                'from pydantic import field_validator,'
            )
        else:
            # Add import if not present
            content = 'from pydantic import field_validator\n' + content
    
    # Fix missing @classmethod decorators
    content = re.sub(
        MISSING_CLASSMETHOD_PATTERN,
        r'@\1\2\n\2@classmethod\n\2def \3(',
        content
    )
    
    # Fix missing type annotations in validator functions
    def add_type_annotation(match):
        func_name = match.group(1)
        cls_or_self = match.group(2)
        ending = match.group(3)
        
        if ending == ')':
            # No parameters other than cls/self
            return f'def {func_name}({cls_or_self}) -> Any:'
        else:
            # Has parameters
            return f'def {func_name}({cls_or_self}, v: Any{ending}'
    
    content = re.sub(MISSING_TYPE_ANNOTATION_PATTERN, add_type_annotation, content)
    
    # Fix redundant casts
    content = re.sub(REDUNDANT_CAST_PATTERN, 'return v', content)
    
    # Write changes back if modified
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_directory(directory: str) -> Dict[str, bool]:
    """
    Fix type issues in all Python files in a directory.
    
    Args:
        directory: Directory to process
        
    Returns:
        Dict[str, bool]: Map of file paths to whether they were modified
    """
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                modified = fix_file(file_path)
                results[file_path] = modified
    
    return results

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    results = fix_directory(directory)
    
    modified_count = sum(1 for modified in results.values() if modified)
    print(f"Fixed {modified_count} files out of {len(results)}")
    
    if modified_count > 0:
        print("\nModified files:")
        for file_path, modified in results.items():
            if modified:
                print(f"  - {file_path}")

if __name__ == "__main__":
    main()
