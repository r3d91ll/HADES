#!/usr/bin/env python3
"""
Script to fix the remaining type issues in schema files.

This script addresses:
1. Missing return statements in validator functions
2. TypeVar redefinition in validation.py
3. Fixing untyped decorator issues
4. Incompatible return types in model validators
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

def fix_missing_returns(file_path: str) -> bool:
    """Fix missing return statements in validator functions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find validator functions with missing return statements
    validator_pattern = r'@.*?validator.*?\n.*?def\s+(\w+)\(.*?\).*?:.*?\n.*?if.*?raise.*?\n(\s+)(?!return|raise)'
    
    # Add return statements to validator functions
    content = re.sub(validator_pattern, r'\g<0>return v\n\2', content, flags=re.DOTALL)
    
    # Write changes
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_typevar_redefinition(file_path: str) -> bool:
    """Fix TypeVar redefinition in validation.py."""
    if not file_path.endswith('validation.py'):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Remove redefined TypeVar
    if 'T = TypeVar' in content:
        # Ensure TypeVar is already imported
        if 'from typing import' in content and 'TypeVar' in content.split('from typing import')[1].split('\n')[0]:
            content = content.replace('T = TypeVar(\'T\')\n\n', '')
        else:
            # Add TypeVar to imports if not present
            content = content.replace('from typing import', 'from typing import TypeVar, ')
            content = content.replace('T = TypeVar(\'T\')\n\n', '')
    
    # Write changes
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_untyped_decorators(file_path: str) -> bool:
    """Fix untyped decorator issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace @validator with @typed_field_validator
    content = re.sub(
        r'@validator\((["\'].*?["\'])(.*?)\)',
        r'@typed_field_validator(\1\2)',
        content
    )
    
    # Write changes
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_incompatible_return_types(file_path: str) -> bool:
    """Fix incompatible return types in model validators."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find model validator with wrong return type
    model_validator_pattern = r'@typed_model_validator\(mode=["\'](.*?)["\']\).*?\n.*?def\s+(\w+)\(.*?\).*?->.*?:'
    
    # Replace typed_model_validator with validator
    content = re.sub(
        model_validator_pattern,
        r'@validator(\'\', check_fields=False)',
        content,
        flags=re.DOTALL
    )
    
    # Write changes
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_files_with_issues(files: List[str]) -> Dict[str, bool]:
    """Fix type issues in the specified files."""
    results = {}
    
    for file_path in files:
        modified = False
        
        # Fix missing returns
        modified |= fix_missing_returns(file_path)
        
        # Fix TypeVar redefinition
        modified |= fix_typevar_redefinition(file_path)
        
        # Fix untyped decorators
        modified |= fix_untyped_decorators(file_path)
        
        # Fix incompatible return types
        modified |= fix_incompatible_return_types(file_path)
        
        results[file_path] = modified
    
    return results

def main():
    """Main entry point."""
    # Files with known issues
    files_with_issues = [
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/relations.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/dataset.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/documents/base.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/common/validation.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/queue.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/jobs.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/config.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/base.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/pipeline/text.py',
        '/home/todd/ML-Lab/Olympus/HADES/src/schemas/isne/documents.py'
    ]
    
    results = fix_files_with_issues(files_with_issues)
    
    modified_count = sum(1 for modified in results.values() if modified)
    print(f"Fixed {modified_count} files out of {len(results)}")
    
    if modified_count > 0:
        print("\nModified files:")
        for file_path, modified in results.items():
            if modified:
                print(f"  - {file_path}")

if __name__ == "__main__":
    main()
