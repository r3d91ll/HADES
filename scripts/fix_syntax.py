#!/usr/bin/env python3
"""
Script to fix syntax issues in the ArangoClient implementation.

This script addresses common syntax errors like:
- Unmatched parentheses
- Missing try-except blocks
- Missing return statements
"""

import re
import sys

def fix_syntax_issues(file_path):
    """Fix syntax issues in the ArangoClient implementation."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Fix unmatched parentheses in return statements
    content = re.sub(r'return\s+\w+\(.*?\)\)', r'return \1)', content)
    
    # 2. Fix extra parentheses in method calls
    content = re.sub(r'self\._retry_operation\(\)(\s+\w+)', r'self._retry_operation(\1', content)
    
    # 3. Make sure all try blocks have an except clause
    content = re.sub(
        r'try:\s+([^}]+)(?!except|finally)',
        r'try:\n        \1\n        except Exception as e:\n            self.logger.error(f"Operation failed: {str(e)}")\n            raise',
        content
    )
    
    # 4. Fix multi-level bool nesting
    content = re.sub(r'bool\(bool\(bool\((.*?)\)\)\)', r'bool(\1)', content)
    content = re.sub(r'bool\(bool\((.*?)\)\)', r'bool(\1)', content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed syntax issues in {file_path}")

if __name__ == "__main__":
    fix_syntax_issues('/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py')
