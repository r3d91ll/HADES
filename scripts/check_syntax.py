#!/usr/bin/env python3
"""
Script to check syntax in Python files using the built-in ast module.
This will identify the exact line where a syntax error occurs.
"""

import os
import sys
import ast

def check_syntax(file_path):
    """Check syntax in a Python file using ast.parse."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        print(f"✅ No syntax errors found in {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path} at line {e.lineno}, column {e.offset}:")
        print(f"   {e.text.strip()}")
        print(f"   {' ' * (e.offset - 1)}^")
        print(f"   Error message: {str(e)}")
        return False

if __name__ == "__main__":
    file_path = '/home/todd/ML-Lab/Olympus/HADES/src/database/arango_client.py'
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_syntax(file_path)
