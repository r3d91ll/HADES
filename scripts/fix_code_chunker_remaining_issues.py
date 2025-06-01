#!/usr/bin/env python3
"""
Script to fix remaining typing issues in the Python code chunker.

This script addresses the remaining typing issues in the Python code chunker:
1. Fixing incompatible types between ChunkMetadata and CodeChunkMetadata
2. Removing redundant casts that are causing mypy warnings
"""

import re
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# The file to fix
PYTHON_CHUNKER_PATH = Path(__file__).parent.parent / "src" / "chunking" / "code_chunkers" / "python_chunker.py"


def fix_incompatible_metadata_types(content: str) -> str:
    """Fix incompatible types between ChunkMetadata and CodeChunkMetadata."""
    
    # Find all instances of cast(ChunkMetadata, metadata) and replace with cast(CodeChunkMetadata, metadata)
    pattern = r'cast\(ChunkMetadata, (chunk_metadata|metadata)\)'
    replacement = r'cast(CodeChunkMetadata, \1)'
    content = re.sub(pattern, replacement, content)
    
    return content


def remove_redundant_casts(content: str) -> str:
    """Remove redundant casts that are causing mypy warnings."""
    
    # Find and remove redundant cast lines
    pattern = r'(\s+)# Cast to match required base class return type\n\s+return cast\(List\[Dict\[str, Any\]\], chunks\)'
    replacement = r'\1return chunks'
    content = re.sub(pattern, replacement, content)
    
    return content


def fix_python_chunker():
    """Fix remaining typing issues in the Python code chunker."""
    
    # Check if the file exists
    if not PYTHON_CHUNKER_PATH.exists():
        print(f"Error: Could not find {PYTHON_CHUNKER_PATH}")
        return False
    
    # Read the file content
    with open(PYTHON_CHUNKER_PATH, 'r') as file:
        content = file.read()
    
    # Apply fixes
    content = fix_incompatible_metadata_types(content)
    content = remove_redundant_casts(content)
    
    # Write the fixed content back to the file
    with open(PYTHON_CHUNKER_PATH, 'w') as file:
        file.write(content)
    
    print(f"Fixed remaining typing issues in {PYTHON_CHUNKER_PATH}")
    return True


if __name__ == "__main__":
    fix_python_chunker()
