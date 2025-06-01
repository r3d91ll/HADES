#!/usr/bin/env python3
"""
Script to fix import and function reference issues in the Python code chunker.

This script addresses the specific import and reference issues in the Python code chunker:
1. Fixing incorrect attribute references (get_adapter → get_adapter_class)
2. Fixing incorrect function references (create_code_chunk_metadata → create_chunk_metadata)
3. Fixing incorrect type references (CodeChunkMetadata → ChunkMetadata)
4. Fixing argument type mismatch in BaseChunker.__init__
"""

import re
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# The file to fix
PYTHON_CHUNKER_PATH = Path(__file__).parent.parent / "src" / "chunking" / "code_chunkers" / "python_chunker.py"


def fix_imports_and_references(content: str) -> str:
    """Fix import and function reference issues."""
    
    # Fix incorrect attribute references
    content = content.replace("get_adapter", "get_adapter_class")
    
    # Fix incorrect function references 
    content = content.replace("create_code_chunk_metadata", "create_chunk_metadata")
    
    # Fix incorrect type references
    content = content.replace("CodeChunkMetadata", "ChunkMetadata")
    
    # Fix BaseChunker.__init__ argument
    pattern = r'super\(\)\.__init__\(config\)'
    replacement = r'super().__init__(name="python_code", config=config)'
    content = re.sub(pattern, replacement, content)
    
    return content


def fix_python_chunker():
    """Fix import and function reference issues in the Python code chunker."""
    
    # Check if the file exists
    if not PYTHON_CHUNKER_PATH.exists():
        print(f"Error: Could not find {PYTHON_CHUNKER_PATH}")
        return False
    
    # Read the file content
    with open(PYTHON_CHUNKER_PATH, 'r') as file:
        content = file.read()
    
    # Apply fixes
    content = fix_imports_and_references(content)
    
    # Write the fixed content back to the file
    with open(PYTHON_CHUNKER_PATH, 'w') as file:
        file.write(content)
    
    print(f"Fixed import and function reference issues in {PYTHON_CHUNKER_PATH}")
    return True


if __name__ == "__main__":
    fix_python_chunker()
