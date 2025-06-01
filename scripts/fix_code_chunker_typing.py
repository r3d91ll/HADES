#!/usr/bin/env python3
"""
Script to fix typing issues in the Python code chunker.

This script addresses several typing issues in the code chunkers, focusing on:
1. Ensuring proper type conversions with ChunkMetadata
2. Fixing return type compatibility between subclasses and their base classes
3. Adding proper casting for dictionary fields
4. Ensuring consistent typing throughout the chunking system
"""

import re
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# The file to fix
PYTHON_CHUNKER_PATH = Path(__file__).parent.parent / "src" / "chunking" / "code_chunkers" / "python_chunker.py"


def fix_metadata_typing(content: str) -> str:
    """Fix metadata typing issues by ensuring proper casting to ChunkMetadata."""
    
    # Fix metadata casting in return statements
    pattern = r'(\s+)"metadata": (?:cast\(Dict\[str, Any\], )?(metadata)'
    replacement = r'\1"metadata": cast(ChunkMetadata, \2)'
    content = re.sub(pattern, replacement, content)
    
    # Fix chunk_metadata in return statements
    pattern = r'(\s+)"metadata": (chunk_metadata)'
    replacement = r'\1"metadata": cast(ChunkMetadata, \2)'
    content = re.sub(pattern, replacement, content)
    
    return content


def fix_return_types(content: str) -> str:
    """Fix return type compatibility issues."""
    
    # Fix the chunk method return type
    pattern = r'def chunk\(self, content: ChunkableDocument, \*\*kwargs: Any\) -> (?:CodeChunkList|List\[CodeChunk\]|List\[Dict\[str, Any\]\]):'
    replacement = r'def chunk(self, content: ChunkableDocument, **kwargs: Any) -> List[Dict[str, Any]]:'
    content = re.sub(pattern, replacement, content)
    
    # Add cast to chunks return statements
    pattern = r'(\s+)return chunks(?!\s*\))'
    replacement = r'\1# Cast to match required base class return type\n\1return cast(List[Dict[str, Any]], chunks)'
    content = re.sub(pattern, replacement, content)
    
    # Fix the _extract_code_chunks method parameter and return type
    pattern = r'def _extract_code_chunks\(self, processed: (?:Dict\[str, Any\]|.*?), metadata: (?:Dict\[str, Any\]|.*?)\) -> .*?:'
    replacement = r'def _extract_code_chunks(self, processed: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:'
    content = re.sub(pattern, replacement, content)
    
    # Fix the _fallback_chunking method parameter and return type
    pattern = r'def _fallback_chunking\(self, text: str, metadata: (?:Dict\[str, Any\]|ChunkMetadata)\) -> .*?:'
    replacement = r'def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:'
    content = re.sub(pattern, replacement, content)
    
    return content


def fix_python_chunker():
    """Fix typing issues in the Python code chunker."""
    
    # Check if the file exists
    if not PYTHON_CHUNKER_PATH.exists():
        print(f"Error: Could not find {PYTHON_CHUNKER_PATH}")
        return False
    
    # Read the file content
    with open(PYTHON_CHUNKER_PATH, 'r') as file:
        content = file.read()
    
    # Apply fixes
    content = fix_metadata_typing(content)
    content = fix_return_types(content)
    
    # Write the fixed content back to the file
    with open(PYTHON_CHUNKER_PATH, 'w') as file:
        file.write(content)
    
    print(f"Fixed typing issues in {PYTHON_CHUNKER_PATH}")
    return True


if __name__ == "__main__":
    fix_python_chunker()
