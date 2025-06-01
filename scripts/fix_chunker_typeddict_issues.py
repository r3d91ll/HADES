#!/usr/bin/env python3
"""
Script to fix TypedDict compatibility issues in the Python code chunker.

This script addresses the specific TypedDict-related typing issues in the Python code chunker:
1. Fixing incompatible types in specific locations (lines 289, 393, 423)
2. Fixing the return type of _extract_fallback_chunks method (line 426)
"""

import re
import os
import sys
from pathlib import Path
import ast
import astor

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# The file to fix
PYTHON_CHUNKER_PATH = Path(__file__).parent.parent / "src" / "chunking" / "code_chunkers" / "python_chunker.py"


def fix_extract_fallback_chunks(source):
    """Fix the return type and return value in _extract_fallback_chunks method."""
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_extract_fallback_chunks':
            # Fix the return type annotation
            for i, ann in enumerate(node.returns.slice.elts):
                if isinstance(ann, ast.Name) and ann.id == 'CodeChunk':
                    node.returns.slice.elts[i] = ast.Name(id='Dict[str, Any]', ctx=ast.Load())
            
            # Find the return statement
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Name) and child.value.id == 'chunks':
                    # Replace with return with a cast
                    child.value = ast.Call(
                        func=ast.Name(id='cast', ctx=ast.Load()),
                        args=[
                            ast.Subscript(
                                value=ast.Name(id='List', ctx=ast.Load()),
                                slice=ast.Subscript(
                                    value=ast.Name(id='Dict', ctx=ast.Load()),
                                    slice=ast.Tuple(
                                        elts=[
                                            ast.Constant(value='str'),
                                            ast.Name(id='Any', ctx=ast.Load())
                                        ],
                                        ctx=ast.Load()
                                    ),
                                    ctx=ast.Load()
                                ),
                                ctx=ast.Load()
                            ),
                            ast.Name(id='chunks', ctx=ast.Load())
                        ],
                        keywords=[]
                    )
    
    return astor.to_source(tree)


def fix_typeddict_issues(source):
    """Fix TypedDict compatibility issues by using Dict[str, Any] instead of specific TypedDict types."""
    
    # Fix metadata TypedDict issues in three specific locations
    patterns = [
        # Line 289 area - fix by using Dict[str, Any] instead of ChunkMetadata
        (r'(\s+)metadata = create_code_chunk_metadata\((?:[\s\S]*?)\)(\s+)return \{(?:[\s\S]*?)"metadata": cast\(Dict\[str, Any\], metadata\)(?:[\s\S]*?)\}',
         r'\1metadata = create_code_chunk_metadata(\2\1# Use Dict[str, Any] to avoid TypedDict incompatibility\2return {\2    "id": chunk_id,\2    "content": content,\2    "metadata": cast(Dict[str, Any], metadata)\2}'),
        
        # Line 393 area - fix by explicitly creating a Dict[str, Any]
        (r'(\s+)metadata = create_code_chunk_metadata\((?:[\s\S]*?)\)(\s+)chunk = \{(?:[\s\S]*?)"metadata": cast\(Dict\[str, Any\], metadata\)(?:[\s\S]*?)\}',
         r'\1metadata = create_code_chunk_metadata(\2\1# Use Dict[str, Any] to avoid TypedDict incompatibility\2chunk = {\2    "id": chunk_id,\2    "content": content,\2    "metadata": cast(Dict[str, Any], metadata)\2}'),
        
        # Line 423 area - fix by explicitly creating a Dict[str, Any]
        (r'(\s+)chunk_metadata = create_code_chunk_metadata\((?:[\s\S]*?)\)(\s+)chunk = \{(?:[\s\S]*?)"metadata": cast\(Dict\[str, Any\], chunk_metadata\)(?:[\s\S]*?)\}',
         r'\1chunk_metadata = create_code_chunk_metadata(\2\1# Use Dict[str, Any] to avoid TypedDict incompatibility\2chunk = {\2    "id": chunk_id,\2    "content": content,\2    "metadata": cast(Dict[str, Any], chunk_metadata)\2}')
    ]
    
    for pattern, replacement in patterns:
        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
    
    return source


def fix_python_chunker():
    """Fix TypedDict compatibility issues in the Python code chunker."""
    
    # Check if the file exists
    if not PYTHON_CHUNKER_PATH.exists():
        print(f"Error: Could not find {PYTHON_CHUNKER_PATH}")
        return False
    
    try:
        # Read the file content
        with open(PYTHON_CHUNKER_PATH, 'r') as file:
            content = file.read()
        
        # Apply fixes
        content = fix_typeddict_issues(content)
        
        # Try to apply AST-based fixes for function return type
        try:
            import astor
            content = fix_extract_fallback_chunks(content)
        except (ImportError, SyntaxError) as e:
            print(f"Warning: Could not apply AST-based fixes: {e}")
            # Fallback to regex-based approach for the _extract_fallback_chunks method
            pattern = r'(def _extract_fallback_chunks.*?-> )(?:List\[CodeChunk\]|CodeChunkList)(:.+)'
            replacement = r'\1List[Dict[str, Any]]\2'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            # Add explicit cast at return
            pattern = r'(\s+return chunks)(?!\s*\))'
            replacement = r'\1  # Explicitly cast to match required return type\n\1 = cast(List[Dict[str, Any]], chunks)\n\1'
            content = re.sub(pattern, replacement, content)
        
        # Write the fixed content back to the file
        with open(PYTHON_CHUNKER_PATH, 'w') as file:
            file.write(content)
        
        print(f"Fixed TypedDict compatibility issues in {PYTHON_CHUNKER_PATH}")
        return True
    
    except Exception as e:
        print(f"Error fixing TypedDict issues: {e}")
        return False


if __name__ == "__main__":
    # Check if astor is installed for AST manipulation
    try:
        import astor
    except ImportError:
        print("Warning: astor not installed. Using regex-based approach only.")
    
    fix_python_chunker()
