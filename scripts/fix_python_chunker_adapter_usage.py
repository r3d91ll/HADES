#!/usr/bin/env python3
"""
Script to fix adapter usage issues in the Python code chunker.

This script addresses the specific adapter usage issues in the Python code chunker:
1. Fixing incorrect call to process_text
2. Fixing type mismatch between ProcessedDocument and Dict[str, Any]
"""

import re
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# The file to fix
PYTHON_CHUNKER_PATH = Path(__file__).parent.parent / "src" / "chunking" / "code_chunkers" / "python_chunker.py"


def fix_adapter_usage(content: str) -> str:
    """Fix adapter usage issues."""
    
    # Fix incorrect adapter usage pattern
    pattern = r'python_adapter = get_adapter_class\(\'python\'\)(.*?)processed = python_adapter\.process_text\(.*?options={.*?}\)'
    replacement = r'python_adapter_cls = get_adapter_class(\'python\')\1if not python_adapter_cls:\1    logger.warning("Python adapter not found. Using fallback chunking.")\1    return self._fallback_chunking(text, metadata)\1\1python_adapter = python_adapter_cls()\1processed = python_adapter.process_text(text, options=None)'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Fix type mismatch between ProcessedDocument and Dict[str, Any]
    pattern = r'(chunks = self\._extract_code_chunks\(processed, metadata\))'
    replacement = r'# Convert ProcessedDocument to Dict[str, Any] for internal processing\nprocessed_dict = {}\nif hasattr(processed, "model_dump"):\n    processed_dict = processed.model_dump()\nelif hasattr(processed, "dict"):\n    processed_dict = processed.dict()\nelse:\n    # Fallback for unknown response type\n    try:\n        processed_dict = dict(processed)\n    except (TypeError, ValueError):\n        logger.warning("Could not convert ProcessedDocument to dictionary. Using fallback chunking.")\n        return self._fallback_chunking(text, metadata)\n\n\1'
    content = re.sub(pattern, replacement, content)
    
    # Fix _extract_code_chunks call to use processed_dict instead of processed
    pattern = r'chunks = self\._extract_code_chunks\(processed, metadata\)'
    replacement = r'chunks = self._extract_code_chunks(processed_dict, metadata)'
    content = re.sub(pattern, replacement, content)
    
    return content


def fix_python_chunker():
    """Fix adapter usage issues in the Python code chunker."""
    
    # Check if the file exists
    if not PYTHON_CHUNKER_PATH.exists():
        print(f"Error: Could not find {PYTHON_CHUNKER_PATH}")
        return False
    
    # Read the file content
    with open(PYTHON_CHUNKER_PATH, 'r') as file:
        content = file.read()
    
    # Apply fixes
    content = fix_adapter_usage(content)
    
    # Write the fixed content back to the file
    with open(PYTHON_CHUNKER_PATH, 'w') as file:
        file.write(content)
    
    print(f"Fixed adapter usage issues in {PYTHON_CHUNKER_PATH}")
    return True


if __name__ == "__main__":
    fix_python_chunker()
