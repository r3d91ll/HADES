#!/usr/bin/env python3
"""
Fix remaining typing issues in adapter files.

This script addresses:
1. Remove unused type: ignore comments
2. Fix redundant casts
3. Fix TypedDict total parameter
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

def remove_unused_type_ignore(file_path: str) -> None:
    """Remove unused type: ignore comments from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    for i in range(len(lines)):
        # Find and remove unused type: ignore comments
        if "# type: ignore[unreachable]" in lines[i]:
            lines[i] = lines[i].replace("# type: ignore[unreachable]", "").rstrip() + "\n"
            modified = True
            print(f"Removed unused type: ignore comment from line {i+1}")
        
        # Fix TypedDict total parameter
        if "MetadataDict, total=False" in lines[i]:
            lines[i] = lines[i].replace("MetadataDict, total=False", "MetadataDict")
            modified = True
            print(f"Fixed TypedDict total parameter in line {i+1}")
        
        # Fix redundant casts
        if "cast(str, options) if isinstance(options, str)" in lines[i]:
            lines[i] = lines[i].replace("cast(str, options)", "options")
            modified = True
            print(f"Fixed redundant cast in line {i+1}")
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Fixed typing issues in {file_path}")
    else:
        print(f"No issues found in {file_path}")

def fix_docling_adapter(file_path: str) -> None:
    """Fix typing issues in docling_adapter.py."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    for i in range(len(lines)):
        if "# type: ignore" in lines[i]:
            # Remove any remaining unused type: ignore comments
            lines[i] = lines[i].replace("# type: ignore", "").rstrip() + "\n"
            modified = True
            print(f"Removed unused type: ignore comment from line {i+1}")
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Fixed typing issues in {file_path}")
    else:
        print(f"No issues found in {file_path}")

def main() -> None:
    """Run the typing fixes."""
    python_adapter_path = Path(__file__).parent.parent / "src" / "docproc" / "adapters" / "python_adapter.py"
    docling_adapter_path = Path(__file__).parent.parent / "src" / "docproc" / "adapters" / "docling_adapter.py"
    
    print(f"Fixing typing issues in {python_adapter_path}")
    remove_unused_type_ignore(str(python_adapter_path))
    
    print(f"Fixing typing issues in {docling_adapter_path}")
    fix_docling_adapter(str(docling_adapter_path))
    
    print("Done. Run mypy to verify all issues are fixed.")

if __name__ == "__main__":
    main()
