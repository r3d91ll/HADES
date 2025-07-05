#!/usr/bin/env python3
"""
Verify that all directories have .hades metadata.

This script checks that our metadata system covers the entire directory tree,
which signals to HADES that the directory structure is fully mapped.
"""

import os
from pathlib import Path
from typing import List, Tuple


def check_metadata_coverage(root_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Check which directories have .hades metadata.
    
    Returns:
        Tuple of (directories_with_metadata, directories_without_metadata)
    """
    with_metadata = []
    without_metadata = []
    
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        current_path = Path(root)
        
        # Skip .hades directories themselves
        if current_path.name == '.hades':
            continue
            
        # Check for .hades directory
        hades_dir = current_path / '.hades'
        if hades_dir.exists() and hades_dir.is_dir():
            with_metadata.append(current_path)
        else:
            without_metadata.append(current_path)
            
    return with_metadata, without_metadata


def main() -> None:
    """Check metadata coverage for HADES."""
    project_root = Path(__file__).parent.parent.parent
    
    with_metadata, without_metadata = check_metadata_coverage(project_root)
    
    print(f"HADES Metadata Coverage Report")
    print(f"=" * 50)
    print(f"Total directories: {len(with_metadata) + len(without_metadata)}")
    print(f"With .hades metadata: {len(with_metadata)}")
    print(f"Without .hades metadata: {len(without_metadata)}")
    print()
    
    if without_metadata:
        print("Directories missing .hades metadata:")
        for path in sorted(without_metadata):
            relative_path = path.relative_to(project_root)
            print(f"  - {relative_path}")
    else:
        print("✅ All directories have .hades metadata!")
        print("The entire directory structure is mapped in the RAG system.")
    
    print()
    print("Directories with .hades metadata:")
    for path in sorted(with_metadata)[:10]:  # Show first 10
        relative_path = path.relative_to(project_root)
        print(f"  ✓ {relative_path}")
    
    if len(with_metadata) > 10:
        print(f"  ... and {len(with_metadata) - 10} more")


if __name__ == "__main__":
    main()