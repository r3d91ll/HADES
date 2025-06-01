#!/usr/bin/env python3
"""
Utility script to clean up unused type: ignore comments in Python files.

This script identifies and removes unused # type: ignore comments that are flagged by mypy.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Match, Set, Tuple

def run_mypy(file_path: str) -> str:
    """Run mypy on a file and capture the output.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        mypy output as a string
    """
    result = subprocess.run(
        ["python3", "-m", "mypy", file_path, "--ignore-missing-imports", "--pretty"],
        capture_output=True,
        text=True
    )
    return result.stdout

def extract_unused_ignores(mypy_output: str) -> List[Tuple[str, int]]:
    """Extract line numbers with unused type: ignore comments.
    
    Args:
        mypy_output: Output from mypy
        
    Returns:
        List of tuples containing file path and line number
    """
    pattern = r"([^:]+):(\d+): error: Unused \"type: ignore\" comment"
    matches = re.findall(pattern, mypy_output)
    return [(file, int(line)) for file, line in matches]

def fix_unused_ignores(file_path: str, line_numbers: List[int]) -> bool:
    """Remove unused type: ignore comments.
    
    Args:
        file_path: Path to the file to fix
        line_numbers: Line numbers with unused type: ignore comments
        
    Returns:
        True if changes were made, False otherwise
    """
    if not line_numbers:
        print(f"No unused type: ignore comments found in {file_path}")
        return False

    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove unused type: ignore comments
    modified = False
    for line_num in line_numbers:
        # Line numbers are 1-indexed, list indices are 0-indexed
        index = line_num - 1
        if index < 0 or index >= len(lines):
            print(f"Warning: Line {line_num} is out of range for {file_path}")
            continue

        line = lines[index]
        # Remove the type: ignore comment
        new_line = re.sub(r'\s*#\s*type:\s*ignore\s*(\[[^\]]+\])?.*$', '', line)
        if new_line != line:
            lines[index] = new_line
            modified = True
            print(f"Removed unused type: ignore comment at line {line_num} in {file_path}")

    # Write the file back if modified
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True
    
    return False

def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_unused_ignores.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    # Run mypy and get unused type: ignore comments
    mypy_output = run_mypy(file_path)
    unused_ignores = extract_unused_ignores(mypy_output)
    
    # Fix unused type: ignore comments
    unused_lines = [line for _, line in unused_ignores]
    if fix_unused_ignores(file_path, unused_lines):
        print(f"Successfully updated {file_path}")
        print("Running final mypy check...")
        final_output = run_mypy(file_path)
        if "error:" in final_output:
            print("Some errors still remain. Check mypy output for details.")
        else:
            print("All errors fixed successfully!")
    else:
        print(f"No changes made to {file_path}")

if __name__ == "__main__":
    main()
