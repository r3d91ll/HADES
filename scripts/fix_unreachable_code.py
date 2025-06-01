#!/usr/bin/env python3
"""
Utility script to fix unreachable code warnings in Python files.

This script adds type: ignore comments to lines that mypy reports as unreachable.
"""

import re
import sys
import subprocess
from pathlib import Path

def run_mypy(file_path):
    """Run mypy on a file and capture the output."""
    result = subprocess.run(
        ["python3", "-m", "mypy", file_path, "--ignore-missing-imports", "--pretty"],
        capture_output=True,
        text=True
    )
    return result.stdout

def extract_unreachable_lines(mypy_output):
    """Extract line numbers with unreachable code warnings."""
    pattern = r"([^:]+):(\d+): error: Statement is unreachable"
    matches = re.findall(pattern, mypy_output)
    return [(file, int(line)) for file, line in matches]

def fix_unreachable_code(file_path, line_numbers):
    """Add type: ignore comments to unreachable code lines."""
    if not line_numbers:
        print(f"No unreachable code found in {file_path}")
        return

    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Add # type: ignore comments
    modified = False
    for line_num in line_numbers:
        # Line numbers are 1-indexed, list indices are 0-indexed
        index = line_num - 1
        if index < 0 or index >= len(lines):
            print(f"Warning: Line {line_num} is out of range for {file_path}")
            continue

        # Skip if already has a type: ignore comment
        if "# type: ignore" in lines[index]:
            continue

        # Add the comment
        line = lines[index].rstrip()
        lines[index] = f"{line}  # type: ignore[unreachable]\n"
        modified = True
        print(f"Fixed line {line_num} in {file_path}")

    # Write the file back if modified
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Successfully updated {file_path}")
    else:
        print(f"No changes needed for {file_path}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_unreachable_code.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    # Run mypy and get unreachable code warnings
    mypy_output = run_mypy(file_path)
    unreachable_lines = extract_unreachable_lines(mypy_output)
    
    # Fix the unreachable code
    fix_unreachable_code(file_path, [line for _, line in unreachable_lines])

if __name__ == "__main__":
    main()
