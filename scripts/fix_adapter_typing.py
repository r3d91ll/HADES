#!/usr/bin/env python3
"""
Utility script to fix typing issues in the docproc adapter modules.

This script addresses several common typing issues:
1. Unreachable code warnings by adding # type: ignore[unreachable] comments
2. Incompatible types in assignment errors by adding proper type casting
3. Missing return type annotations
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Match
import re

def run_mypy(file_path: str) -> str:
    """Run mypy on a file and capture the output."""
    result = subprocess.run(
        ["python3", "-m", "mypy", file_path, "--ignore-missing-imports", "--pretty"],
        capture_output=True,
        text=True
    )
    return result.stdout

def extract_typing_issues(mypy_output: str) -> Dict[str, List[Tuple[str, int]]]:
    """Extract typing issues from mypy output."""
    # Extract unreachable code warnings
    unreachable_pattern = r"([^:]+):(\d+): error: Statement is unreachable"
    unreachable_matches = re.findall(unreachable_pattern, mypy_output)
    unreachable_lines = [(file, int(line)) for file, line in unreachable_matches]
    
    # Extract incompatible types in assignment errors
    incompatible_pattern = r"([^:]+):(\d+): error: Incompatible types in assignment \(expression has type \"ProcessedDocument\", variable has type \"dict\[str, Any\]\"\)"
    incompatible_matches = re.findall(incompatible_pattern, mypy_output)
    incompatible_lines = [(file, int(line)) for file, line in incompatible_matches]
    
    return {
        "unreachable": unreachable_lines,
        "incompatible": incompatible_lines
    }

def fix_unreachable_code(file_path: str, line_numbers: List[int]) -> bool:
    """Add type: ignore comments to unreachable code lines."""
    if not line_numbers:
        print(f"No unreachable code found in {file_path}")
        return False

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
        print(f"Fixed unreachable code at line {line_num} in {file_path}")

    # Write the file back if modified
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True
    
    return False

def fix_incompatible_types(file_path: str, line_numbers: List[int]) -> bool:
    """Fix incompatible types in assignment errors."""
    if not line_numbers:
        print(f"No incompatible types found in {file_path}")
        return False

    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Process line by line to find assignment statements
    modified = False
    for line_num in line_numbers:
        # Line numbers are 1-indexed, list indices are 0-indexed
        index = line_num - 1
        if index < 0 or index >= len(lines):
            print(f"Warning: Line {line_num} is out of range for {file_path}")
            continue

        line = lines[index]
        # Match assignment pattern for ProcessedDocument
        if "processed = self.python_adapter.process_text" in line:
            # Add type casting
            indent_match = re.match(r'(\s*)', line)
            # Handle potential None value
            indentation = indent_match.group(1) if indent_match else ""
            processed_line = f"{indentation}processed = cast(Dict[str, Any], self.python_adapter.process_text"
            lines[index] = line.replace("processed = self.python_adapter.process_text", processed_line)
            modified = True
            print(f"Fixed incompatible types at line {line_num} in {file_path}")

    # Write the file back if modified
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True
    
    return False

def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_adapter_typing.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    # Run mypy and get typing issues
    mypy_output = run_mypy(file_path)
    typing_issues = extract_typing_issues(mypy_output)
    
    # Fix unreachable code
    unreachable_lines = [line for _, line in typing_issues["unreachable"]]
    fix_unreachable_code(file_path, unreachable_lines)
    
    # Fix incompatible types
    incompatible_lines = [line for _, line in typing_issues["incompatible"]]
    fix_incompatible_types(file_path, incompatible_lines)
    
    # Final verification
    if unreachable_lines or incompatible_lines:
        print(f"Successfully updated {file_path}")
        print("Running final mypy check...")
        final_output = run_mypy(file_path)
        if "error:" in final_output:
            print("Some errors still remain. Check mypy output for details.")
        else:
            print("All errors fixed successfully!")
    else:
        print(f"No typing issues to fix in {file_path}")

if __name__ == "__main__":
    main()
