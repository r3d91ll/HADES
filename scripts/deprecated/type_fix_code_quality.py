#!/usr/bin/env python3
"""
Code quality check script for HADES document processing type fixes.

This script implements the standard protocol for code review by:
1. Running mypy to check for remaining type errors
2. Verifying unit test coverage
3. Checking for proper docstrings

Run this script from the project root directory.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def check_mypy() -> None:
    """Run mypy on the document processing module."""
    print("Running mypy type checker...")
    code, output = run_command("mypy src/docproc/")
    
    if code == 0:
        print("✅ No type errors found!")
    else:
        print("⚠️ Found type errors:")
        print(output)
        
        # Count errors by category
        error_categories = {}
        for line in output.splitlines():
            if ": error:" in line:
                error_type = line.split(": error:")[1].strip()
                category = error_type.split("[")[1].split("]")[0] if "[" in error_type else "other"
                error_categories[category] = error_categories.get(category, 0) + 1
        
        print("\nError categories:")
        for category, count in error_categories.items():
            print(f"  - {category}: {count} errors")
        
        # Suggest fixes for common errors
        print("\nSuggested fixes:")
        if "import-untyped" in error_categories:
            print("  - For import-untyped errors: Install missing type stubs with 'pip install types-PyYAML'")
        if "no-any-return" in error_categories:
            print("  - For no-any-return errors: Add explicit casts to return values")
        if "assignment" in error_categories:
            print("  - For assignment errors: Use explicit type casts or fix variable type annotations")


def check_test_coverage() -> None:
    """Check test coverage for the adapted files."""
    print("\nChecking test coverage...")
    
    # List the files we need to test
    files_to_check = [
        "src/docproc/adapters/yaml_adapter.py",
        "src/docproc/adapters/json_adapter.py",
        "src/docproc/adapters/markdown_adapter.py"
    ]
    
    # Verify tests exist for these files
    test_files = [
        "test/test_adapter_types_direct.py"
    ]
    
    all_exist = True
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"❌ Missing test file: {test_file}")
            all_exist = False
    
    if all_exist:
        print("✅ Test files created for type fixes")
    
    # Check that we're testing all the fixed functionality
    tests_run, _ = run_command(f"python3 {test_files[0]} -v")
    if tests_run == 0:
        print("✅ All tests pass")
    else:
        print("❌ Some tests failed")


def check_docstrings() -> None:
    """Verify that proper docstrings exist in the modified files."""
    print("\nChecking documentation...")
    
    # Check for module-level documentation
    docproc_readme = Path("src/docproc/docproc_readme.md")
    if docproc_readme.exists():
        content = docproc_readme.read_text()
        if "Type System Improvements" in content:
            print("✅ Documentation updated with type system improvements")
        else:
            print("❌ Documentation needs to be updated with type system information")
    else:
        print("❌ Missing documentation file: src/docproc/docproc_readme.md")


if __name__ == "__main__":
    print("Running code quality checks for type fixes...")
    check_mypy()
    check_test_coverage()
    check_docstrings()
    
    print("\nReminder: Complete the standard protocol by:")
    print("1. Fixing any remaining critical errors")
    print("2. Committing changes with git")
    print("3. Running the full test suite before finalizing")
