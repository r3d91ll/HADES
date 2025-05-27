#!/usr/bin/env python3
"""
Test file for verifying YAML adapter type fixes.

This module runs mypy on the yaml_adapter.py file to validate type fixes.
"""

import subprocess
from pathlib import Path

def test_yaml_adapter():
    """Run mypy on yaml_adapter.py and report errors."""
    result = subprocess.run(
        ["mypy", "src/docproc/adapters/yaml_adapter.py"],
        capture_output=True,
        text=True
    )
    
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("\nMyPy output:")
        print(result.stdout)
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)

if __name__ == "__main__":
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent.parent
    
    print(f"Testing YAML adapter at {project_root}")
    test_yaml_adapter()
