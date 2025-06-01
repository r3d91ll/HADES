#!/usr/bin/env python3
"""
Run coverage tests for the docproc module.

This script runs the test suite for docproc with coverage reporting to verify
that we meet the 85% coverage standard required by the project.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_coverage_tests() -> int:
    """Run tests with coverage for the docproc module."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Ensure the src directory is in Python's path
    src_path = str(project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print("\n=== Running coverage tests for docproc module ===\n")
    
    # Run coverage for all docproc tests
    try:
        # First, make sure coverage is installed
        subprocess.run(
            ["pip", "install", "coverage"],
            check=True,
            capture_output=True,
        )
        
        # Run coverage on the docproc tests
        subprocess.run(
            [
                "coverage", "run", "--source=src/docproc",
                "-m", "unittest", "discover", "-s", "tests/docproc",
            ],
            check=True,
        )
        
        # Generate coverage report
        subprocess.run(["coverage", "report", "-m"], check=True)
        
        # Check if coverage is sufficient
        result = subprocess.run(
            ["coverage", "report", "--fail-under=85"],
            capture_output=True,
        )
        
        if result.returncode == 0:
            print("\n✅ Coverage meets the 85% standard!")
            return 0
        else:
            print("\n❌ Coverage is below the 85% standard.")
            
            # Run specific reports for adapter modules to identify gaps
            print("\n=== Adapter Module Coverage ===")
            adapter_files = [
                "src/docproc/adapters/python_adapter.py",
                "src/docproc/adapters/docling_adapter.py",
                "src/docproc/adapters/python_code_adapter.py",
            ]
            
            for file in adapter_files:
                result = subprocess.run(
                    ["coverage", "report", "-m", file],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"\nCoverage for {file}:")
                    coverage_line = [line for line in result.stdout.split('\n') if file in line]
                    if coverage_line:
                        print(coverage_line[0])
            
            return 1
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running tests: {e}")
        if e.stdout:
            print(f"Output: {e.stdout.decode()}")
        if e.stderr:
            print(f"Error: {e.stderr.decode()}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_coverage_tests())
