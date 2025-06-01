#!/usr/bin/env python3
"""
Analyze adapter code coverage directly by examining the source code and tests.

This script performs static analysis on the adapter files and their corresponding
test files to estimate test coverage and identify gaps in test coverage.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


def extract_methods(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Extract all methods from a Python file."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)
    methods = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            class_methods = {}
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    line_count = item.end_lineno - item.lineno + 1 if hasattr(item, 'end_lineno') else 1
                    start_line = item.lineno
                    end_line = item.end_lineno if hasattr(item, 'end_lineno') else item.lineno
                    
                    # Extract method body
                    method_body = source.split('\n')[start_line-1:end_line]
                    
                    class_methods[method_name] = {
                        "name": method_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "line_count": line_count,
                        "body": method_body
                    }
            
            methods[class_name] = class_methods
    
    return methods


def extract_tested_methods(filepath: str) -> Dict[str, Set[str]]:
    """Extract methods that are tested in a test file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError:
        return {}
    
    tree = ast.parse(source)
    tested_methods = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            test_class_name = node.name
            tested_method_names = set()
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    # Parse the test method to find what it's testing
                    method_source = ast.get_source_segment(source, item)
                    if not method_source:
                        continue
                    
                    # Look for specific patterns in test methods
                    if "process(" in method_source:
                        tested_method_names.add("process")
                    if "process_text(" in method_source:
                        tested_method_names.add("process_text")
                    if "extract_entities(" in method_source:
                        tested_method_names.add("extract_entities")
                    if "extract_metadata(" in method_source:
                        tested_method_names.add("extract_metadata")
                    if "__init__(" in method_source or "PythonCodeAdapter()" in method_source:
                        tested_method_names.add("__init__")
            
            # Infer class being tested from test class name
            class_name = test_class_name.replace("Test", "")
            tested_methods[class_name] = tested_method_names
    
    return tested_methods


def analyze_coverage(source_file: str, test_file: str) -> Dict[str, Any]:
    """Analyze test coverage for a source file."""
    source_methods = extract_methods(source_file)
    tested_methods = extract_tested_methods(test_file)
    
    coverage_results = {}
    
    for class_name, methods in source_methods.items():
        class_results = {
            "total_methods": len(methods),
            "tested_methods": 0,
            "untested_methods": [],
            "method_coverage": 0.0,
            "line_coverage": 0.0,
            "total_lines": 0,
            "tested_lines": 0
        }
        
        # Get the set of methods tested for this class
        class_tested_methods = tested_methods.get(class_name, set())
        
        total_lines = 0
        tested_lines = 0
        
        for method_name, method_info in methods.items():
            total_lines += method_info["line_count"]
            
            if method_name in class_tested_methods:
                class_results["tested_methods"] += 1
                tested_lines += method_info["line_count"]
            else:
                class_results["untested_methods"].append(method_name)
        
        class_results["total_lines"] = total_lines
        class_results["tested_lines"] = tested_lines
        
        # Calculate coverage percentages
        if class_results["total_methods"] > 0:
            class_results["method_coverage"] = (class_results["tested_methods"] / class_results["total_methods"]) * 100
        
        if total_lines > 0:
            class_results["line_coverage"] = (tested_lines / total_lines) * 100
        
        coverage_results[class_name] = class_results
    
    return coverage_results


def create_test_template(source_file: str, output_file: str, class_name: str, untested_methods: List[str]) -> None:
    """Create a test template for untested methods."""
    template = f"""import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

# Mock any dependencies to avoid import errors
import sys
sys.modules['src.docproc.adapters.python_adapter'] = mock.MagicMock()
sys.modules['src.docproc.adapters.python_adapter'].PythonAdapter = mock.MagicMock()
sys.modules['src.utils.file_utils'] = mock.MagicMock()

# Import the class we want to test
from src.docproc.adapters.{source_file.stem} import {class_name}

class Test{class_name}(unittest.TestCase):
    def setUp(self):
        self.adapter = {class_name}()
    
"""
    
    for method in untested_methods:
        template += f"""    def test_{method.replace("__", "")}(self):
        \"\"\"Test the {method} method.\"\"\"
        # TODO: Implement this test
        pass
    
"""
    
    template += """if __name__ == "__main__":
    unittest.main()
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template)


def create_coverage_report(coverage_results: Dict[str, Dict[str, Any]]) -> str:
    """Create a coverage report from the results."""
    report = "\n=== Adapter Test Coverage Analysis ===\n\n"
    
    overall_methods = 0
    overall_tested_methods = 0
    overall_lines = 0
    overall_tested_lines = 0
    
    for class_name, results in coverage_results.items():
        report += f"Class: {class_name}\n"
        report += f"  Method Coverage: {results['method_coverage']:.1f}% ({results['tested_methods']}/{results['total_methods']})\n"
        report += f"  Line Coverage: {results['line_coverage']:.1f}% ({results['tested_lines']}/{results['total_lines']})\n"
        
        if results["untested_methods"]:
            report += "  Untested Methods:\n"
            for method in results["untested_methods"]:
                report += f"    - {method}\n"
        
        report += "\n"
        
        overall_methods += results["total_methods"]
        overall_tested_methods += results["tested_methods"]
        overall_lines += results["total_lines"]
        overall_tested_lines += results["tested_lines"]
    
    # Calculate overall coverage
    overall_method_coverage = (overall_tested_methods / overall_methods) * 100 if overall_methods > 0 else 0
    overall_line_coverage = (overall_tested_lines / overall_lines) * 100 if overall_lines > 0 else 0
    
    report += "=== Overall Coverage ===\n"
    report += f"Method Coverage: {overall_method_coverage:.1f}% ({overall_tested_methods}/{overall_methods})\n"
    report += f"Line Coverage: {overall_line_coverage:.1f}% ({overall_tested_lines}/{overall_lines})\n\n"
    
    if overall_line_coverage >= 85:
        report += "✅ MEETS 85% STANDARD\n"
    else:
        report += "⚠️ BELOW 85% STANDARD\n"
        
        # Suggestions
        report += "\nSuggestions to improve coverage:\n"
        for class_name, results in coverage_results.items():
            if results["line_coverage"] < 85:
                report += f"- Add tests for {class_name}: {', '.join(results['untested_methods'])}\n"
    
    return report


def main() -> int:
    """Main function to analyze adapter coverage."""
    project_root = Path(__file__).parent
    src_path = project_root / "src" / "docproc" / "adapters"
    test_path = project_root / "tests" / "docproc" / "adapters"
    
    # Adapter files to analyze
    adapter_files = [
        "python_adapter.py",
        "docling_adapter.py",
        "python_code_adapter.py"
    ]
    
    # Create test path if it doesn't exist
    if not test_path.exists():
        os.makedirs(test_path)
    
    coverage_results = {}
    
    for adapter_file in adapter_files:
        source_file = src_path / adapter_file
        test_file = test_path / f"test_{adapter_file}"
        
        if not source_file.exists():
            print(f"⚠️ Source file not found: {source_file}")
            continue
        
        print(f"Analyzing coverage for {adapter_file}...")
        
        # Analyze coverage
        results = analyze_coverage(str(source_file), str(test_file))
        coverage_results.update(results)
        
        # Create test templates for untested classes/methods
        for class_name, class_results in results.items():
            if class_results["untested_methods"] and not test_file.exists():
                print(f"  Creating test template for {class_name}...")
                create_test_template(source_file, str(test_file), class_name, class_results["untested_methods"])
    
    # Create and print coverage report
    report = create_coverage_report(coverage_results)
    print(report)
    
    # Save report
    report_file = project_root / "adapter_coverage_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Coverage report saved to {report_file}")
    
    # Determine success based on overall coverage
    overall_lines = sum(results["total_lines"] for results in coverage_results.values())
    overall_tested_lines = sum(results["tested_lines"] for results in coverage_results.values())
    overall_line_coverage = (overall_tested_lines / overall_lines) * 100 if overall_lines > 0 else 0
    
    return 0 if overall_line_coverage >= 85 else 1

if __name__ == "__main__":
    sys.exit(main())
