#!/usr/bin/env python3
"""
Script to analyze test coverage for adapter files by comparing adapter methods to test cases.
"""

import os
import ast
import sys
from typing import Dict, List, Set, Tuple

# Configure paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
ADAPTERS_DIR = os.path.join(SRC_DIR, "docproc", "adapters")
ADAPTER_TESTS_DIR = os.path.join(TESTS_DIR, "docproc", "adapters")

# Adapters we've been working with
ADAPTERS_OF_INTEREST = [
    "python_adapter.py",
    "docling_adapter.py",
    "python_code_adapter.py"
]

class MethodVisitor(ast.NodeVisitor):
    """AST visitor to collect method and function definitions."""
    
    def __init__(self):
        self.methods = []
        self.classes = []
        
    def visit_FunctionDef(self, node):
        self.methods.append(node.name)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)

class TestMethodVisitor(ast.NodeVisitor):
    """AST visitor to collect test method definitions."""
    
    def __init__(self):
        self.test_methods = []
        
    def visit_FunctionDef(self, node):
        if node.name.startswith("test_"):
            self.test_methods.append(node.name)
        self.generic_visit(node)

def parse_file(file_path: str) -> ast.Module:
    """Parse a Python file into an AST."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return ast.parse(content)

def get_module_methods(file_path: str) -> Tuple[List[str], List[str]]:
    """Get all method and class names defined in a module."""
    try:
        tree = parse_file(file_path)
        visitor = MethodVisitor()
        visitor.visit(tree)
        return visitor.methods, visitor.classes
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []

def get_test_methods(file_path: str) -> List[str]:
    """Get all test method names defined in a test file."""
    try:
        tree = parse_file(file_path)
        visitor = TestMethodVisitor()
        visitor.visit(tree)
        return visitor.test_methods
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def find_matching_test_file(adapter_file: str) -> str:
    """Find the corresponding test file for an adapter."""
    base_name = os.path.basename(adapter_file)
    adapter_name = os.path.splitext(base_name)[0]
    test_file = os.path.join(ADAPTER_TESTS_DIR, f"test_{adapter_name}.py")
    if os.path.exists(test_file):
        return test_file
    return None

def analyze_adapter_coverage():
    """Analyze test coverage for adapter files."""
    print("\n=== Adapter Test Coverage Analysis ===\n")
    
    for adapter_file in ADAPTERS_OF_INTEREST:
        adapter_path = os.path.join(ADAPTERS_DIR, adapter_file)
        if not os.path.exists(adapter_path):
            print(f"Adapter file not found: {adapter_path}")
            continue
            
        methods, classes = get_module_methods(adapter_path)
        
        print(f"\n## {adapter_file}")
        print(f"Classes defined: {len(classes)}")
        print(f"Methods/functions defined: {len(methods)}")
        
        # Find matching test file
        test_file = find_matching_test_file(adapter_path)
        if test_file and os.path.exists(test_file):
            test_methods = get_test_methods(test_file)
            print(f"Test file: {os.path.basename(test_file)}")
            print(f"Test methods defined: {len(test_methods)}")
            
            # Simple coverage estimate
            coverage_ratio = min(1.0, len(test_methods) / max(1, len(methods)))
            coverage_percent = coverage_ratio * 100
            print(f"Estimated test coverage: {coverage_percent:.1f}%")
            
            if coverage_percent < 85:
                print("⚠️ BELOW 85% STANDARD: More tests needed")
            else:
                print("✅ MEETS 85% STANDARD")
                
            print("\nTest methods:")
            for method in test_methods:
                print(f"  - {method}")
        else:
            print("❌ No matching test file found")
            print("⚠️ BELOW 85% STANDARD: No tests")
            
        print("\nAdapter methods/functions:")
        for method in methods:
            print(f"  - {method}")

if __name__ == "__main__":
    analyze_adapter_coverage()
