#!/usr/bin/env python3
"""
Direct verification script for PythonCodeAdapter typing fixes.

This script directly tests the class structure and method signatures of the
PythonCodeAdapter to verify that our typing fixes are correct.
"""

import sys
import os
import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple

# Parse the Python code adapter file to analyze its structure
adapter_file = Path(__file__).parent / "src" / "docproc" / "adapters" / "python_code_adapter.py"

def get_class_info(file_path: str) -> Dict[str, Any]:
    """Extract class information from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return {}
    
    class_info = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Extract method information
                    arg_info = []
                    
                    # Get args
                    for arg in item.args.args:
                        arg_name = arg.arg
                        arg_type = "Any"
                        if arg.annotation:
                            if isinstance(arg.annotation, ast.Name):
                                arg_type = arg.annotation.id
                            elif isinstance(arg.annotation, ast.Subscript):
                                if isinstance(arg.annotation.value, ast.Name):
                                    arg_type = arg.annotation.value.id
                        arg_info.append({"name": arg_name, "type": arg_type})
                    
                    # Get return type
                    return_type = "None"
                    if item.returns:
                        if isinstance(item.returns, ast.Name):
                            return_type = item.returns.id
                        elif isinstance(item.returns, ast.Subscript):
                            if isinstance(item.returns.value, ast.Name):
                                return_type = item.returns.value.id
                    
                    methods.append({
                        "name": item.name,
                        "args": arg_info,
                        "return_type": return_type,
                        "line_no": item.lineno
                    })
            
            class_info[node.name] = {
                "methods": methods,
                "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                "line_no": node.lineno
            }
    
    return class_info

def verify_typing() -> None:
    """Verify typing of the PythonCodeAdapter class."""
    print(f"\n=== Analyzing PythonCodeAdapter typing ===\n")
    
    class_info = get_class_info(adapter_file)
    
    if not class_info:
        print("❌ Failed to extract class information")
        sys.exit(1)
    
    adapter_class = class_info.get("PythonCodeAdapter")
    if not adapter_class:
        print("❌ PythonCodeAdapter class not found")
        sys.exit(1)
    
    print(f"✅ Found PythonCodeAdapter class (line {adapter_class['line_no']})")
    print(f"✅ Base classes: {', '.join(adapter_class['bases'])}")
    
    methods = adapter_class["methods"]
    method_names = [m["name"] for m in methods]
    
    print(f"\nMethods found ({len(methods)}):")
    for method in sorted(methods, key=lambda m: m["line_no"]):
        args_str = ", ".join([f"{arg['name']}: {arg['type']}" for arg in method["args"]])
        print(f"  - {method['name']}({args_str}) -> {method['return_type']}")
    
    # Check for required methods
    required_methods = ["__init__", "process", "process_text", "extract_entities", "extract_metadata"]
    missing_methods = [m for m in required_methods if m not in method_names]
    
    if missing_methods:
        print(f"\n❌ Missing required methods: {', '.join(missing_methods)}")
    else:
        print(f"\n✅ All required methods are present")
    
    # Check typing for each method
    typing_issues = []
    
    # Check __init__
    init_method = next((m for m in methods if m["name"] == "__init__"), None)
    if init_method:
        if init_method["return_type"] != "None":
            typing_issues.append(f"__init__ should return None, not {init_method['return_type']}")
        
        options_arg = next((a for a in init_method["args"][1:] if a["name"] == "options"), None)
        if options_arg and "Optional" not in options_arg["type"] and "Union" not in options_arg["type"]:
            typing_issues.append(f"options parameter should be Optional, not {options_arg['type']}")
    
    # Check process
    process_method = next((m for m in methods if m["name"] == "process"), None)
    if process_method:
        if "ProcessedDocument" not in process_method["return_type"]:
            typing_issues.append(f"process should return ProcessedDocument, not {process_method['return_type']}")
    
    # Check process_text
    process_text_method = next((m for m in methods if m["name"] == "process_text"), None)
    if process_text_method:
        if "ProcessedDocument" not in process_text_method["return_type"]:
            typing_issues.append(f"process_text should return ProcessedDocument, not {process_text_method['return_type']}")
    
    # Check extract_entities
    extract_entities_method = next((m for m in methods if m["name"] == "extract_entities"), None)
    if extract_entities_method:
        if "List" not in extract_entities_method["return_type"]:
            typing_issues.append(f"extract_entities should return List[EntityDict], not {extract_entities_method['return_type']}")
    
    # Check extract_metadata
    extract_metadata_method = next((m for m in methods if m["name"] == "extract_metadata"), None)
    if extract_metadata_method:
        if "MetadataDict" not in extract_metadata_method["return_type"]:
            typing_issues.append(f"extract_metadata should return MetadataDict, not {extract_metadata_method['return_type']}")
    
    if typing_issues:
        print("\n❌ Typing issues found:")
        for issue in typing_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No typing issues found!")
    
    # Calculate coverage
    method_count = len(methods)
    required_count = len(required_methods)
    
    coverage = min(100.0, (method_count / required_count) * 100)
    print(f"\nMethod coverage: {coverage:.1f}% ({method_count}/{required_count})")
    
    if coverage >= 85.0:
        print("✅ MEETS 85% STANDARD")
    else:
        print("⚠️ BELOW 85% STANDARD")
    
    # Overall assessment
    if not missing_methods and not typing_issues:
        print("\n✅ PythonCodeAdapter typing is fully correct!")
    else:
        print("\n⚠️ PythonCodeAdapter has typing issues that need to be addressed")

if __name__ == "__main__":
    verify_typing()
