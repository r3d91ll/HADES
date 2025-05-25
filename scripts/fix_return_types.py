#!/usr/bin/env python3
"""
Script to add explicit type casts to return values in the ArangoClient class
to satisfy mypy type checking.
"""

import re

def fix_return_types(file_path):
    """
    Add explicit type casting to return values in ArangoClient methods.
    
    Args:
        file_path: Path to the file to update
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    method_signatures = {}
    current_method = None
    method_start_line = 0
    inside_method = False
    
    # First pass: find method signatures and their return types
    for i, line in enumerate(lines):
        # Check for method definition with return type annotation
        if re.search(r'^\s+def\s+\w+\s*\(.*\)\s*->\s*\w+', line):
            # Extract method name and return type
            match = re.search(r'^\s+def\s+(\w+).*->\s*(\w+|\w+\[.*\])', line)
            if match:
                current_method = match.group(1)
                return_type = match.group(2)
                method_signatures[current_method] = return_type
                method_start_line = i
                inside_method = True
        
        # Check for end of method (indentation decreases)
        elif inside_method and line.strip() and not line.startswith('    '):
            inside_method = False
    
    # List of methods to fix and their return patterns
    methods_to_fix = {
        "database_exists": {"return_type": "bool", "pattern": r'return\s+self\.client\.has_database'},
        "create_database": {"return_type": "bool", "pattern": r'return\s+\w+'},
        "delete_database": {"return_type": "bool", "pattern": r'return\s+self\.client\.delete_database'},
        "collection_exists": {"return_type": "bool", "pattern": r'return\s+collection_name in db\.collections\(\)'},
        "delete_collection": {"return_type": "bool", "pattern": r'return\s+True'},
        "list_documents": {"return_type": "List[Dict[str, Any]]", "pattern": r'return\s+list\(cursor\)'},
        "count_documents": {"return_type": "int", "pattern": r'return\s+\w+'},
        "get_document": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'},
        "insert_document": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'},
        "update_document": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'},
        "replace_document": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'},
        "delete_document": {"return_type": "bool", "pattern": r'return\s+True'},
        "get_edge": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'},
        "create_edge": {"return_type": "Dict[str, Any]", "pattern": r'return\s+\w+'}
    }
    
    updated_lines = lines.copy()
    changes_made = 0
    
    # Second pass: add type casting to return statements
    for method_name, info in methods_to_fix.items():
        for i, line in enumerate(lines):
            if re.search(info["pattern"], line) and method_name in "".join(lines[max(0, i-20):i]):
                # Extract the existing return expression
                match = re.search(r'return\s+(.*)', line)
                if match:
                    expr = match.group(1)
                    # Replace with cast version
                    if info["return_type"] == "bool":
                        updated_lines[i] = line.replace(f"return {expr}", f"return bool({expr})")
                    elif info["return_type"] == "int":
                        updated_lines[i] = line.replace(f"return {expr}", f"return int({expr})")
                    elif "Dict" in info["return_type"]:
                        updated_lines[i] = line.replace(f"return {expr}", 
                                                       f"return cast(Dict[str, Any], {expr})")
                    elif "List" in info["return_type"]:
                        updated_lines[i] = line.replace(f"return {expr}", 
                                                       f"return cast(List[Dict[str, Any]], {expr})")
                    changes_made += 1
    
    # Write back to the file if changes were made
    if changes_made > 0:
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated {changes_made} return type castings in {file_path}")
    else:
        print(f"No changes needed in {file_path}")

if __name__ == "__main__":
    fix_return_types('/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py')
