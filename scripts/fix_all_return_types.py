#!/usr/bin/env python3
"""
Script to fix all return type issues in the ArangoClient class
by adding explicit type casting to method return values.
"""

import re
import sys

def fix_return_types(file_path):
    """
    Add explicit type casting to all methods with return type issues.
    
    Args:
        file_path: Path to the file to update
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Methods returning bool
    bool_methods = [
        "database_exists", "create_database", "delete_database", 
        "collection_exists", "delete_collection", "delete_document"
    ]
    
    # Methods returning Dict[str, Any]
    dict_methods = [
        "get_document", "insert_document", "update_document", 
        "replace_document", "get_edge", "create_edge"
    ]
    
    # Methods returning List[Dict[str, Any]]
    list_methods = ["list_documents"]
    
    # Methods returning int
    int_methods = ["count_documents"]
    
    # Define pattern for finding method return statements
    method_pattern = r'def\s+({}).*?return\s+(.*?)(?:$|\n\s+\w)'
    
    # Fix bool return methods
    bool_pattern = method_pattern.format('|'.join(bool_methods))
    content = re.sub(
        bool_pattern, 
        lambda m: m.group(0).replace(
            f"return {m.group(2)}", 
            f"return bool({m.group(2)})"
        ),
        content, 
        flags=re.DOTALL
    )
    
    # Fix Dict return methods
    dict_pattern = method_pattern.format('|'.join(dict_methods))
    content = re.sub(
        dict_pattern, 
        lambda m: m.group(0).replace(
            f"return {m.group(2)}", 
            f"return cast(Dict[str, Any], {m.group(2)})"
        ),
        content, 
        flags=re.DOTALL
    )
    
    # Fix List return methods
    list_pattern = method_pattern.format('|'.join(list_methods))
    content = re.sub(
        list_pattern, 
        lambda m: m.group(0).replace(
            f"return {m.group(2)}", 
            f"return cast(List[Dict[str, Any]], {m.group(2)})"
        ),
        content, 
        flags=re.DOTALL
    )
    
    # Fix int return methods
    int_pattern = method_pattern.format('|'.join(int_methods))
    content = re.sub(
        int_pattern, 
        lambda m: m.group(0).replace(
            f"return {m.group(2)}", 
            f"return int({m.group(2)})"
        ),
        content, 
        flags=re.DOTALL
    )
    
    # Write updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated return types in {file_path}")

if __name__ == "__main__":
    file_path = '/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py'
    fix_return_types(file_path)
