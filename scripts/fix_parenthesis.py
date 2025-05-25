#!/usr/bin/env python3
"""
Script to fix unmatched parenthesis in the ArangoClient implementation.
"""

def fix_unmatched_parenthesis(file_path):
    """Fix unmatched parenthesis in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix unmatched parenthesis
    content = content.replace('))))', ')))')
    content = content.replace(')))', '))')
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed unmatched parenthesis in {file_path}")

if __name__ == "__main__":
    fix_unmatched_parenthesis('/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py')
