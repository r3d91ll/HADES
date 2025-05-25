#!/usr/bin/env python3
"""
Script to fix syntax issues in the ArangoClient implementation.
"""

def fix_syntax_issues(file_path):
    """Fix syntax issues in the ArangoClient implementation."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Track fixes
    fixes_applied = 0
    
    # Process the file line by line
    for i in range(len(lines)):
        # Fix extra parentheses in return statements
        if "return" in lines[i] and ")))" in lines[i]:
            lines[i] = lines[i].replace(")))", "))")
            fixes_applied += 1
        
        if "return" in lines[i] and "))" in lines[i] and "cast" in lines[i]:
            # Only fix if there are too many closing parentheses
            open_count = lines[i].count("(")
            close_count = lines[i].count(")")
            if close_count > open_count:
                lines[i] = lines[i].replace("))", ")")
                fixes_applied += 1
        
        # Fix missing closing parentheses
        if "bool(False" in lines[i] and ")" not in lines[i]:
            lines[i] = lines[i].rstrip() + ")\n"
            fixes_applied += 1
        
        if "None" in lines[i] and "cast" in lines[i] and ")" not in lines[i]:
            lines[i] = lines[i].rstrip() + ")\n"
            fixes_applied += 1
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Applied {fixes_applied} syntax fixes to {file_path}")

if __name__ == "__main__":
    fix_syntax_issues('/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py')
