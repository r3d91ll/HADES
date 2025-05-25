#!/usr/bin/env python3
"""
Script to replace all standalone logger references with self.logger
in the ArangoClient class.
"""

import re

def fix_logger_references(file_path):
    """
    Replace 'logger.' with 'self.logger.' in the ArangoClient class.
    
    Args:
        file_path: Path to the file to update
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define the pattern to match - logger. that's not already self.logger.
    pattern = r'(?<!self\.)logger\.'
    
    # Replace all occurrences
    updated_content = re.sub(pattern, 'self.logger.', content)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated logger references in {file_path}")

if __name__ == "__main__":
    fix_logger_references('/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/database/arango_client.py')
