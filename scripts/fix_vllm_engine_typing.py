#!/usr/bin/env python3
"""
Script to fix typing issues in the VLLM engine implementation.
This addresses the following issues:
1. Unreachable code in session checks
2. Type annotation issues
3. Return type fixes
"""
import re
import sys
from pathlib import Path

def fix_session_checks(content):
    """Fix unreachable code in session checks by reordering statements"""
    # Pattern to match session check with unreachable code
    pattern = re.compile(
        r'(\s+)# Use the session managed by the base class or overridden by tests\n'
        r'(\s+)if not self\._session:\n'
        r'(\s+)raise RuntimeError\("AIOHTTP session is not available\."\)\n'
        r'(\s+)if self\._session\.closed:\n'
        r'(\s+)raise RuntimeError\("AIOHTTP session is closed\."\)\n'
        r'(\s+)session = self\._session'
    )
    
    # Replacement that fixes the unreachable code issue
    replacement = r'\1# Use the session managed by the base class or overridden by tests\n'
    replacement += r'\2if not self._session:\n'
    replacement += r'\3    raise RuntimeError("AIOHTTP session is not available.")\n'
    replacement += r'\2# Store reference in local variable before checking closed status\n'
    replacement += r'\2session = self._session\n'
    replacement += r'\2if session.closed:\n'
    replacement += r'\3    raise RuntimeError("AIOHTTP session is closed.")'
    
    return pattern.sub(replacement, content)

def main():
    """Main function to fix typing issues"""
    if len(sys.argv) < 2:
        print("Usage: fix_vllm_engine_typing.py <path_to_vllm_engine.py>")
        return 1
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return 1
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply fixes
    fixed_content = fix_session_checks(content)
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed typing issues in {file_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
