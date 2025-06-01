#!/usr/bin/env python3
"""
Targeted fix for incompatible types in python_code_adapter.py
"""
import re

file_path = "src/docproc/adapters/python_code_adapter.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix for incompatible types by adding cast to Dict[str, Any]
pattern1 = r"([ \t]+)processed = self\.python_adapter\.process_text\((.*?)\)"
replacement1 = r"\1processed = cast(Dict[str, Any], self.python_adapter.process_text(\2))"

# Apply the replacement
modified_content = re.sub(pattern1, replacement1, content)

# Write back to file
with open(file_path, 'w') as f:
    f.write(modified_content)

print(f"Fixed incompatible types in {file_path}")
