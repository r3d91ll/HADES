#!/usr/bin/env python3
"""
Utility script to fix type issues in the docproc module.

This script addresses common mypy errors by making targeted changes to 
source files after migrating the type definitions to centralized type files.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Union, Optional


def fix_python_document_imports() -> None:
    """Fix import statements in files that reference the old type definitions."""
    # List of files that might need import fixes
    files_to_check = [
        Path('src/docproc/adapters/python_adapter.py'),
        Path('src/docproc/parsers/python_parser.py'),
        Path('src/docproc/processors/python_processor.py'),
        Path('src/chunking/chunkers/python_chunker.py')
    ]
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        print(f"Checking {file_path} for import fixes...")
        content = file_path.read_text()
        
        # Replace direct imports from old files with centralized types
        old_imports = [
            r'from src\.docproc\.models\.python_code import (.*?)RelationshipType(.*?)',
            r'from src\.docproc\.models\.python_code import (.*?)AccessLevel(.*?)',
            r'from src\.docproc\.models\.python_code import (.*?)ImportSourceType(.*?)',
            r'from src\.docproc\.models\.python_code import (.*?)CodeRelationship(.*?)',
            r'from src\.docproc\.schemas\.python_document import (.*?)PythonDocument(.*?)',
            r'from src\.docproc\.schemas\.python_document import (.*?)PythonMetadata(.*?)',
            r'from src\.docproc\.schemas\.python_document import (.*?)PythonEntity(.*?)',
            r'from src\.docproc\.schemas\.python_document import (.*?)CodeElement(.*?)',
            r'from src\.docproc\.schemas\.python_document import (.*?)SymbolTable(.*?)'
        ]
        
        new_imports = [
            r'from src.types.docproc.enums import \1RelationshipType\2',
            r'from src.types.docproc.enums import \1AccessLevel\2',
            r'from src.types.docproc.enums import \1ImportSourceType\2',
            r'from src.types.docproc.code_elements import \1CodeRelationship\2',
            r'from src.types.docproc.python import \1PythonDocument\2',
            r'from src.types.docproc.python import \1PythonMetadata\2',
            r'from src.types.docproc.python import \1PythonEntity\2',
            r'from src.types.docproc.python import \1CodeElement\2',
            r'from src.types.docproc.python import \1SymbolTable\2'
        ]
        
        modified = False
        for old, new in zip(old_imports, new_imports):
            if re.search(old, content):
                content = re.sub(old, new, content)
                modified = True
        
        if modified:
            file_path.write_text(content)
            print(f"Fixed imports in {file_path}")
        else:
            print(f"No import fixes needed in {file_path}")


def fix_validator_decorators() -> None:
    """Fix untyped validator decorators to use typed versions."""
    files_to_check = [
        Path('src/docproc/schemas/python_document.py'),
        Path('src/docproc/schemas/base.py')
    ]
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        print(f"Checking {file_path} for validator decorator fixes...")
        content = file_path.read_text()
        
        # Replace field_validator with typed_field_validator
        if "@field_validator" in content and "@typed_field_validator" not in content:
            content = re.sub(
                r'@field_validator\((".*?")\)',
                r'@typed_field_validator(\1)',
                content
            )
            
        # Replace model_validator with typed_model_validator
        if "@model_validator" in content and "@typed_model_validator" not in content:
            content = re.sub(
                r'@model_validator\(mode=(".*?")\)',
                r'@typed_model_validator(mode=\1)',
                content
            )
            
        # Make sure typed validators are imported
        if re.search(r'@typed_[a-z_]+_validator', content) and "typed_field_validator" not in content:
            if "from src.types.docproc.python import" in content:
                content = re.sub(
                    r'from src\.types\.docproc\.python import (.*)',
                    r'from src.types.docproc.python import \1, typed_field_validator, typed_model_validator',
                    content
                )
            else:
                content = re.sub(
                    r'from pydantic import (.*)',
                    r'from pydantic import \1\n\nfrom src.types.docproc.python import typed_field_validator, typed_model_validator',
                    content
                )
        
        file_path.write_text(content)
        print(f"Fixed validator decorators in {file_path}")


def run_mypy_check() -> None:
    """Run mypy on the docproc module to check for type issues."""
    import subprocess
    
    print("\nRunning mypy check on docproc module...")
    result = subprocess.run(
        ["mypy", "src/docproc", "--show-column-numbers", "--pretty"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("No type issues found! 🎉")
    else:
        print("Type issues found:")
        print(result.stdout)
        print(result.stderr)
        

if __name__ == "__main__":
    print("Running docproc type fixes...")
    fix_python_document_imports()
    fix_validator_decorators()
    
    # Uncomment to run mypy check - be sure mypy is installed first
    # run_mypy_check()
    
    print("\nCompleted docproc type fixes!")
