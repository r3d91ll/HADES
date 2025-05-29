#!/usr/bin/env python3
"""
Fix type issues in the docproc module.

This script systematically fixes type annotation issues in the docproc module:
1. Adds proper import statements for typing modules
2. Ensures proper return type annotations on all functions
3. Adds type annotations for parameters and variables
4. Converts untyped decorators to typed versions
5. Adds proper casting for safer type handling

Usage:
    python3 fix_docproc_type_issues.py

The script will automatically scan the docproc module and apply fixes.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Pattern, Match, Union, Callable

# Root directory of the HADES project
HADES_ROOT = Path(__file__).parent.parent
DOCPROC_DIR = HADES_ROOT / "src" / "docproc"

# Regex patterns for finding and fixing common type issues
MISSING_RETURN_TYPE_PATTERN = re.compile(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)\s*:(?!\s*->\s*)', re.DOTALL)
UNTYPED_VARIABLE_PATTERN = re.compile(r'(\s+)([a-zA-Z0-9_]+)\s*=\s*(?!.*:\s*[A-Za-z\[\]]+)')
UNTYPED_DICT_PATTERN = re.compile(r'(\s+)([a-zA-Z0-9_]+)\s*=\s*\{\}(?!\s*#\s*type:)')
UNTYPED_LIST_PATTERN = re.compile(r'(\s+)([a-zA-Z0-9_]+)\s*=\s*\[\](?!\s*#\s*type:)')

# Map of function names to their return types (for functions with complex return types)
FUNCTION_RETURN_TYPES = {
    "process": "Dict[str, Any]",
    "process_text": "Dict[str, Any]",
    "extract_metadata": "Dict[str, Any]",
    "extract_entities": "List[Dict[str, Any]]",
    "load_config": "Dict[str, Any]",
    "detect_format": "str",
    "get_content_category": "str",
    "get_adapter_for_format": "BaseAdapter",
    "select_adapter_for_document": "BaseAdapter",
    "process_document": "Dict[str, Any]",
    "process_documents_batch": "Dict[str, Any]",
    "save_processed_document": "Path",
    "validate_document": "bool",
}

# Common parameter types
PARAM_TYPES = {
    "file_path": "Union[str, Path]",
    "options": "Optional[Dict[str, Any]]",
    "text": "str",
    "content": "str",
    "format_type": "str",
    "document": "Dict[str, Any]",
    "output_path": "Union[str, Path]",
    "validate": "bool",
    "on_success": "Optional[Callable[[Dict[str, Any], Path], None]]",
    "on_error": "Optional[Callable[[str, Exception], None]]",
}


def fix_imports(content: str) -> str:
    """Add necessary import statements for typing modules."""
    typing_imports = "from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast, Callable, TypeVar"
    path_import = "from pathlib import Path"
    
    # Check if these imports are already present
    if "from typing import" not in content:
        # Add after existing imports
        if "import" in content:
            content = re.sub(
                r'((?:^import.*?$|^from.*?import.*?$)(?:\n|$))+',
                r'\g<0>\n' + typing_imports + '\n',
                content,
                flags=re.MULTILINE
            )
        else:
            # Add at the top if no imports exist
            content = typing_imports + '\n\n' + content
    
    # Add Path import if not present and file_path is used
    if "from pathlib import Path" not in content and "Path" in content:
        if "import" in content:
            content = re.sub(
                r'((?:^import.*?$|^from.*?import.*?$)(?:\n|$))+',
                r'\g<0>\n' + path_import + '\n',
                content,
                flags=re.MULTILINE
            )
        else:
            content = path_import + '\n\n' + content
    
    return content


def fix_missing_return_types(content: str) -> str:
    """Add return type annotations to functions that are missing them."""
    def replacement(match: Match) -> str:
        func_name = match.group(1)
        params = match.group(2)
        
        # Default return type is Any if not specified
        return_type = FUNCTION_RETURN_TYPES.get(func_name, "Any")
        
        # Special case for __init__ and other functions that don't return a value
        if func_name == "__init__" or func_name.startswith("_"):
            return_type = "None"
        
        return f"def {func_name}({params}) -> {return_type}:"
    
    return MISSING_RETURN_TYPE_PATTERN.sub(replacement, content)


def fix_untyped_variables(content: str) -> str:
    """Add type annotations to untyped variables."""
    def dict_replacement(match: Match) -> str:
        indent = match.group(1)
        var_name = match.group(2)
        return f"{indent}{var_name}: Dict[str, Any] = {{}}"
    
    def list_replacement(match: Match) -> str:
        indent = match.group(1)
        var_name = match.group(2)
        return f"{indent}{var_name}: List[Any] = []"
    
    content = UNTYPED_DICT_PATTERN.sub(dict_replacement, content)
    content = UNTYPED_LIST_PATTERN.sub(list_replacement, content)
    return content


def fix_function_parameters(content: str) -> str:
    """Add type annotations to function parameters."""
    def add_param_type(match: Match) -> str:
        func_def = match.group(0)
        func_name = match.group(1)
        params_str = match.group(2)
        
        # Skip if already properly typed
        if ":" in params_str:
            return func_def
        
        # Handle parameters
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        typed_params = []
        
        for param in params:
            # Skip self and cls
            if param in ["self", "cls"]:
                typed_params.append(param)
                continue
                
            # Handle default values
            if "=" in param:
                param_name, default = param.split("=", 1)
                param_name = param_name.strip()
                param_type = PARAM_TYPES.get(param_name, "Any")
                typed_params.append(f"{param_name}: {param_type} = {default}")
            else:
                param_name = param.strip()
                param_type = PARAM_TYPES.get(param_name, "Any")
                typed_params.append(f"{param_name}: {param_type}")
        
        return f"def {func_name}({', '.join(typed_params)})"
    
    return re.sub(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)', add_param_type, content)


def fix_yaml_imports(content: str) -> str:
    """Fix imports and type annotations for YAML handling."""
    # Add type annotations for yaml imports
    if "import yaml" in content and "from typing import" not in content:
        content = re.sub(
            r'import yaml',
            'import yaml\nfrom typing import Any, Dict, List, Optional, Union, cast',
            content
        )
    
    # Fix functions that use yaml
    content = re.sub(
        r'def parse_yaml_metadata\((.*?)\):',
        r'def parse_yaml_metadata(\1) -> Dict[str, Any]:',
        content
    )
    
    content = re.sub(
        r'def load_yaml\((.*?)\):',
        r'def load_yaml(\1) -> Dict[str, Any]:',
        content
    )
    
    return content


def fix_file(file_path: Path) -> None:
    """Apply all type fixes to a single file."""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply fixes
    content = fix_imports(content)
    content = fix_missing_return_types(content)
    content = fix_untyped_variables(content)
    content = fix_function_parameters(content)
    
    # Apply special fixes for certain files
    if file_path.name in ["yaml_adapter.py", "markdown_adapter.py"]:
        content = fix_yaml_imports(content)
    
    # Write back the updated content
    with open(file_path, 'w') as f:
        f.write(content)


def process_directory(directory: Path) -> None:
    """Process all Python files in a directory recursively."""
    for path in directory.glob("**/*.py"):
        fix_file(path)


def add_typed_validator_wrappers(content: str) -> str:
    """Add typed wrappers for decorators if needed."""
    # Check if the file uses Pydantic validators
    if "field_validator" in content and "typed_field_validator" not in content:
        # Add typed wrapper similar to what was done for schemas
        typed_wrapper = """
# Type variables for validator functions
T = TypeVar('T')
ValidatorFunc = Callable[[Any, T], T]

# Create a typed wrapper for field validators
def typed_field_validator(field_name: str) -> Callable[[Callable[[Any, T], T]], Callable[[Any, T], T]]:
    \"\"\"Typed wrapper for field_validator to make mypy happy.\"\"\"
    def decorator(func: Callable[[Any, T], T]) -> Callable[[Any, T], T]:
        return cast(Callable[[Any, T], T], field_validator(field_name)(func))
    return decorator
"""
        # Add after imports
        content = re.sub(
            r'((?:^import.*?$|^from.*?import.*?$)(?:\n|$))+',
            r'\g<0>\n' + typed_wrapper + '\n',
            content,
            flags=re.MULTILINE
        )
        
        # Replace field_validator with typed_field_validator
        content = re.sub(
            r'@field_validator\([\'"]([^\'"]+)[\'"]\)',
            r'@typed_field_validator("\1")',
            content
        )
    
    return content


def main() -> None:
    """Main entry point for the script."""
    if not DOCPROC_DIR.exists():
        print(f"Directory not found: {DOCPROC_DIR}")
        return
    
    print(f"Fixing type issues in: {DOCPROC_DIR}")
    process_directory(DOCPROC_DIR)
    print("Type fixing complete!")


if __name__ == "__main__":
    main()
