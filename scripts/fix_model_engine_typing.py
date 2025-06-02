#!/usr/bin/env python3
"""
Utility script to fix common typing issues in the model_engine module.

This script applies systematic fixes to typing issues in the model_engine module,
similar to how we fixed the schema typing issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import Union

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Regex patterns for common typing issues
MISSING_RETURN_TYPE = re.compile(r'def\s+(\w+)\s*\(([^)]*)\)\s*:')
MISSING_PARAM_TYPE = re.compile(r'def\s+\w+\s*\(([^)]*)\)\s*(?:->|:)')


def fix_missing_return_types(file_path: Path) -> bool:
    """
    Fix missing return type annotations in Python files.
    
    Args:
        file_path: Path to the Python file to fix
    
    Returns:
        bool: True if any changes were made, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip files that are already properly typed or are __init__.py
    if file_path.name == '__init__.py' or '# type: ignore' in content:
        return False
    
    # Look for missing return types
    changed = False
    
    # TODO: Implement specific fixes based on patterns found
    
    return changed


def fix_vllm_session_command_call(file_path: Path) -> bool:
    """
    Fix the make_vllm_command calls in vllm_session.py to match the expected signature.
    
    Args:
        file_path: Path to vllm_session.py
    
    Returns:
        bool: True if any changes were made, False otherwise
    """
    if file_path.name != 'vllm_session.py':
        return False
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for the make_vllm_command call pattern
    pattern = re.compile(r'cmd\s*=\s*make_vllm_command\s*\(\s*model_alias\s*=\s*([^,\n]*),\s*mode\s*=\s*([^,\n]*),\s*yaml_path\s*=\s*([^,\n]*),\s*vllm_executable\s*=\s*([^,\n]*)\s*\)')
    
    match = pattern.search(content)
    if not match:
        return False
    
    # Extract the arguments
    model_alias = match.group(1)
    
    # Replace with properly structured call
    replacement = f"""
            # Create server config from YAML
            from src.config.vllm_config import VLLMConfig
            config = VLLMConfig.from_yaml(self.config_path)
            model_config = config.get_model_config({model_alias})
            server_config = config.server_config
            
            # Generate command to start model
            cmd = make_vllm_command(
                server_config=server_config,
                model_id=model_config.model_id
            )
    """
    
    # Replace the entire make_vllm_command call block
    new_content = pattern.sub(replacement, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    
    return False


def process_file(file_path: Union[str, Path]) -> bool:
    """
    Process a single Python file to fix typing issues.
    
    Args:
        file_path: Path to the Python file to process
    
    Returns:
        bool: True if any changes were made, False otherwise
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.suffix != '.py':
        return False
    
    changed = False
    
    # Apply specific fixes based on the file
    if path.name == 'vllm_session.py':
        changed |= fix_vllm_session_command_call(path)
    
    # Apply generic fixes
    changed |= fix_missing_return_types(path)
    
    return changed


def main() -> None:
    """
    Main entry point for the script.
    """
    # Process all Python files in the model_engine directory
    model_engine_dir = project_root / 'src' / 'model_engine'
    
    print(f"Scanning {model_engine_dir} for typing issues...")
    
    files_changed = 0
    for root, _, files in os.walk(model_engine_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if process_file(file_path):
                    files_changed += 1
                    print(f"Fixed typing issues in {file_path}")
    
    print(f"Fixed typing issues in {files_changed} files.")


if __name__ == '__main__':
    main()
