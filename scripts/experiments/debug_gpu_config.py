#!/usr/bin/env python
"""
GPU Configuration Diagnostics Utility

This script helps diagnose GPU configuration and mapping issues, particularly when
working with CUDA_VISIBLE_DEVICES environment variable settings. It provides detailed 
information about the available GPU devices, how they're mapped, and how PyTorch 
sees them.

Usage:
    python debug_gpu_config.py [--verbose]

Options:
    --verbose   Show more detailed diagnostics including memory usage

Example:
    CUDA_VISIBLE_DEVICES=1,0 python debug_gpu_config.py
    # Shows how devices 1 and 0 are mapped to PyTorch's cuda:0 and cuda:1
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add the project root to the path so we can import the project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device_utils import get_device_info, get_gpu_diagnostics


def format_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """Format a dictionary for prettier console output"""
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            result.append(" " * indent + f"{key}:")
            result.append(format_dict(value, indent + 2))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            result.append(" " * indent + f"{key}:")
            for i, item in enumerate(value):
                result.append(" " * (indent + 2) + f"[{i}]:")
                result.append(format_dict(item, indent + 4))
        else:
            result.append(" " * indent + f"{key}: {value}")
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(description="GPU Configuration Diagnostics Utility")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed diagnostics")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    # Print basic environment info
    print("\n=== GPU CONFIGURATION DIAGNOSTICS ===\n")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Get basic device info
    device_info = get_device_info()
    
    print(f"\n=== DEVICE SUMMARY ===\n")
    print(f"GPU Available: {device_info['gpu_available']}")
    print(f"Device Type: {device_info['device_type']}")
    print(f"Default Device: {device_info['device_name']}")
    
    if device_info['gpu_available']:
        print(f"\nGPU Count: {device_info['gpu_details']['gpu_count']}")
        print(f"Current Device: {device_info['gpu_details']['current_device']}")
        print(f"Device Name: {device_info['gpu_details']['device_name']}")
    
    # Get detailed diagnostics if verbose
    if args.verbose or args.json:
        diagnostics = get_gpu_diagnostics()
        
        if args.json:
            print("\n=== DETAILED DIAGNOSTICS (JSON) ===\n")
            print(json.dumps(diagnostics, indent=2))
        else:
            print("\n=== DETAILED DIAGNOSTICS ===\n")
            print(format_dict(diagnostics))
            
            if "device_mapping" in diagnostics:
                print("\n=== DEVICE MAPPING ===\n")
                print(diagnostics["mapping_explanation"])
                print("\nLogical to Physical GPU Mapping:")
                for mapping in diagnostics["device_mapping"]:
                    print(f"  {mapping['logical_device']} → {mapping['physical_device']}")
                    
            if diagnostics.get("error"):
                print(f"\nERROR: {diagnostics['error']}")
    
    print("\n=== PYTORCH CODE EXAMPLE ===\n")
    print("To select a specific device in your code:")
    print("  import torch")
    print("  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')")
    print("  model = model.to(device)")
    
    print("\n=== ENVIRONMENT VARIABLE USAGE ===\n")
    print("To select specific GPUs before running your script:")
    print("  CUDA_VISIBLE_DEVICES=0,1 python your_script.py  # Use GPUs 0 and 1")
    print("  CUDA_VISIBLE_DEVICES=1 python your_script.py    # Use only GPU 1")
    print("  CUDA_VISIBLE_DEVICES= python your_script.py     # Force CPU mode")
    

if __name__ == "__main__":
    main()
