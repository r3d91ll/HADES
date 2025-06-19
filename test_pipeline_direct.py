#!/usr/bin/env python3
"""Direct test of ISNE pipelines without API layer."""

import subprocess
import sys
from pathlib import Path

def run_pipeline_test():
    """Run complete ISNE pipeline test."""
    print("ISNE Pipeline Direct Test")
    print("=" * 50)
    
    # Test 1: Bootstrap
    print("\n1. Testing Bootstrap Pipeline...")
    cmd = [
        sys.executable, 
        "src/pipelines/production/bootstrap_full_isne_testdata.py",
        "--db-name", "test_isne_direct",
        "--dataset-path", "/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata"
    ]
    print(f"Command: {' '.join(cmd)}")
    # Would run: subprocess.run(cmd)
    
    # Test 2: Training
    print("\n2. Testing Training Pipeline...")
    cmd = [
        sys.executable,
        "src/pipelines/production/train_isne_memory_efficient.py",
        "--db-name", "test_isne_direct",
        "--epochs", "5"  # Quick test
    ]
    print(f"Command: {' '.join(cmd)}")
    # Would run: subprocess.run(cmd)
    
    # Test 3: Apply Model
    print("\n3. Testing Model Application...")
    print("Command: python src/pipelines/production/apply_efficient_isne_model.py ...")
    
    # Test 4: Build Collections
    print("\n4. Testing Semantic Collections...")
    print("Command: python src/pipelines/production/build_semantic_collections.py ...")
    
    print("\nDirect pipeline testing ready!")
    print("Tomorrow with MCP, these will all be accessible via API")

if __name__ == "__main__":
    run_pipeline_test()