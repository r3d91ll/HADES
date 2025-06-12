#!/usr/bin/env python3
"""
Test runner for chunking components.

This script runs all tests for the chunking components and provides
a summary of the results.
"""

import sys
import pytest
from pathlib import Path

def main():
    """Run all chunking component tests."""
    
    # Get the test directory
    test_dir = Path(__file__).parent
    
    print("Running chunking component tests...")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            "name": "Core Chunker Tests",
            "path": test_dir / "core",
            "markers": "not slow"
        },
        {
            "name": "CPU Chunker Tests", 
            "path": test_dir / "chunkers" / "cpu",
            "markers": "not slow"
        },
        {
            "name": "Text Chunker Tests",
            "path": test_dir / "chunkers" / "text", 
            "markers": "not slow"
        },
        {
            "name": "Code Chunker Tests",
            "path": test_dir / "chunkers" / "code",
            "markers": "not slow"
        },
        {
            "name": "Chonky Chunker Tests (Optional)",
            "path": test_dir / "chunkers" / "chonky",
            "markers": "not slow",
            "optional": True
        }
    ]
    
    overall_success = True
    results = []
    
    for config in test_configs:
        print(f"\nRunning {config['name']}...")
        print("-" * 40)
        
        if not config["path"].exists():
            if config.get("optional", False):
                print(f"Skipping optional tests (path not found): {config['path']}")
                results.append((config["name"], "SKIPPED", "Path not found"))
                continue
            else:
                print(f"ERROR: Test path not found: {config['path']}")
                results.append((config["name"], "ERROR", "Path not found"))
                overall_success = False
                continue
        
        # Run tests for this component
        args = [
            str(config["path"]),
            "-v",
            "--tb=short",
            f"-m", config["markers"]
        ]
        
        try:
            result = pytest.main(args)
            
            if result == 0:
                print(f"✅ {config['name']} - ALL TESTS PASSED")
                results.append((config["name"], "PASSED", "All tests passed"))
            else:
                print(f"❌ {config['name']} - SOME TESTS FAILED")
                results.append((config["name"], "FAILED", f"Exit code: {result}"))
                overall_success = False
                
        except Exception as e:
            print(f"❌ {config['name']} - ERROR RUNNING TESTS: {e}")
            results.append((config["name"], "ERROR", str(e)))
            overall_success = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("CHUNKING COMPONENT TEST SUMMARY")
    print("=" * 50)
    
    for name, status, details in results:
        status_icon = "✅" if status == "PASSED" else "⚠️" if status == "SKIPPED" else "❌"
        print(f"{status_icon} {name}: {status}")
        if details and status != "PASSED":
            print(f"   {details}")
    
    print("\n" + "=" * 50)
    
    if overall_success:
        print("🎉 ALL CHUNKING COMPONENT TESTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("⚠️  SOME CHUNKING COMPONENT TESTS FAILED")
        print("\nTo debug failures, run individual test files:")
        for name, status, _ in results:
            if status == "FAILED":
                # Convert name to path suggestion
                component_name = name.lower().replace(" tests", "").replace(" ", "_")
                print(f"  pytest test/components/chunking/{component_name}/ -v")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())