#!/usr/bin/env python3
"""
Test runner for Graph Enhancement Components.

This script runs all unit tests for the graph enhancement components with coverage reporting.
Follows the Standard Protocol requirements for comprehensive testing with >85% coverage.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    test_dir = Path(__file__).parent
    
    # Set PYTHONPATH to include src directory
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root / 'src') + ':' + env.get('PYTHONPATH', '')
    
    print("=" * 80)
    print("GRAPH ENHANCEMENT COMPONENT TESTS")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Test directory: {test_dir}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print()
    
    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=src.components.graph_enhancement",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html/graph_enhancement",
        "--cov-fail-under=85",  # Require 85% coverage as per Standard Protocol
        "-x"  # Stop on first failure for faster feedback
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_individual_component_tests():
    """Run tests for individual components."""
    
    project_root = Path(__file__).parent.parent.parent.parent
    test_dir = Path(__file__).parent / "isne"
    
    # Set PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root / 'src') + ':' + env.get('PYTHONPATH', '')
    
    components = [
        ("Core ISNE", "test_core_processor.py"),
        ("Training ISNE", "test_training_processor.py"),
        ("Inference ISNE", "test_inference_processor.py"),
        ("Inductive ISNE", "test_inductive_processor.py"),
        ("None/Passthrough ISNE", "test_none_processor.py")
    ]
    
    results = {}
    
    for component_name, test_file in components:
        print(f"\n{'=' * 60}")
        print(f"TESTING {component_name.upper()}")
        print(f"{'=' * 60}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_dir / test_file),
            "-v",
            "--tb=short"
        ]
        
        try:
            result = subprocess.run(cmd, env=env, cwd=project_root)
            results[component_name] = result.returncode == 0
            
            if result.returncode == 0:
                print(f"✅ {component_name} tests PASSED")
            else:
                print(f"❌ {component_name} tests FAILED")
                
        except Exception as e:
            print(f"❌ Error testing {component_name}: {e}")
            results[component_name] = False
    
    return results

def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{component:<25} {status}")
    
    print("-" * 40)
    print(f"Total: {passed}/{total} components passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Graph enhancement components are ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} component(s) failed. Please fix before proceeding.")
        return False

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run graph enhancement component tests")
    parser.add_argument("--coverage", "-c", action="store_true", 
                       help="Run with coverage reporting")
    parser.add_argument("--individual", "-i", action="store_true",
                       help="Run individual component tests")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all test modes")
    
    args = parser.parse_args()
    
    success = True
    
    if args.coverage or args.all:
        print("Running tests with coverage...")
        success = run_tests_with_coverage()
        
    if args.individual or args.all:
        print("\nRunning individual component tests...")
        results = run_individual_component_tests()
        individual_success = print_summary(results)
        success = success and individual_success
    
    if not (args.coverage or args.individual or args.all):
        # Default: run with coverage
        success = run_tests_with_coverage()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()