#!/usr/bin/env python3
"""
Simple runner for adapter tests that bypasses the need for all dependencies.
This script directly imports and runs the test classes for adapters without going through pytest.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Directly import the test classes we need to run
try:
    from tests.docproc.adapters.test_python_adapter import TestPythonAdapter
except ImportError as e:
    print(f"Error importing TestPythonAdapter: {e}")
    sys.exit(1)

# Create a test suite with just our adapter tests
def create_test_suite():
    """Create a test suite with the adapter tests."""
    suite = unittest.TestSuite()
    
    # Add all test methods from TestPythonAdapter
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestPythonAdapter))
    
    return suite

if __name__ == "__main__":
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
