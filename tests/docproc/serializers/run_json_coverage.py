#!/usr/bin/env python
"""
Script to run JSON serializer tests with coverage measurements.
"""

import sys
import unittest
import coverage
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Setup coverage
cov = coverage.Coverage(include=["src/docproc/serializers/json_serializer.py"])
cov.start()

# Import the test module
from tests.docproc.serializers.test_json_serializer_unit import TestJsonSerializerUnit

# Run the tests
runner = unittest.TextTestRunner(verbosity=2)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestJsonSerializerUnit)
result = runner.run(suite)

# Stop coverage and generate report
cov.stop()
cov.save()

# Print coverage report
print("\nCoverage Report:")
total_coverage = cov.report()
print(f"Total coverage: {total_coverage:.2f}%")

# Check if it meets the 85% threshold
if total_coverage >= 85.0:
    print("✅ Coverage meets the 85% requirement")
else:
    print("❌ Coverage does not meet the 85% requirement")
    print(f"Current coverage: {total_coverage:.2f}%")
    print("Required coverage: 85.00%")

# Exit with appropriate code
sys.exit(0 if result.wasSuccessful() and total_coverage >= 85.0 else 1)
