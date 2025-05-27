"""
Run coverage tests on the sampler module.
"""

import os
import sys
import unittest
import coverage
from simple_sampler_test import TestNeighborSampler, TestRandomWalkSampler

def run_tests_with_coverage():
    """Run all tests with coverage report."""
    # Start coverage
    cov = coverage.Coverage(source=["src.isne.training.sampler"])
    cov.start()
    
    # Run tests
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNeighborSampler))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRandomWalkSampler))
    unittest.TextTestRunner().run(suite)
    
    # Stop coverage and report
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report
    cov.html_report(directory="coverage_html")
    print("\nHTML report generated in coverage_html directory")

if __name__ == "__main__":
    run_tests_with_coverage()
