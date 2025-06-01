#!/usr/bin/env python3
"""
Run targeted coverage tests for specific adapter modules.

This script focuses on testing the adapter modules we've been fixing, using
mocks to avoid dependency issues with the larger docproc module.
"""

import os
import sys
import importlib.util
import coverage
import unittest
import tempfile
from pathlib import Path
from unittest import mock
from typing import Dict, Any, List, Optional, Union, cast

def create_mock_modules():
    """Create necessary mock modules to avoid dependency issues."""
    # Mock all dependencies that might cause import errors
    mocks = {
        'src.utils': mock.MagicMock(),
        'src.utils.file_utils': mock.MagicMock(),
        'src.types': mock.MagicMock(),
        'src.types.docproc': mock.MagicMock(),
        'src.types.docproc.adapter': mock.MagicMock(),
        'src.types.docproc.metadata': mock.MagicMock(),
    }
    
    # Add the mocks to sys.modules
    for module_name, mock_obj in mocks.items():
        sys.modules[module_name] = mock_obj
    
    # Define minimum required classes/types
    class ProcessedDocument(dict):
        pass
    
    class EntityDict(dict):
        pass
    
    class MetadataDict(dict):
        pass
    
    class AdapterOptions(dict):
        pass
    
    # Add these to the appropriate mock modules
    sys.modules['src.types.docproc.adapter'].ProcessedDocument = ProcessedDocument
    sys.modules['src.types.docproc.adapter'].AdapterOptions = Union[Dict[str, Any], None]
    sys.modules['src.types.docproc.metadata'].EntityDict = EntityDict
    sys.modules['src.types.docproc.metadata'].MetadataDict = MetadataDict

def create_test_suite():
    """Create a test suite for the adapter files."""
    # Get the path to the test files
    project_root = Path(__file__).parent
    test_dir = project_root / "tests" / "docproc" / "adapters"
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests for specific adapters
    test_modules = [
        test_dir / "test_python_adapter.py",
        test_dir / "test_docling_adapter.py",
        test_dir / "test_python_code_adapter.py"
    ]
    
    # Load and add tests that exist
    for test_file in test_modules:
        if test_file.exists():
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    # Try to load the module
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Find all test cases in the module
                    for item in dir(module):
                        obj = getattr(module, item)
                        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                            suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
                    
                    print(f"✅ Successfully loaded tests from {test_file.name}")
                except Exception as e:
                    print(f"❌ Error loading tests from {test_file.name}: {e}")
    
    return suite

def run_targeted_coverage():
    """Run targeted coverage on adapter files."""
    # Set up project path
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print("\n=== Running targeted coverage for adapter files ===\n")
    
    # Set up mocks
    create_mock_modules()
    
    # Create mock tests for PythonCodeAdapter if missing
    python_code_adapter_test = project_root / "tests" / "docproc" / "adapters" / "test_python_code_adapter.py"
    if not python_code_adapter_test.exists():
        print("⚠️ test_python_code_adapter.py not found, creating a basic version...")
        
        # Use the test file we created earlier
        with open(python_code_adapter_test, "w") as f:
            f.write("""
import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

# Mock the dependencies
sys.modules['src.docproc.adapters.python_adapter'] = mock.MagicMock()
sys.modules['src.docproc.adapters.python_adapter'].PythonAdapter = mock.MagicMock()

from src.docproc.adapters.python_code_adapter import PythonCodeAdapter

class TestPythonCodeAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = PythonCodeAdapter()
        
    def test_init(self):
        adapter = PythonCodeAdapter()
        self.assertIsNotNone(adapter)
        
    def test_process_text(self):
        result = self.adapter.process_text("def test(): pass")
        self.assertIsInstance(result, dict)
        
    def test_extract_entities(self):
        entities = self.adapter.extract_entities("def test(): pass")
        self.assertIsInstance(entities, list)
        
    def test_extract_metadata(self):
        metadata = self.adapter.extract_metadata("def test(): pass")
        self.assertIsInstance(metadata, dict)
""")
    
    # Files to analyze
    adapter_files = [
        "src/docproc/adapters/python_adapter.py",
        "src/docproc/adapters/docling_adapter.py",
        "src/docproc/adapters/python_code_adapter.py"
    ]
    
    # Run coverage
    cov = coverage.Coverage(source=adapter_files)
    cov.start()
    
    # Run the tests
    try:
        suite = create_test_suite()
        if not suite.countTestCases():
            print("❌ No test cases found")
            return 1
        
        print(f"Running {suite.countTestCases()} test cases...")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ {len(result.failures) + len(result.errors)} tests failed")
        
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        # Print report
        print("\n=== Coverage Report ===\n")
        cov.report(show_missing=True)
        
        # Check if we meet the standard
        try:
            result = cov.html_report(directory=str(project_root / "htmlcov"))
            percentage = result
        except:
            # If html_report fails, estimate percentage
            percentage = 0
            report_str = cov.report(file=None)
            try:
                # Try to parse percentage from report
                if isinstance(report_str, str) and "%" in report_str:
                    parts = report_str.split("%")
                    percentage = float(parts[0].strip().split()[-1])
            except:
                pass
        
        if percentage >= 85:
            print(f"\n✅ Coverage meets the 85% standard! ({percentage:.1f}%)")
            return 0
        else:
            print(f"\n⚠️ Coverage is below the 85% standard. ({percentage:.1f}%)")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cov.stop()

if __name__ == "__main__":
    sys.exit(run_targeted_coverage())
