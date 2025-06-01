#!/usr/bin/env python3
"""
Direct verification script for PythonCodeAdapter typing fixes with mocked dependencies.

This script creates a mock version of the necessary dependencies to isolate and test
the PythonCodeAdapter, verifying that our typing fixes don't break functionality.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast
from unittest import mock

# First create mocks for the dependencies
sys.modules['src.docproc.adapters.base'] = mock.MagicMock()
sys.modules['src.docproc.adapters.registry'] = mock.MagicMock()
sys.modules['src.docproc.utils.code_analysis'] = mock.MagicMock()
sys.modules['src.types.docproc.adapter'] = mock.MagicMock()
sys.modules['src.types.docproc.metadata'] = mock.MagicMock()

# Create minimal mock classes/types for the imports
class MockBaseAdapter:
    def __init__(self):
        pass

class MockEntityDict(dict):
    pass

class MockMetadataDict(dict):
    pass

class MockProcessedDocument(dict):
    pass

# Patch the imports in the module we're testing
sys.modules['src.docproc.adapters.base'].BaseAdapter = MockBaseAdapter
sys.modules['src.types.docproc.metadata'].EntityDict = MockEntityDict
sys.modules['src.types.docproc.metadata'].MetadataDict = MockMetadataDict
sys.modules['src.types.docproc.adapter'].ProcessedDocument = MockProcessedDocument
sys.modules['src.types.docproc.adapter'].AdapterOptions = Union[Dict[str, Any], str, None]

# Create a mock for the PythonAdapter that will be imported by python_code_adapter
class MockPythonAdapter:
    def __init__(self, create_symbol_table=False, options=None):
        self.create_symbol_table = create_symbol_table
        self.options = options or {}
    
    def process_text(self, text):
        # Return mock analysis data that matches the structure expected by PythonCodeAdapter
        return {
            "code_analysis": {
                "module": {
                    "name": "mock_module",
                    "docstring": "Mock module docstring",
                    "imports": ["os", "sys"],
                    "line_count": 42
                },
                "functions": {
                    "hello": {
                        "docstring": "Mock function docstring",
                        "args": ["name"],
                        "returns": "str",
                        "line_start": 10,
                        "line_end": 15,
                        "calls": ["print"]
                    }
                },
                "classes": {
                    "TestClass": {
                        "docstring": "Mock class docstring",
                        "bases": ["object"],
                        "methods": ["__init__", "greet"],
                        "line_start": 20,
                        "line_end": 30
                    }
                }
            }
        }

# Add the mock PythonAdapter to the sys.modules
mock_python_adapter = mock.MagicMock()
mock_python_adapter.PythonAdapter = MockPythonAdapter
sys.modules['src.docproc.adapters.python_adapter'] = mock_python_adapter

# Now that we've set up all the mocks, import the module we want to test
try:
    # Use a clean import to make sure we get our module with the mocks in place
    import importlib
    import importlib.util
    
    # Add src directory to path
    src_dir = Path(__file__).parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Create a custom loader for the file
    adapter_file = src_dir / "docproc" / "adapters" / "python_code_adapter.py"
    module_name = "custom_python_code_adapter"
    spec = importlib.util.spec_from_file_location(module_name, adapter_file)
    python_code_adapter = importlib.util.module_from_spec(spec)
    
    # Add mocks for potential imports in the module
    sys.modules[module_name] = python_code_adapter
    sys.modules['docproc'] = mock.MagicMock()
    sys.modules['docproc.adapters'] = mock.MagicMock()
    sys.modules['docproc.adapters.base'] = mock.MagicMock()
    
    # Execute the module
    spec.loader.exec_module(python_code_adapter)
    
    # Get the PythonCodeAdapter class
    PythonCodeAdapter = python_code_adapter.PythonCodeAdapter
    
    print("✅ Successfully imported PythonCodeAdapter with mocked dependencies")
except ImportError as e:
    print(f"❌ Error importing PythonCodeAdapter: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

def test_adapter_init():
    """Test adapter initialization"""
    try:
        adapter = PythonCodeAdapter()
        print("✅ Successfully initialized PythonCodeAdapter")
        
        # Check if the PythonAdapter was initialized correctly
        assert hasattr(adapter, 'python_adapter'), "Missing python_adapter attribute"
        print("✅ Correctly initialized underlying PythonAdapter")
        
        # Test with custom options
        custom_options = {"test_option": "value"}
        adapter_with_options = PythonCodeAdapter(custom_options)
        assert adapter_with_options.options == custom_options, "Options not stored correctly"
        print("✅ Custom options correctly stored")
        
        return True
    except Exception as e:
        print(f"❌ Error in test_adapter_init: {e}")
        return False

def test_process_text():
    """Test process_text method"""
    try:
        adapter = PythonCodeAdapter()
        test_code = 'print("Hello, world!")'
        
        result = adapter.process_text(test_code)
        
        # Check the result structure
        assert isinstance(result, dict), "Result should be a dict"
        assert "id" in result, "Missing id in result"
        assert "source" in result, "Missing source in result"
        assert "content" in result, "Missing content in result"
        assert "format" in result, "Missing format in result"
        assert "metadata" in result, "Missing metadata in result"
        assert "entities" in result, "Missing entities in result"
        
        # Check metadata
        assert "code_analysis" in result["metadata"], "Missing code_analysis in metadata"
        
        print("✅ process_text method works correctly")
        return True
    except Exception as e:
        print(f"❌ Error in test_process_text: {e}")
        return False

def test_extract_entities():
    """Test extract_entities method"""
    try:
        adapter = PythonCodeAdapter()
        test_code = 'print("Hello, world!")'
        
        # Test with string input
        entities = adapter.extract_entities(test_code)
        assert isinstance(entities, list), "Entities should be a list"
        
        # Check for expected entity types
        entity_types = [e["type"] for e in entities]
        assert "module" in entity_types, "Missing module entity"
        assert "function" in entity_types, "Missing function entity"
        assert "class" in entity_types, "Missing class entity"
        
        # Test with dict input
        processed = adapter.process_text(test_code)
        entities_from_dict = adapter.extract_entities(processed)
        assert isinstance(entities_from_dict, list), "Entities from dict should be a list"
        
        print("✅ extract_entities method works correctly")
        return True
    except Exception as e:
        print(f"❌ Error in test_extract_entities: {e}")
        return False

def test_extract_metadata():
    """Test extract_metadata method"""
    try:
        adapter = PythonCodeAdapter()
        test_code = 'print("Hello, world!")'
        
        # Test with string input
        metadata = adapter.extract_metadata(test_code)
        assert isinstance(metadata, dict), "Metadata should be a dict"
        
        # Check for expected metadata fields
        assert "language" in metadata, "Missing language in metadata"
        assert "document_type" in metadata, "Missing document_type in metadata"
        assert "function_count" in metadata, "Missing function_count in metadata"
        assert "class_count" in metadata, "Missing class_count in metadata"
        
        # Test with dict input
        processed = adapter.process_text(test_code)
        metadata_from_dict = adapter.extract_metadata(processed)
        assert isinstance(metadata_from_dict, dict), "Metadata from dict should be a dict"
        
        print("✅ extract_metadata method works correctly")
        return True
    except Exception as e:
        print(f"❌ Error in test_extract_metadata: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("\n=== PythonCodeAdapter Typing Verification Tests ===\n")
    
    tests = [
        ("Adapter initialization", test_adapter_init),
        ("Process text method", test_process_text),
        ("Extract entities method", test_extract_entities),
        ("Extract metadata method", test_extract_metadata)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\nTesting: {name}")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== Test Results: {passed} passed, {failed} failed ===")
    
    # Calculate coverage
    total_methods = 5  # __init__, process, process_text, extract_entities, extract_metadata
    tested_methods = 4  # Excluding process since we can't easily mock file operations
    coverage = (tested_methods / total_methods) * 100
    
    print(f"Estimated test coverage: {coverage:.1f}% ({tested_methods}/{total_methods} methods)")
    
    if coverage >= 85:
        print("✅ MEETS 85% STANDARD")
    else:
        print("⚠️ BELOW 85% STANDARD")
    
    if failed == 0:
        print("\n✅ All PythonCodeAdapter tests passed!")
        print("✅ Typing fixes did not break functionality")
        return 0
    else:
        print(f"\n❌ {failed} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
