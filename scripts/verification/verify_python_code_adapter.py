#!/usr/bin/env python3
"""
Direct verification script for the PythonCodeAdapter after type fixes.

This script directly tests the functionality of the PythonCodeAdapter class
to ensure our typing fixes haven't broken core functionality.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast

# Add src to path to allow importing directly
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import the adapter we want to test
    from src.docproc.adapters.python_code_adapter import PythonCodeAdapter
    from src.types.docproc.adapter import ProcessedDocument
    from src.types.docproc.metadata import MetadataDict, EntityDict
    print("✅ Successfully imported PythonCodeAdapter")
except ImportError as e:
    print(f"❌ Error importing PythonCodeAdapter: {e}")
    sys.exit(1)

def verify_python_code_adapter() -> None:
    """Verify the basic functionality of PythonCodeAdapter after type fixes."""
    adapter = PythonCodeAdapter()
    print("✅ Successfully created PythonCodeAdapter instance")
    
    # Simple Python code for testing
    test_code = """
\"\"\"Module docstring for test.\"\"\"

def hello():
    \"\"\"Say hello.\"\"\"
    print("Hello, world!")

class TestClass:
    \"\"\"A test class.\"\"\"
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
"""
    
    # Test process_text
    try:
        result = adapter.process_text(test_code)
        print("✅ Successfully processed text")
        print(f"  - Result type: {type(result)}")
        print(f"  - Document ID: {result.get('id', 'Missing')}")
        print(f"  - Format: {result.get('format', 'Missing')}")
    except Exception as e:
        print(f"❌ Error in process_text: {e}")
        sys.exit(1)
    
    # Test extract_entities
    try:
        entities = adapter.extract_entities(test_code)
        print("✅ Successfully extracted entities")
        print(f"  - Entities found: {len(entities)}")
        for entity in entities:
            print(f"  - Entity: {entity.get('type', 'Unknown')} - {entity.get('name', 'Unnamed')}")
    except Exception as e:
        print(f"❌ Error in extract_entities: {e}")
        sys.exit(1)
    
    # Test extract_metadata
    try:
        metadata = adapter.extract_metadata(test_code)
        print("✅ Successfully extracted metadata")
        print(f"  - Language: {metadata.get('language', 'Missing')}")
        print(f"  - Function count: {metadata.get('function_count', 'Missing')}")
        print(f"  - Class count: {metadata.get('class_count', 'Missing')}")
    except Exception as e:
        print(f"❌ Error in extract_metadata: {e}")
        sys.exit(1)
    
    # Create a temporary file for testing process()
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as tmp:
            tmp.write(test_code)
            tmp.flush()
            
            result = adapter.process(tmp.name)
            print("✅ Successfully processed file")
            print(f"  - Source: {result.get('source', 'Missing')}")
    except Exception as e:
        print(f"❌ Error in process: {e}")
        sys.exit(1)
    
    print("\n✅ All PythonCodeAdapter tests passed!")
    print("✅ Type fixes did not break functionality")

if __name__ == "__main__":
    verify_python_code_adapter()
