#!/usr/bin/env python3
"""
Direct verification script for the Python adapter functionality.
This script tests the core functionality of python_adapter.py after type fixes.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, cast, Optional, Union

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the adapters we modified
try:
    from src.docproc.adapters.python_adapter import PythonAdapter
    from src.types.docproc.adapter import ProcessedDocument, AdapterOptions
    from src.types.docproc.metadata import MetadataDict
    print("✅ Successfully imported PythonAdapter and required types")
except ImportError as e:
    print(f"❌ Error importing PythonAdapter: {e}")
    sys.exit(1)

def verify_python_adapter():
    """Verify the basic functionality of PythonAdapter after type fixes."""
    adapter = PythonAdapter()
    
    # Simple Python code for testing
    test_code = """
\"\"\"Module docstring for test.\"\"\"

def hello():
    \"\"\"Say hello.\"\"\"
    print("Hello, world!")

class TestClass:
    \"\"\"A test class.\"\"\"
    def method(self):
        \"\"\"A test method.\"\"\"
        pass
"""
    
    # Test file path for metadata extraction
    test_file = Path(__file__).parent / "verify_adapter_fixes.py"
    
    try:
        # Test process_text function (this was modified for type safety)
        processed = adapter.process_text(test_code)
        print(f"✅ process_text returned {type(processed)}")
        
        # Verify the processed document structure
        if "content" in processed and "entities" in processed:
            print(f"✅ Processed document has expected structure")
        else:
            print(f"❌ Processed document missing expected keys")
            return False
        
        # Test extract_metadata (this was modified to return PythonMetadataDict)
        metadata = adapter.extract_metadata({"content": test_code, "file_path": str(test_file)})
        print(f"✅ extract_metadata returned {type(metadata)}")
        
        # Verify the metadata structure with file_path
        if "file_path" in metadata:
            print(f"✅ Metadata contains file_path: {metadata['file_path']}")
        else:
            print(f"❌ Metadata missing file_path")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing PythonAdapter: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Verifying Python Adapter functionality after type fixes...")
    success = verify_python_adapter()
    print(f"{'🎉 All tests passed!' if success else '❌ Tests failed!'}")
    sys.exit(0 if success else 1)
