"""
Simple script to check adapter registration without requiring the actual Docling library.
"""

import sys
import os
from unittest.mock import MagicMock

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Mock Docling imports
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.datamodel'] = MagicMock()
sys.modules['docling.datamodel.base_models'] = MagicMock()
sys.modules['docling.datamodel.document'] = MagicMock()

# Now import the registry
from src.docproc.adapters.registry import get_supported_formats, get_adapter_class

def main():
    """Check the adapter registry."""
    print("Supported formats:", get_supported_formats())
    
    print("\nTesting adapter classes:")
    test_formats = ['pdf', 'html', 'json', 'yaml', 'xml', 'csv', 'text', 'python']
    for fmt in test_formats:
        try:
            adapter_class = get_adapter_class(fmt)
            print(f"- {fmt}: {adapter_class.__name__}")
        except ValueError as e:
            print(f"- {fmt}: Not registered ({e})")

if __name__ == "__main__":
    main()
