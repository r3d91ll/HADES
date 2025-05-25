"""
Test utilities for the docproc module.

These utilities help create isolated tests that don't rely on external dependencies.
"""

import sys
from unittest.mock import MagicMock
from typing import Any, Dict, Optional, Callable

# Create mock for docling module to avoid import errors
class MockDoclingModule:
    """Mock implementation of the docling module."""
    
    class DocumentConverter:
        """Mock implementation of DocumentConverter."""
        def convert(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            """Mock convert method."""
            class MockDoc:
                def __init__(self) -> None:
                    self.metadata = {}

                def export_to_markdown(self) -> str:  # type: ignore
                    return "# Mock Heading\n\nMock paragraph."

                def export_to_text(self) -> str:  # type: ignore
                    return "Mock Heading\n\nMock paragraph."

                @property
                def pages(self):  # type: ignore
                    return []

            return {"document": MockDoc()}
    
    class InputFormat:
        """Mock implementation of InputFormat."""
        PDF = "pdf"
        HTML = "html"
        TEXT = "text"


# Function to patch modules for testing
def patch_modules() -> None:
    """
    Patch modules with mocks to avoid external dependencies.
    
    This should be called before importing any docproc modules in tests.
    """
    # Create mock for docling module
    sys.modules['docling'] = MagicMock()
    sys.modules['docling.document_converter'] = MagicMock()
    sys.modules['docling.document_converter'].DocumentConverter = MockDoclingModule.DocumentConverter
    sys.modules['docling.document_converter'].InputFormat = MockDoclingModule.InputFormat
