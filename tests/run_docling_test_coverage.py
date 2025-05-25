#!/usr/bin/env python
"""
Script to run test coverage for the docling adapter with mocked imports
to avoid pydantic validation errors in docling library.
"""

import sys
import os
import pytest
from unittest.mock import patch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock problematic docling imports before they're loaded
sys.modules["docling.document_converter"] = type("MockModule", (), {"DocumentConverter": type("MockClass", (), {})})
sys.modules["docling.datamodel.document"] = type("MockModule", (), {})
sys.modules["docling.datamodel.settings"] = type("MockModule", (), {})
sys.modules["docling.backend.asciidoc_backend"] = type("MockModule", (), {})

# Run pytest with coverage
if __name__ == "__main__":
    pytest.main([
        "tests/unit/docproc/adapters/test_docling_adapter.py",
        "--cov=src.docproc.adapters.docling_adapter",
        "--cov-report=term",
        "-v"
    ])
