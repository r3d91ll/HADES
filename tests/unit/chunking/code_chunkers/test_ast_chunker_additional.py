"""Additional unit tests for the AST-based code chunker.

This module contains additional comprehensive tests for the AST-based code chunker,
focusing on edge cases and complex Python structures to increase test coverage.
"""

from __future__ import annotations

import os
import ast
import pytest
import json
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

from src.chunking.code_chunkers.ast_chunker import (
    chunk_python_code,
    extract_chunk_content,
    estimate_tokens,
    create_chunk_id
)
from tests.unit.common_fixtures import (
    sample_code_document,
    create_expected_chunks,
    SAMPLE_CODE_CONTENT
)


class TestASTChunkerAdditional:
    """Additional test suite for the AST-based code chunker."""
    
    def test_chunk_python_code_with_complex_structure(self):
        """Test chunking with complex nested class and function structure."""
        # Create a complex Python document with nested classes and functions
        complex_code = """
class OuterClass:
    \"\"\"An outer class with nested elements.\"\"\"
    
    def __init__(self, value):
        self.value = value
    
    class InnerClass:
        \"\"\"A nested inner class.\"\"\"
        
        def __init__(self, inner_value):
            self.inner_value = inner_value
        
        def inner_method(self):
            return self.inner_value
    
    def outer_method(self):
        return self.InnerClass(self.value * 2)

def standalone_function():
    \"\"\"A standalone function outside of classes.\"\"\"
    return OuterClass(42)
"""
        
        document = {
            "id": "complex-doc",
            "content": complex_code,
            "path": "/path/to/complex.py",
            "type": "python"
        }
        
        # Chunk the complex code
        chunks = chunk_python_code(document)
        
        # Should produce chunks for: module, OuterClass, __init__, InnerClass, 
        # inner_method, outer_method, and standalone_function
        assert len(chunks) >= 6
        
        # Verify each chunk has correct structure
        for chunk in chunks:
            assert "id" in chunk
            assert "content" in chunk
            assert "symbol_type" in chunk
            assert "line_start" in chunk
            assert "line_end" in chunk
        
        # Check that we have both class and function chunks
        class_chunks = [c for c in chunks if c["symbol_type"] == "class"]
        function_chunks = [c for c in chunks if c["symbol_type"] == "function"]
        assert len(class_chunks) >= 2  # OuterClass and InnerClass
        assert len(function_chunks) >= 4  # Various methods and functions
        
        # Verify parent-child relationships
        outer_class = next((c for c in class_chunks if c["name"] == "OuterClass"), None)
        inner_class = next((c for c in class_chunks if c["name"] == "InnerClass"), None)
        assert outer_class is not None
        assert inner_class is not None
        
        # InnerClass should have OuterClass as parent
        assert inner_class.get("parent") == outer_class.get("id")
    
    def test_chunk_python_code_with_docstrings(self):
        """Test that docstrings are properly included in chunks."""
        # Code with module, class, and function docstrings
        code_with_docstrings = '''"""Module level docstring.

This is a detailed module docstring that should be included in the module chunk.
"""

class TestClass:
    """Class docstring.
    
    This is a detailed class docstring that should be included in the class chunk.
    """
    
    def test_method(self):
        """Method docstring.
        
        This is a detailed method docstring that should be included in the method chunk.
        """
        return "test"
'''
        
        document = {
            "id": "docstring-doc",
            "content": code_with_docstrings,
            "path": "/path/to/docstring.py",
            "type": "python"
        }
        
        # Chunk the code
        chunks = chunk_python_code(document)
        
        # Should have module, class, and method chunks
        assert len(chunks) >= 3
        
        # Check that each chunk includes its docstring
        module_chunk = next((c for c in chunks if c["symbol_type"] == "module"), None)
        class_chunk = next((c for c in chunks if c["symbol_type"] == "class"), None)
        method_chunk = next((c for c in chunks if c["symbol_type"] == "function"), None)
        
        assert module_chunk is not None
        assert class_chunk is not None
        assert method_chunk is not None
        
        assert 'Module level docstring' in module_chunk["content"]
        assert 'Class docstring' in class_chunk["content"]
        assert 'Method docstring' in method_chunk["content"]
    
    def test_chunk_python_code_with_imports(self):
        """Test that import statements are properly handled."""
        # Code with various import styles
        code_with_imports = '''import os
import sys, json
from datetime import datetime
from typing import (
    Dict, 
    List, 
    Optional
)
import numpy as np
from pathlib import Path

# Function using imports
def process_data():
    now = datetime.now()
    data = json.loads('{"key": "value"}')
    path = Path(os.path.join("/tmp", "data.json"))
    return data, now, path
'''
        
        document = {
            "id": "imports-doc",
            "content": code_with_imports,
            "path": "/path/to/imports.py",
            "type": "python"
        }
        
        # Chunk the code
        chunks = chunk_python_code(document)
        
        # Should have at least module and function chunks
        assert len(chunks) >= 2
        
        # Module chunk should contain all imports
        module_chunk = next((c for c in chunks if c["symbol_type"] == "module"), None)
        assert module_chunk is not None
        
        # Check that imports are included
        assert 'import os' in module_chunk["content"]
        assert 'import sys, json' in module_chunk["content"]
        assert 'from datetime import datetime' in module_chunk["content"]
        assert 'from typing import' in module_chunk["content"]
        
        # Function chunk should have its body
        function_chunk = next((c for c in chunks if c["symbol_type"] == "function"), None)
        assert function_chunk is not None
        assert 'def process_data' in function_chunk["content"]
        assert 'now = datetime.now()' in function_chunk["content"]
    
    def test_chunking_with_syntax_errors_in_functions(self):
        """Test handling of syntax errors within functions."""
        # Code with syntax error in a function
        code_with_error = '''
def valid_function():
    return "This is valid"

def invalid_function():
    return "This is incomplete
    
def another_valid_function():
    return "This should still be processed"
'''
        
        document = {
            "id": "error-doc",
            "content": code_with_error,
            "path": "/path/to/error.py",
            "type": "python"
        }
        
        # Should still process the file, even with syntax errors
        chunks = chunk_python_code(document)
        
        # Should have module and valid function chunks
        assert len(chunks) >= 2
        
        # Check that valid functions are still chunked
        valid_chunks = [c for c in chunks if c["name"] == "valid_function" or c["name"] == "another_valid_function"]
        assert len(valid_chunks) >= 1  # Should get at least one valid function
        
        # Check that error content is preserved in some form
        content_joined = " ".join([c["content"] for c in chunks])
        assert "invalid_function" in content_joined
    
    def test_extract_chunk_content_edge_cases(self):
        """Test extract_chunk_content with edge cases."""
        # Test with line numbers out of bounds
        source = "line1\nline2\nline3"
        
        # Test with line numbers greater than source
        assert extract_chunk_content(source, 4, 5) == ""
        
        # Test with zero line number (1-indexed expected)
        assert extract_chunk_content(source, 0, 1) == "line1"
        
        # Test with end before start
        assert extract_chunk_content(source, 3, 2) == "line3"  # Should handle this gracefully
        
        # Test with negative line numbers
        assert extract_chunk_content(source, -1, 2) == "line1\nline2"  # Should handle this gracefully
    
    def test_create_chunk_id_with_special_characters(self):
        """Test create_chunk_id with special characters."""
        # Test with special characters in name and path
        id1 = create_chunk_id("/path/with spaces/file.py", "class", "Class With Spaces", 1, 10)
        id2 = create_chunk_id("/path/with-hyphens/file.py", "function", "function_with_underscore", 1, 10)
        id3 = create_chunk_id("/path/with/symbols/!@#.py", "module", "module!@#", 1, 10)
        
        # IDs should be valid strings without causing errors
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert isinstance(id3, str)
        
        # Different inputs should give different IDs
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3
    
    def test_very_large_document(self):
        """Test chunking a very large document."""
        # Create a large document with many functions
        large_code = "\n".join([f"def function_{i}():\n    return {i}\n" for i in range(100)])
        
        document = {
            "id": "large-doc",
            "content": large_code,
            "path": "/path/to/large.py",
            "type": "python"
        }
        
        # Chunk the large document
        chunks = chunk_python_code(document)
        
        # Should have many chunks (module + 100 functions)
        assert len(chunks) > 50
        
        # Verify function names are preserved
        function_names = [c.get("name") for c in chunks if c.get("symbol_type") == "function"]
        assert len(function_names) == 100
        assert "function_0" in function_names
        assert "function_99" in function_names
    
    def test_line_based_fallback(self):
        """Test the line-based fallback for large functions."""
        # Create a document with a very large function that should trigger line-based fallback
        large_function = "def large_function():\n" + "    x = 1\n" * 1000
        
        document = {
            "id": "fallback-doc",
            "content": large_function,
            "path": "/path/to/fallback.py",
            "type": "python"
        }
        
        # Use a small max_tokens to force line-based fallback
        chunks = chunk_python_code(document, max_tokens=100)
        
        # Should have broken the function into multiple chunks
        assert len(chunks) > 2
        
        # The first chunk should be the function header
        first_chunk = chunks[0]
        assert "def large_function()" in first_chunk["content"]
        
        # Subsequent chunks should contain the function body
        subsequent_chunks = chunks[1:]
        for chunk in subsequent_chunks:
            assert "x = 1" in chunk["content"]
            
        # Check that line ranges are ascending and non-overlapping
        for i in range(1, len(chunks)):
            assert chunks[i]["line_start"] > chunks[i-1]["line_start"]
            assert chunks[i]["line_start"] > chunks[i-1]["line_end"]
