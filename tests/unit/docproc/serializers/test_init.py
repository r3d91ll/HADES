"""
Unit tests for the serializers module initialization.

Tests that the serializers module correctly exposes the expected functions.
"""

import importlib
import pytest

class TestSerializersInit:
    """Tests for the serializers module initialization."""

    def test_module_imports(self):
        """Test that the module correctly imports and exposes functions."""
        # Import the module
        from src.docproc.serializers import serialize_to_json, save_to_json_file
        
        # Check that the imported functions are callable
        assert callable(serialize_to_json)
        assert callable(save_to_json_file)

    def test_all_variable(self):
        """Test that __all__ contains the expected functions."""
        # Import the module
        import src.docproc.serializers
        
        # Check __all__ contents
        assert hasattr(src.docproc.serializers, "__all__")
        assert "serialize_to_json" in src.docproc.serializers.__all__
        assert "save_to_json_file" in src.docproc.serializers.__all__

    def test_import_reloading(self):
        """Test that the module can be reloaded without errors."""
        import src.docproc.serializers
        
        # Reload the module
        importlib.reload(src.docproc.serializers)
        
        # Check that functions are still accessible after reload
        assert callable(src.docproc.serializers.serialize_to_json)
        assert callable(src.docproc.serializers.save_to_json_file)

    def test_function_importing(self):
        """Test importing functions directly."""
        # Test various import patterns
        from src.docproc.serializers import serialize_to_json
        assert callable(serialize_to_json)
        
        from src.docproc.serializers import save_to_json_file as save_json
        assert callable(save_json)
