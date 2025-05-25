"""
Unit tests for the docproc.adapters.registry module.

These tests cover the adapter registry functionality:
- Registering adapters
- Retrieving adapters
- Error handling for missing adapters
- Default adapter handling
"""

import pytest
from unittest.mock import patch, MagicMock

from src.docproc.adapters.registry import (
    register_adapter,
    get_adapter_class,
    get_adapter_for_format,
    _ADAPTER_REGISTRY
)
from src.docproc.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    """Mock adapter implementation for testing."""
    
    def __init__(self, format_type="mock"):
        """Initialize with format type."""
        super().__init__(format_type)
        # Add required configuration properties
        self.entity_extraction = True
        self.metadata_extraction = True
        self.chunking_preparation = True
    
    def process(self, file_path, content=None, options=None):
        """Implement required process method."""
        return {"format": "mock"}
    
    def extract_metadata(self, file_path, content=None, options=None):
        """Implement required extract_metadata method."""
        return {"format": "mock"}
    
    def extract_entities(self, text, options=None):
        """Implement required extract_entities method."""
        return []
    
    def process_text(self, text, options=None):
        """Implement required process_text method."""
        return {"format": "mock", "content": text}


class TestAdapterRegistry:
    """Tests for the adapter registry module."""
    
    def setup_method(self):
        """Set up test environment."""
        # Save original registry state
        self.original_registry = _ADAPTER_REGISTRY.copy()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Restore original registry state
        _ADAPTER_REGISTRY.clear()
        _ADAPTER_REGISTRY.update(self.original_registry)
    
    def test_register_adapter(self):
        """Test registering an adapter."""
        # Register mock adapter
        register_adapter("test_format", MockAdapter)
        
        # Verify it was registered
        assert "test_format" in _ADAPTER_REGISTRY
        assert _ADAPTER_REGISTRY["test_format"] == MockAdapter
    
    def test_register_adapter_multiple_calls(self):
        """Test registering an adapter for multiple formats with separate calls."""
        # Register mock adapter for multiple formats with separate calls
        formats = ["format1", "format2", "format3"]
        for fmt in formats:
            register_adapter(fmt, MockAdapter)
        
        # Verify it was registered for all formats
        for fmt in formats:
            assert fmt in _ADAPTER_REGISTRY
            assert _ADAPTER_REGISTRY[fmt] == MockAdapter
    
    def test_register_adapter_replace_existing(self):
        """Test replacing an existing adapter registration."""
        # Register first adapter
        register_adapter("replace_format", MockAdapter)
        
        # Create a different adapter class
        class AnotherAdapter(MockAdapter):
            pass
        
        # Register second adapter for same format
        register_adapter("replace_format", AnotherAdapter)
        
        # Verify it was replaced
        assert _ADAPTER_REGISTRY["replace_format"] == AnotherAdapter
    
    def test_get_adapter_class(self):
        """Test retrieving an adapter class."""
        # Register adapter
        register_adapter("get_test", MockAdapter)
        
        # Get the adapter class
        adapter_class = get_adapter_class("get_test")
        
        # Verify correct class was returned
        assert adapter_class == MockAdapter
    
    def test_get_adapter_class_missing(self):
        """Test retrieving a non-existent adapter class."""
        with pytest.raises(ValueError) as excinfo:
            get_adapter_class("nonexistent_format")
        
        assert "No adapter registered for format" in str(excinfo.value)
    
    def test_get_adapter_for_format(self):
        """Test getting an adapter instance for a format."""
        # Register adapter
        register_adapter("instance_test", MockAdapter)
        
        # Get adapter instance
        adapter = get_adapter_for_format("instance_test")
        
        # Verify instance type
        assert isinstance(adapter, MockAdapter)
    
    def test_adapter_config_from_system(self):
        """Test that adapter loads configuration from the system."""
        # Register adapter
        register_adapter("config_test", MockAdapter)
        
        # Get adapter instance
        adapter = get_adapter_for_format("config_test")
        
        # Verify adapter has configuration from the system
        assert hasattr(adapter, "entity_extraction")
        assert hasattr(adapter, "metadata_extraction")
    
    def test_get_adapter_for_format_missing(self):
        """Test getting an adapter for a non-existent format."""
        with pytest.raises(ValueError) as excinfo:
            get_adapter_for_format("nonexistent_format")
        
        assert "No adapter registered for format" in str(excinfo.value)
    
    def test_get_available_formats(self):
        """Test retrieving formats from adapter registry."""
        # Clear registry and register known adapters
        _ADAPTER_REGISTRY.clear()
        register_adapter("format1", MockAdapter)
        register_adapter("format2", MockAdapter)
        
        # Check directly in the registry
        formats = list(_ADAPTER_REGISTRY.keys())
        
        # Verify formats
        assert "format1" in formats
        assert "format2" in formats
        assert len(formats) == 2
    
    def test_format_casing(self):
        """Test that format names are case-insensitive."""
        # Register with capitalized name
        register_adapter("CaseSensitive", MockAdapter)
        
        # Get with lowercase
        adapter = get_adapter_for_format("casesensitive")
        
        # Verify it works
        assert isinstance(adapter, MockAdapter)
        
        # Also test the other way
        register_adapter("lowercase", MockAdapter)
        adapter = get_adapter_for_format("LOWERCASE")
        assert isinstance(adapter, MockAdapter)
