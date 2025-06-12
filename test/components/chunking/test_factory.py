"""
Unit tests for chunking component factory.
"""

import pytest
from typing import Dict, Any

from src.components.chunking.factory import (
    create_chunking_component,
    get_available_chunking_components,
    register_chunking_component,
    is_chunking_component_available
)
from src.types.components.protocols import Chunker
from src.types.components.contracts import ComponentType


class TestChunkingFactory:
    """Test suite for chunking component factory."""
    
    def test_create_core_chunker(self):
        """Test creating core chunker component."""
        chunker = create_chunking_component("core")
        
        assert isinstance(chunker, Chunker)
        assert chunker.name == "core"
        assert chunker.version == "1.0.0"
        assert chunker.component_type == ComponentType.CHUNKING
    
    def test_create_cpu_chunker(self):
        """Test creating CPU chunker component."""
        chunker = create_chunking_component("cpu")
        
        assert isinstance(chunker, Chunker)
        assert chunker.name == "cpu"
        assert chunker.component_type == ComponentType.CHUNKING
    
    def test_create_text_chunker(self):
        """Test creating text chunker component."""
        chunker = create_chunking_component("text")
        
        assert isinstance(chunker, Chunker)
        assert chunker.name == "text"
        assert chunker.component_type == ComponentType.CHUNKING
    
    def test_create_code_chunker(self):
        """Test creating code chunker component."""
        chunker = create_chunking_component("code")
        
        assert isinstance(chunker, Chunker)
        assert chunker.name == "code"
        assert chunker.component_type == ComponentType.CHUNKING
    
    def test_create_with_config(self):
        """Test creating chunker with configuration."""
        config = {
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        
        chunker = create_chunking_component("cpu", config=config)
        
        assert isinstance(chunker, Chunker)
        assert chunker._config["chunk_size"] == 500
        assert chunker._config["chunk_overlap"] == 50
    
    def test_create_unknown_component(self):
        """Test creating unknown component raises error."""
        with pytest.raises(ValueError, match="Unknown chunking component"):
            create_chunking_component("unknown_chunker")
    
    def test_get_available_components(self):
        """Test getting available components."""
        components = get_available_chunking_components()
        
        assert isinstance(components, dict)
        assert "core" in components
        assert "cpu" in components
        assert "text" in components
        assert "code" in components
        
        # Check that all values are classes
        for name, cls in components.items():
            assert isinstance(cls, type)
    
    def test_is_component_available(self):
        """Test checking component availability."""
        assert is_chunking_component_available("core") == True
        assert is_chunking_component_available("cpu") == True
        assert is_chunking_component_available("text") == True
        assert is_chunking_component_available("code") == True
        assert is_chunking_component_available("unknown") == False
    
    def test_register_new_component(self):
        """Test registering a new component."""
        
        # Create a mock chunker class
        class MockChunker:
            def __init__(self, config=None):
                self.config = config or {}
                self.name = "mock"
                self.version = "1.0.0"
                self.component_type = ComponentType.CHUNKING
        
        # Register the component
        register_chunking_component("mock", MockChunker)
        
        # Verify it's available
        assert is_chunking_component_available("mock") == True
        
        # Verify we can create it
        chunker = create_chunking_component("mock")
        assert chunker.name == "mock"
        
        # Verify it appears in available components
        components = get_available_chunking_components()
        assert "mock" in components
        assert components["mock"] == MockChunker
    
    def test_register_with_custom_factory(self):
        """Test registering component with custom factory function."""
        
        class CustomChunker:
            def __init__(self, special_param=None):
                self.special_param = special_param
                self.name = "custom"
                self.version = "2.0.0"
                self.component_type = ComponentType.CHUNKING
        
        # Custom factory function
        def custom_factory(config):
            special_param = config.get("special_param", "default") if config else "default"
            return CustomChunker(special_param=special_param)
        
        # Register with custom factory
        register_chunking_component("custom", CustomChunker, factory_func=custom_factory)
        
        # Test creating with config
        config = {"special_param": "test_value"}
        chunker = create_chunking_component("custom", config=config)
        
        assert chunker.name == "custom"
        assert chunker.special_param == "test_value"
    
    def test_factory_error_handling(self):
        """Test factory error handling."""
        
        class FailingChunker:
            def __init__(self, config=None):
                raise RuntimeError("Initialization failed")
        
        # Register failing component
        register_chunking_component("failing", FailingChunker)
        
        # Should raise RuntimeError when trying to create
        with pytest.raises(RuntimeError, match="Component creation failed"):
            create_chunking_component("failing")
    
    def test_all_default_components_creatable(self):
        """Test that all default components can be created successfully."""
        default_components = ["core", "cpu", "text", "code"]
        
        for component_name in default_components:
            try:
                chunker = create_chunking_component(component_name)
                assert chunker.name == component_name
                assert chunker.component_type == ComponentType.CHUNKING
                assert hasattr(chunker, 'version')
                assert hasattr(chunker, 'chunk')
                
            except Exception as e:
                pytest.fail(f"Failed to create {component_name} component: {e}")
    
    def test_component_interface_compliance(self):
        """Test that created components comply with Chunker interface."""
        chunker = create_chunking_component("core")
        
        # Check required properties
        assert hasattr(chunker, 'name')
        assert hasattr(chunker, 'version')
        assert hasattr(chunker, 'component_type')
        
        # Check required methods
        assert hasattr(chunker, 'configure')
        assert hasattr(chunker, 'validate_config')
        assert hasattr(chunker, 'get_config_schema')
        assert hasattr(chunker, 'health_check')
        assert hasattr(chunker, 'get_metrics')
        assert hasattr(chunker, 'chunk')
        assert hasattr(chunker, 'estimate_chunks')
        assert hasattr(chunker, 'supports_content_type')
        assert hasattr(chunker, 'get_optimal_chunk_size')
        
        # Check methods are callable
        assert callable(chunker.configure)
        assert callable(chunker.validate_config)
        assert callable(chunker.get_config_schema)
        assert callable(chunker.health_check)
        assert callable(chunker.get_metrics)
        assert callable(chunker.chunk)
        assert callable(chunker.estimate_chunks)
        assert callable(chunker.supports_content_type)
        assert callable(chunker.get_optimal_chunk_size)


if __name__ == "__main__":
    pytest.main([__file__])