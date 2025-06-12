"""
Unit tests for docproc component factory.
"""

import pytest
from unittest.mock import patch, Mock

from src.components.docproc.factory import (
    create_docproc_component,
    get_available_docproc_components,
    register_docproc_component,
    is_docproc_component_available,
    DOCPROC_REGISTRY
)
from src.components.docproc.core.processor import CoreDocumentProcessor
from src.components.docproc.docling.processor import DoclingDocumentProcessor
from src.types.components.protocols import DocumentProcessor


class TestDocprocFactory:
    """Test suite for docproc component factory."""
    
    def test_create_core_component(self):
        """Test creating core docproc component."""
        component = create_docproc_component("core")
        
        assert isinstance(component, CoreDocumentProcessor)
        assert component.name == "core"
        assert component.component_type.value == "docproc"
    
    def test_create_core_component_with_config(self):
        """Test creating core component with configuration."""
        config = {
            "processing_options": {
                "extract_metadata": False
            }
        }
        
        component = create_docproc_component("core", config=config)
        
        assert isinstance(component, CoreDocumentProcessor)
        assert component._config["processing_options"]["extract_metadata"] == False
    
    @patch('src.components.docproc.docling.processor.DoclingDocumentProcessor._check_docling_availability', return_value=True)
    def test_create_docling_component(self, mock_availability):
        """Test creating docling docproc component."""
        component = create_docproc_component("docling")
        
        assert isinstance(component, DoclingDocumentProcessor)
        assert component.name == "docling"
        assert component.component_type.value == "docproc"
    
    def test_create_unknown_component(self):
        """Test creating unknown component raises ValueError."""
        with pytest.raises(ValueError, match="Unknown docproc component: unknown"):
            create_docproc_component("unknown")
    
    def test_create_component_creation_failure(self):
        """Test handling component creation failure."""
        # Mock a component that fails during creation
        original_registry = DOCPROC_REGISTRY.copy()
        
        def failing_factory(config):
            raise Exception("Creation failed")
        
        DOCPROC_REGISTRY["failing"] = (CoreDocumentProcessor, failing_factory)
        
        try:
            with pytest.raises(RuntimeError, match="Component creation failed"):
                create_docproc_component("failing")
        finally:
            # Restore original registry
            DOCPROC_REGISTRY.clear()
            DOCPROC_REGISTRY.update(original_registry)
    
    def test_get_available_components(self):
        """Test getting available components."""
        components = get_available_docproc_components()
        
        assert isinstance(components, dict)
        assert "core" in components
        assert "docling" in components
        assert components["core"] == CoreDocumentProcessor
        assert components["docling"] == DoclingDocumentProcessor
    
    def test_register_component(self):
        """Test registering a new component."""
        # Create a mock component class
        class MockDocprocComponent:
            def __init__(self, config=None):
                self.config = config or {}
                self.name = "mock"
                self.component_type = Mock()
                self.component_type.value = "docproc"
        
        # Save original registry state
        original_registry = DOCPROC_REGISTRY.copy()
        
        try:
            # Register the mock component
            register_docproc_component("mock", MockDocprocComponent)
            
            # Verify registration
            assert "mock" in DOCPROC_REGISTRY
            assert is_docproc_component_available("mock")
            
            # Test creating the registered component
            component = create_docproc_component("mock")
            assert isinstance(component, MockDocprocComponent)
            assert component.name == "mock"
            
        finally:
            # Restore original registry
            DOCPROC_REGISTRY.clear()
            DOCPROC_REGISTRY.update(original_registry)
    
    def test_register_component_with_custom_factory(self):
        """Test registering component with custom factory function."""
        class MockDocprocComponent:
            def __init__(self, custom_param=None):
                self.custom_param = custom_param
                self.name = "mock_custom"
                self.component_type = Mock()
                self.component_type.value = "docproc"
        
        def custom_factory(config):
            return MockDocprocComponent(custom_param=config.get("custom_param", "default"))
        
        # Save original registry state
        original_registry = DOCPROC_REGISTRY.copy()
        
        try:
            # Register with custom factory
            register_docproc_component("mock_custom", MockDocprocComponent, custom_factory)
            
            # Test creating with custom factory
            config = {"custom_param": "test_value"}
            component = create_docproc_component("mock_custom", config=config)
            
            assert isinstance(component, MockDocprocComponent)
            assert component.custom_param == "test_value"
            
        finally:
            # Restore original registry
            DOCPROC_REGISTRY.clear()
            DOCPROC_REGISTRY.update(original_registry)
    
    def test_is_component_available(self):
        """Test checking component availability."""
        assert is_docproc_component_available("core") == True
        assert is_docproc_component_available("docling") == True
        assert is_docproc_component_available("unknown") == False
    
    def test_registry_integrity(self):
        """Test that registry contains expected components."""
        assert "core" in DOCPROC_REGISTRY
        assert "docling" in DOCPROC_REGISTRY
        
        # Check registry structure
        for name, (cls, factory) in DOCPROC_REGISTRY.items():
            assert callable(factory)
            assert hasattr(cls, '__name__')
    
    def test_factory_functions_work(self):
        """Test that all factory functions in registry work."""
        for name, (cls, factory) in DOCPROC_REGISTRY.items():
            if name == "docling":
                # Skip docling as it requires special setup
                continue
                
            # Test factory function
            component = factory(None)
            assert hasattr(component, 'name')
            assert hasattr(component, 'component_type')


if __name__ == "__main__":
    pytest.main([__file__])