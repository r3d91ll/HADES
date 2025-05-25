"""
Unit tests for the docproc.adapters.base module.

These tests cover the BaseAdapter abstract class and its implementations:
- Abstract method signatures
- Implementation requirements
- Default method behaviors
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.docproc.adapters.base import BaseAdapter


class ConcreteAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing."""
    
    def __init__(self, format_type: str = "test"):
        """Initialize with test format type."""
        super().__init__(format_type)
        # Add required configuration properties
        self.entity_extraction = True
        self.metadata_extraction = True
        self.chunking_preparation = True
    
    def process(self, file_path, content=None, options=None):
        """Implement required process method."""
        options = options or {}
        content = content or "Test content"
        return {
            "id": f"test_{os.path.basename(file_path)}",
            "content": content,
            "format": self.format_type,
            "metadata": self.extract_metadata(file_path, content, options),
            "entities": self.extract_entities(content, options)
        }
    
    def extract_metadata(self, file_path, content=None, options=None):
        """Implement required extract_metadata method."""
        return {
            "title": "Test Document",
            "format": self.format_type
        }
    
    def extract_entities(self, text, options=None):
        """Implement required extract_entities method."""
        return [
            {
                "type": "test_entity",
                "value": "Test Entity",
                "confidence": 1.0
            }
        ]
    
    def process_text(self, text, options=None):
        """Implement required process_text method."""
        options = options or {}
        return {
            "id": "test_text_doc",
            "content": text,
            "format": self.format_type,
            "metadata": {
                "title": "Text Document",
                "format": self.format_type
            },
            "entities": self.extract_entities(text, options)
        }


class TestBaseAdapter:
    """Tests for the BaseAdapter abstract class."""
    
    def test_cannot_instantiate_base_adapter(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()
    
    def test_concrete_adapter_instantiation(self):
        """Test that a concrete implementation can be instantiated."""
        adapter = ConcreteAdapter()
        assert isinstance(adapter, BaseAdapter)
    
    def test_init_with_format_type(self):
        """Test initializing adapter with format type."""
        adapter = ConcreteAdapter(format_type="custom_format")
        assert adapter.format_type == "custom_format"
    
    def test_init_default_format_type(self):
        """Test initializing adapter with default format type."""
        adapter = ConcreteAdapter()
        assert adapter.format_type == "test"
    
    def test_has_configuration(self):
        """Test that configuration is loaded."""
        adapter = ConcreteAdapter()
        # Check that some configuration properties exist
        assert hasattr(adapter, "entity_extraction")
        assert hasattr(adapter, "metadata_extraction")
        assert hasattr(adapter, "chunking_preparation")


class TestConcreteAdapter:
    """Tests for a concrete implementation of BaseAdapter."""
    
    def test_process_method(self):
        """Test the process method implementation."""
        adapter = ConcreteAdapter()
        
        with tempfile.NamedTemporaryFile() as tmp:
            result = adapter.process(tmp.name)
            
            # Check required fields
            assert "id" in result
            assert "content" in result
            assert "format" in result
            assert "metadata" in result
            assert "entities" in result
            
            # Check content
            assert result["content"] == "Test content"
            assert result["format"] == "test"
    
    def test_process_with_content(self):
        """Test process method with provided content."""
        adapter = ConcreteAdapter()
        
        with tempfile.NamedTemporaryFile() as tmp:
            content = "Provided content"
            result = adapter.process(tmp.name, content=content)
            
            # Check that provided content is used
            assert result["content"] == content
    
    def test_process_with_options(self):
        """Test process method with options."""
        adapter = ConcreteAdapter()
        
        with tempfile.NamedTemporaryFile() as tmp:
            options = {"option1": "value1"}
            result = adapter.process(tmp.name, options=options)
            
            # Result should still have all required fields
            assert "id" in result
            assert "content" in result
            assert "format" in result
            assert "metadata" in result
            assert "entities" in result
    
    def test_extract_metadata(self):
        """Test the extract_metadata method."""
        adapter = ConcreteAdapter()
        
        with tempfile.NamedTemporaryFile() as tmp:
            metadata = adapter.extract_metadata(tmp.name)
            
            # Check metadata fields
            assert "title" in metadata
            assert "format" in metadata
            assert metadata["title"] == "Test Document"
            assert metadata["format"] == "test"
    
    def test_extract_entities(self):
        """Test the extract_entities method."""
        adapter = ConcreteAdapter()
        
        text = "Test text for entity extraction"
        entities = adapter.extract_entities(text)
        
        # Check entity structure
        assert len(entities) > 0
        assert "type" in entities[0]
        assert "value" in entities[0]
        assert "confidence" in entities[0]
    
    def test_process_text(self):
        """Test the process_text method."""
        adapter = ConcreteAdapter()
        
        text = "Test text for processing"
        result = adapter.process_text(text)
        
        # Check required fields
        assert "id" in result
        assert "content" in result
        assert "format" in result
        assert "metadata" in result
        assert "entities" in result
        
        # Check content
        assert result["content"] == text
    
    def test_process_text_with_options(self):
        """Test process_text method with options."""
        adapter = ConcreteAdapter()
        
        text = "Test text for processing"
        options = {"option1": "value1"}
        result = adapter.process_text(text, options=options)
        
        # Result should still have all required fields
        assert "id" in result
        assert "content" in result
        assert "format" in result
        assert "metadata" in result
        assert "entities" in result
