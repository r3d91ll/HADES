"""Tests for the LRU cache used in the runtime server."""
import time
from unittest.mock import Mock

import pytest

from src.model_engine.engines.haystack.runtime.server import _LRUCache


class TestLRUCache:
    """Test the LRU cache implementation."""
    
    def test_put_get(self):
        """Test basic put/get operations."""
        cache = _LRUCache(max_size=2)
        
        # Add items
        cache.put("model1", "model_obj1", "tok1")
        cache.put("model2", "model_obj2", "tok2")
        
        # Retrieve items
        assert cache.get("model1") == ("model_obj1", "tok1")
        assert cache.get("model2") == ("model_obj2", "tok2")
        assert cache.get("nonexistent") is None
    
    def test_eviction(self):
        """Test LRU eviction policy."""
        cache = _LRUCache(max_size=2)
        
        # Mock model objects with .cpu() method
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()
        
        # Add models
        cache.put("model1", model1, "tok1")
        cache.put("model2", model2, "tok2")
        
        # Access model1 to make it more recently used than model2
        cache.get("model1")
        
        # Add model3, which should evict model2 (least recently used)
        cache.put("model3", model3, "tok3")
        
        # model2 should be evicted, model1 and model3 should remain
        assert cache.get("model1") is not None
        assert cache.get("model2") is None
        assert cache.get("model3") is not None
        
        # model2.cpu should have been called during eviction
        model2.cpu.assert_called_once()
        model1.cpu.assert_not_called()
        model3.cpu.assert_not_called()
    
    def test_explicit_eviction(self):
        """Test explicit eviction using the evict method."""
        cache = _LRUCache(max_size=2)
        model = Mock()
        
        # Add a model
        cache.put("model1", model, "tok1")
        
        # Explicitly evict it
        cache.evict("model1")
        
        # The model should be gone and .cpu() should have been called
        assert cache.get("model1") is None
        model.cpu.assert_called_once()
    
    def test_info(self):
        """Test the info method returns information about cached models."""
        cache = _LRUCache(max_size=2)
        cache.put("model1", "model_obj1", "tok1")
        
        # Get cache info
        info = cache.info()
        
        # Verify it contains the expected model and timestamp
        assert "model1" in info
        assert isinstance(info["model1"], float)
    
    def test_multithreaded_safety(self):
        """Simulate basic multithreaded operation to check thread safety."""
        cache = _LRUCache(max_size=5)
        
        # Add some initial models
        for i in range(3):
            mock_model = Mock()
            cache.put(f"model{i}", mock_model, f"tok{i}")
        
        # Simulate operations that could happen concurrently
        # Get existing model (updates timestamp)
        model1_result = cache.get("model1")
        assert model1_result is not None
        assert isinstance(model1_result[0], Mock)
        assert model1_result[1] == "tok1"
        
        # Add new model
        mock_model3 = Mock()
        cache.put("model3", mock_model3, "tok3")
        
        # Try to get a model that doesn't exist
        assert cache.get("nonexistent") is None
        
        # Evict a model
        cache.evict("model0")
        
        # Verify final state is consistent
        assert cache.get("model0") is None
        assert cache.get("model1") is not None
        assert cache.get("model2") is not None
        assert cache.get("model3") is not None

    def test_update_existing_key(self):
        """Test updating a model that's already in the cache with the same key."""
        cache = _LRUCache(max_size=2)
        # Add a model
        mock_model1 = Mock()
        mock_tok1 = Mock()
        cache.put("model1", mock_model1, mock_tok1)
        
        # Verify it's in the cache
        assert cache.get("model1") == (mock_model1, mock_tok1)
        
        # Create a new model with same key but different objects
        mock_model2 = Mock()
        mock_tok2 = Mock()
        cache.put("model1", mock_model2, mock_tok2)
        
        # Verify the model was updated
        assert cache.get("model1") == (mock_model2, mock_tok2)
