"""
Unit tests for the base schema functionality in the HADES-PathRAG system.

Tests the core schema functionality including validation, serialization, and
utility methods provided by the BaseSchema class.
"""

import unittest
import numpy as np
from pydantic import Field, ValidationError

from src.schemas.common.base import BaseSchema
from src.schemas.common.types import EmbeddingVector


class TestBaseSchema(unittest.TestCase):
    """Test the core BaseSchema class functionality."""
    
    def test_base_schema_instantiation(self):
        """Test that BaseSchema can be instantiated with basic attributes."""
        
        class TestSchema(BaseSchema):
            name: str = Field(..., description="Test name")
            value: int = Field(default=0, description="Test value")
            
        # Test with required fields
        schema = TestSchema(name="test")
        self.assertEqual(schema.name, "test")
        self.assertEqual(schema.value, 0)
        
        # Test with all fields
        schema = TestSchema(name="test", value=10)
        self.assertEqual(schema.name, "test")
        self.assertEqual(schema.value, 10)
    
    def test_validation(self):
        """Test schema validation behavior."""
        
        class TestSchema(BaseSchema):
            name: str = Field(..., description="Test name")
            value: int = Field(default=0, description="Test value", ge=0, le=100)
            
        # Test valid values
        schema = TestSchema(name="test", value=50)
        self.assertEqual(schema.value, 50)
        
        # Test invalid values
        with self.assertRaises(ValidationError):
            TestSchema(name="test", value=101)  # Value too high
        
        with self.assertRaises(ValidationError):
            TestSchema(name="test", value=-1)  # Value too low
        
        with self.assertRaises(ValidationError):
            TestSchema(value=50)  # Missing required field
    
    def test_model_dump_safe(self):
        """Test the model_dump_safe method with various types."""
        
        class TestSchema(BaseSchema):
            name: str
            values: list[float] = Field(default_factory=list)
            vector: EmbeddingVector = Field(default=None)
            
        # Test with regular values
        schema = TestSchema(name="test", values=[1.0, 2.0, 3.0])
        data = schema.model_dump_safe(exclude_none=False)
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["values"], [1.0, 2.0, 3.0])
        self.assertIsNone(data["vector"])
        
        # Test with numpy array
        numpy_vector = np.array([1.0, 2.0, 3.0])
        schema = TestSchema(name="test", vector=numpy_vector)
        data = schema.model_dump_safe()
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["vector"], [1.0, 2.0, 3.0])  # Should be converted to list
        self.assertIsInstance(data["vector"], list)  # Check type conversion
        
        # Test exclude_none option
        schema = TestSchema(name="test")
        data = schema.model_dump_safe(exclude_none=True)
        self.assertNotIn("vector", data)
        
        data = schema.model_dump_safe(exclude_none=False)
        self.assertIn("vector", data)
        self.assertIsNone(data["vector"])


if __name__ == "__main__":
    unittest.main()
