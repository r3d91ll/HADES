"""
Unit tests for the docproc enum type definitions.

This module tests the enum definitions that have been migrated to 
centralized type files in the src/types/docproc directory.
"""

import os
import sys
import unittest
import enum

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.types.docproc.enums import (
    RelationshipType,
    ImportSourceType,
    AccessLevel
)


class TestRelationshipType(unittest.TestCase):
    """Test cases for the RelationshipType enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        expected_values = {
            "CALLS": "CALLS",
            "CONTAINS": "CONTAINS",
            "IMPLEMENTS": "IMPLEMENTS",
            "IMPORTS": "IMPORTS",
            "REFERENCES": "REFERENCES",
            "EXTENDS": "EXTENDS",
            "SIMILAR_TO": "SIMILAR_TO",
            "RELATED_TO": "RELATED_TO"
        }
        
        # Verify all expected values exist
        for name, value in expected_values.items():
            self.assertTrue(hasattr(RelationshipType, name))
            self.assertEqual(getattr(RelationshipType, name).value, value)
            
        # Verify no unexpected values exist
        self.assertEqual(len(RelationshipType), len(expected_values))
    
    def test_is_str_enum(self) -> None:
        """Test that RelationshipType inherits from str and enum.Enum."""
        self.assertTrue(issubclass(RelationshipType, str))
        self.assertTrue(issubclass(RelationshipType, enum.Enum))
        
    def test_primary_relationships(self) -> None:
        """Test identifying primary relationships."""
        primary_relationships = {
            RelationshipType.CALLS,
            RelationshipType.CONTAINS,
            RelationshipType.IMPLEMENTS
        }
        
        for rel_type in RelationshipType:
            if rel_type in primary_relationships:
                self.assertIn(rel_type, primary_relationships)
            else:
                self.assertNotIn(rel_type, primary_relationships)
    
    def test_secondary_relationships(self) -> None:
        """Test identifying secondary relationships."""
        secondary_relationships = {
            RelationshipType.IMPORTS,
            RelationshipType.REFERENCES,
            RelationshipType.EXTENDS
        }
        
        for rel_type in RelationshipType:
            if rel_type in secondary_relationships:
                self.assertIn(rel_type, secondary_relationships)
            else:
                self.assertNotIn(rel_type, secondary_relationships)
    
    def test_tertiary_relationships(self) -> None:
        """Test identifying tertiary relationships."""
        tertiary_relationships = {
            RelationshipType.SIMILAR_TO,
            RelationshipType.RELATED_TO
        }
        
        for rel_type in RelationshipType:
            if rel_type in tertiary_relationships:
                self.assertIn(rel_type, tertiary_relationships)
            else:
                self.assertNotIn(rel_type, tertiary_relationships)


class TestImportSourceType(unittest.TestCase):
    """Test cases for the ImportSourceType enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        expected_values = {
            "STDLIB": "stdlib",
            "THIRD_PARTY": "third-party",
            "LOCAL": "local",
            "UNKNOWN": "unknown"
        }
        
        # Verify all expected values exist
        for name, value in expected_values.items():
            self.assertTrue(hasattr(ImportSourceType, name))
            self.assertEqual(getattr(ImportSourceType, name).value, value)
            
        # Verify no unexpected values exist
        self.assertEqual(len(ImportSourceType), len(expected_values))
    
    def test_is_str_enum(self) -> None:
        """Test that ImportSourceType inherits from str and enum.Enum."""
        self.assertTrue(issubclass(ImportSourceType, str))
        self.assertTrue(issubclass(ImportSourceType, enum.Enum))


class TestAccessLevel(unittest.TestCase):
    """Test cases for the AccessLevel enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        expected_values = {
            "PUBLIC": "public",
            "PROTECTED": "protected",
            "PRIVATE": "private"
        }
        
        # Verify all expected values exist
        for name, value in expected_values.items():
            self.assertTrue(hasattr(AccessLevel, name))
            self.assertEqual(getattr(AccessLevel, name).value, value)
            
        # Verify no unexpected values exist
        self.assertEqual(len(AccessLevel), len(expected_values))
    
    def test_is_str_enum(self) -> None:
        """Test that AccessLevel inherits from str and enum.Enum."""
        self.assertTrue(issubclass(AccessLevel, str))
        self.assertTrue(issubclass(AccessLevel, enum.Enum))
    
    def test_name_prefix_matching(self) -> None:
        """Test that each access level corresponds to a Python naming convention."""
        # Public: no underscore prefix
        self.assertEqual(AccessLevel.PUBLIC.value, "public")
        
        # Protected: single underscore prefix
        self.assertEqual(AccessLevel.PROTECTED.value, "protected")
        
        # Private: double underscore prefix
        self.assertEqual(AccessLevel.PRIVATE.value, "private")


if __name__ == "__main__":
    unittest.main()
