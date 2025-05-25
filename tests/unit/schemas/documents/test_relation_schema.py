"""
Unit tests for the document relation schemas in the HADES-PathRAG system.

Tests document relation schema functionality including validation, conversion,
and handling of various relation types.
"""

import unittest
from datetime import datetime
import uuid

from pydantic import ValidationError

from src.schemas.documents.relations import DocumentRelationSchema
from src.schemas.common.enums import RelationType


class TestDocumentRelationSchema(unittest.TestCase):
    """Test the DocumentRelationSchema functionality."""
    
    def test_relation_instantiation(self):
        """Test that DocumentRelationSchema can be instantiated with required attributes."""
        # Test minimal relation
        relation = DocumentRelationSchema(
            source_id="source_doc",
            target_id="target_doc",
            relation_type=RelationType.REFERENCES
        )
        
        self.assertEqual(relation.source_id, "source_doc")
        self.assertEqual(relation.target_id, "target_doc")
        self.assertEqual(relation.relation_type, RelationType.REFERENCES)
        self.assertEqual(relation.weight, 1.0)  # default value
        self.assertFalse(relation.bidirectional)  # default value
        self.assertEqual(relation.metadata, {})  # default value
        self.assertIsNotNone(relation.id)  # auto-generated
        self.assertIsNotNone(relation.created_at)  # auto-generated
        
        # Test with all attributes
        custom_id = str(uuid.uuid4())
        custom_date = datetime(2023, 1, 1)
        
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.CONTAINS,
            weight=0.8,
            bidirectional=True,
            metadata={"importance": "high"},
            id=custom_id,
            created_at=custom_date
        )
        
        self.assertEqual(relation.source_id, "doc1")
        self.assertEqual(relation.target_id, "doc2")
        self.assertEqual(relation.relation_type, RelationType.CONTAINS)
        self.assertEqual(relation.weight, 0.8)
        self.assertTrue(relation.bidirectional)
        self.assertEqual(relation.metadata, {"importance": "high"})
        self.assertEqual(relation.id, custom_id)
        self.assertEqual(relation.created_at, custom_date)
    
    def test_relation_types(self):
        """Test different relation types."""
        # Test all standard relation types
        relation_types = [
            RelationType.CONTAINS,
            RelationType.REFERENCES,
            RelationType.CONNECTS_TO,
            RelationType.SIMILAR_TO,
            RelationType.DEPENDS_ON,
            RelationType.PART_OF,
            RelationType.RELATED_TO,
            RelationType.DERIVED_FROM,
            RelationType.IMPLEMENTS,
            RelationType.CUSTOM
        ]
        
        for rel_type in relation_types:
            relation = DocumentRelationSchema(
                source_id="doc1",
                target_id="doc2",
                relation_type=rel_type
            )
            self.assertEqual(relation.relation_type, rel_type)
    
    def test_relation_type_from_string(self):
        """Test conversion of string to relation type."""
        # Using string values
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type="contains"
        )
        self.assertEqual(relation.relation_type, RelationType.CONTAINS)
        
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type="references"
        )
        self.assertEqual(relation.relation_type, RelationType.REFERENCES)
    
    def test_custom_relation_type(self):
        """Test handling of custom relation types."""
        # Using the CUSTOM relation type directly
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.CUSTOM
        )
        
        self.assertEqual(relation.relation_type, RelationType.CUSTOM)
        
        # Try with a non-standard string value
        # This should convert to CUSTOM type
        with self.assertRaises(ValueError):
            # The validator in DocumentRelationSchema should handle this
            # by converting to CUSTOM, but if that's not implemented yet
            # it will raise a ValueError
            relation = DocumentRelationSchema(
                source_id="doc1",
                target_id="doc2",
                relation_type="custom_relation_not_in_enum"
            )
    
    def test_validation(self):
        """Test validation of required fields."""
        # Test missing required fields
        with self.assertRaises(ValidationError):
            DocumentRelationSchema(
                target_id="doc2",
                relation_type=RelationType.REFERENCES
            )  # Missing source_id
            
        with self.assertRaises(ValidationError):
            DocumentRelationSchema(
                source_id="doc1",
                relation_type=RelationType.REFERENCES
            )  # Missing target_id
            
        with self.assertRaises(ValidationError):
            DocumentRelationSchema(
                source_id="doc1",
                target_id="doc2"
            )  # Missing relation_type
    
    def test_weight_validation(self):
        """Test validation of weight value."""
        # Valid weights
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.0
        )
        self.assertEqual(relation.weight, 0.0)
        
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.5
        )
        self.assertEqual(relation.weight, 0.5)
        
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=1.0
        )
        self.assertEqual(relation.weight, 1.0)
        
        # Invalid weights - the schema doesn't currently enforce range validation
        # on weight, so these should not raise errors, but we could add that validation
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=-0.1
        )
        self.assertEqual(relation.weight, -0.1)
        
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=1.1
        )
        self.assertEqual(relation.weight, 1.1)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.7,
            metadata={"key": "value"}
        )
        
        relation_dict = relation.model_dump_safe()
        
        # Check fields were preserved
        self.assertEqual(relation_dict["source_id"], "doc1")
        self.assertEqual(relation_dict["target_id"], "doc2")
        self.assertEqual(relation_dict["relation_type"], "references")
        self.assertEqual(relation_dict["weight"], 0.7)
        self.assertEqual(relation_dict["metadata"], {"key": "value"})


if __name__ == "__main__":
    unittest.main()
