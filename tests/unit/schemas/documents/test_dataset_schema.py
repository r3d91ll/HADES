"""
Unit tests for the dataset schemas in the HADES-PathRAG system.

Tests dataset schema functionality including validation, document and relation
management, and conversion from other formats.
"""

import unittest
import uuid
from datetime import datetime
from typing import Dict

from pydantic import ValidationError

from src.schemas.common.enums import DocumentType, SchemaVersion
from src.schemas.documents.base import DocumentSchema
from src.schemas.documents.dataset import DatasetSchema
from src.schemas.documents.relations import DocumentRelationSchema
from src.schemas.common.enums import RelationType


class TestDatasetSchema(unittest.TestCase):
    """Test the DatasetSchema functionality."""
    
    def test_dataset_instantiation(self):
        """Test that DatasetSchema can be instantiated with required attributes."""
        # Test minimal dataset
        dataset = DatasetSchema(
            id="dataset_1",
            name="Test Dataset"
        )
        
        self.assertEqual(dataset.id, "dataset_1")
        self.assertEqual(dataset.name, "Test Dataset")
        self.assertEqual(dataset.schema_version, SchemaVersion.V2)
        self.assertEqual(dataset.documents, {})
        self.assertEqual(dataset.relations, [])
        self.assertIsNotNone(dataset.created_at)
        self.assertIsNotNone(dataset.updated_at)
        
        # Test with more attributes
        dataset = DatasetSchema(
            id="dataset_2",
            name="Test Dataset 2",
            description="A test dataset",
            schema_version=SchemaVersion.V1,
            created_at=datetime(2023, 1, 1),
            metadata={"source": "test"}
        )
        
        self.assertEqual(dataset.id, "dataset_2")
        self.assertEqual(dataset.name, "Test Dataset 2")
        self.assertEqual(dataset.description, "A test dataset")
        self.assertEqual(dataset.schema_version, SchemaVersion.V1)
        self.assertEqual(dataset.created_at, datetime(2023, 1, 1))
        self.assertEqual(dataset.updated_at, datetime(2023, 1, 1))  # Should be set to created_at
        self.assertEqual(dataset.metadata, {"source": "test"})
    
    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        # Empty ID should be auto-generated
        dataset = DatasetSchema(
            id="",  # This empty string will be replaced with an auto-generated UUID
            name="Auto ID Dataset"
        )
        self.assertIsNotNone(dataset.id)
        self.assertTrue(len(dataset.id) > 0)
        
        # In Pydantic v2, required fields can't be omitted,
        # but the validator should still work with empty strings
        dataset2 = DatasetSchema(
            id="",  # Empty string should also trigger auto-generation
            name="Another Dataset"
        )
        self.assertIsNotNone(dataset2.id)
        self.assertTrue(len(dataset2.id) > 0)
        
        # Verify different IDs are generated each time
        self.assertNotEqual(dataset.id, dataset2.id)
    
    def test_add_document(self):
        """Test adding documents to the dataset."""
        dataset = DatasetSchema(
            id="dataset_3",
            name="Document Test Dataset"
        )
        
        # Create a test document
        doc1 = DocumentSchema(
            id="doc1",
            content="Test document content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        
        # Add document to dataset
        dataset.add_document(doc1)
        
        # Check document was added
        self.assertEqual(len(dataset.documents), 1)
        self.assertIn("doc1", dataset.documents)
        self.assertEqual(dataset.documents["doc1"], doc1)
        
        # Add another document
        doc2 = DocumentSchema(
            id="doc2",
            content="Another test document",
            source="test2.txt",
            document_type=DocumentType.TEXT
        )
        dataset.add_document(doc2)
        
        # Check second document was added
        self.assertEqual(len(dataset.documents), 2)
        self.assertIn("doc2", dataset.documents)
        
        # Check updated_at was changed
        self.assertNotEqual(dataset.updated_at, dataset.created_at)
    
    def test_add_relation(self):
        """Test adding relations between documents."""
        dataset = DatasetSchema(
            id="dataset_4",
            name="Relation Test Dataset"
        )
        
        # Create test documents
        doc1 = DocumentSchema(
            id="doc1",
            content="Test document content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        
        doc2 = DocumentSchema(
            id="doc2",
            content="Another test document",
            source="test2.txt",
            document_type=DocumentType.TEXT
        )
        
        # Add documents to dataset
        dataset.add_document(doc1)
        dataset.add_document(doc2)
        
        # Create a relation
        relation = DocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.8
        )
        
        # Add relation to dataset
        dataset.add_relation(relation)
        
        # Check relation was added
        self.assertEqual(len(dataset.relations), 1)
        self.assertEqual(dataset.relations[0], relation)
        
        # Create and add another relation
        relation2 = DocumentRelationSchema(
            source_id="doc2",
            target_id="doc1",
            relation_type=RelationType.PART_OF,
            weight=0.5
        )
        dataset.add_relation(relation2)
        
        # Check second relation was added
        self.assertEqual(len(dataset.relations), 2)
    
    def test_add_relation_validation(self):
        """Test validation when adding relations."""
        dataset = DatasetSchema(
            id="dataset_5",
            name="Validation Test Dataset"
        )
        
        # Create a test document
        doc1 = DocumentSchema(
            id="doc1",
            content="Test document content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        
        # Add document to dataset
        dataset.add_document(doc1)
        
        # Create invalid relations (referencing non-existent documents)
        invalid_relation1 = DocumentRelationSchema(
            source_id="nonexistent",
            target_id="doc1",
            relation_type=RelationType.REFERENCES
        )
        
        invalid_relation2 = DocumentRelationSchema(
            source_id="doc1",
            target_id="nonexistent",
            relation_type=RelationType.REFERENCES
        )
        
        # Check that adding invalid relations raises ValueError
        with self.assertRaises(ValueError):
            dataset.add_relation(invalid_relation1)
            
        with self.assertRaises(ValueError):
            dataset.add_relation(invalid_relation2)
    
    def test_from_ingest_dataset(self):
        """Test conversion from IngestDataset."""
        # Create a mock ingest dataset dictionary
        ingest_data = {
            "id": "ingest_dataset_1",
            "name": "Ingest Test Dataset",
            "documents": {
                "doc1": {
                    "id": "doc1",
                    "content": "Test document content",
                    "source": "test.txt",
                    "type": "text"  # Note: using 'type' instead of 'document_type'
                },
                "doc2": {
                    "id": "doc2",
                    "content": "Another test document",
                    "source": "test2.txt",
                    "type": "markdown"
                }
            },
            "relations": [
                {
                    "source_id": "doc1",
                    "target_id": "doc2",
                    "relation_type": "references",
                    "weight": 0.7
                }
            ],
            "metadata": {
                "source": "ingest_test"
            }
        }
        
        # Convert to DatasetSchema
        dataset = DatasetSchema.from_ingest_dataset(ingest_data)
        
        # Check dataset fields
        self.assertEqual(dataset.id, "ingest_dataset_1")
        self.assertEqual(dataset.name, "Ingest Test Dataset")
        self.assertEqual(dataset.metadata, {"source": "ingest_test"})
        
        # Check documents were converted
        self.assertEqual(len(dataset.documents), 2)
        self.assertIn("doc1", dataset.documents)
        self.assertIn("doc2", dataset.documents)
        
        doc1 = dataset.documents["doc1"]
        self.assertEqual(doc1.id, "doc1")
        self.assertEqual(doc1.content, "Test document content")
        self.assertEqual(doc1.document_type, DocumentType.TEXT)
        
        doc2 = dataset.documents["doc2"]
        self.assertEqual(doc2.id, "doc2")
        self.assertEqual(doc2.content, "Another test document")
        self.assertEqual(doc2.document_type, DocumentType.MARKDOWN)
        
        # Check relations were converted
        self.assertEqual(len(dataset.relations), 1)
        relation = dataset.relations[0]
        self.assertEqual(relation.source_id, "doc1")
        self.assertEqual(relation.target_id, "doc2")
        self.assertEqual(relation.relation_type, RelationType.REFERENCES)
        self.assertEqual(relation.weight, 0.7)
    
    def test_from_ingest_dataset_with_object(self):
        """Test conversion from an object with to_dict method."""
        # Create a mock IngestDataset class
        class MockIngestDataset:
            def __init__(self, data: Dict):
                self.data = data
                
            def to_dict(self):
                return self.data
        
        # Create mock data
        mock_data = {
            "id": "mock_dataset_1",
            "name": "Mock Dataset",
            "documents": {
                "doc1": {
                    "id": "doc1",
                    "content": "Mock document content",
                    "source": "mock.txt",
                    "type": "text"
                }
            },
            "relations": []
        }
        
        # Create mock object
        mock_dataset = MockIngestDataset(mock_data)
        
        # Convert to DatasetSchema
        dataset = DatasetSchema.from_ingest_dataset(mock_dataset)
        
        # Check conversion
        self.assertEqual(dataset.id, "mock_dataset_1")
        self.assertEqual(dataset.name, "Mock Dataset")
        self.assertEqual(len(dataset.documents), 1)
        self.assertIn("doc1", dataset.documents)


if __name__ == "__main__":
    unittest.main()
