"""
Additional unit tests for document schema module to improve coverage.

This module adds tests for methods that weren't covered in the main test file.
"""
import unittest
from datetime import datetime
import uuid
import numpy as np
from typing import Dict, Any, List, Optional

from src.schema.document_schema import (
    SchemaVersion,
    DocumentType,
    RelationType,
    EmbeddingVector,
    ChunkMetadata,
    DocumentSchema,
    DocumentRelationSchema,
    DatasetSchema
)


class TestDocumentSchemaAdditional(unittest.TestCase):
    """Additional test cases for DocumentSchema model."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_document_data = {
            "id": str(uuid.uuid4()),
            "content": "This is test content.",
            "source": "test.txt",
            "document_type": "text"
        }
        
        self.document_with_metadata = {
            "id": str(uuid.uuid4()),
            "content": "Document with metadata.",
            "source": "metadata.txt",
            "document_type": "text",
            "title": "Test Document",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {"key1": "value1", "key2": 123},
            "tags": ["tag1", "tag2"]
        }
        
        self.document_with_embedding = {
            "id": str(uuid.uuid4()),
            "content": "Document with embedding.",
            "source": "embedding.txt",
            "document_type": "text",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "embedding_model": "test-model"
        }
        
        # Create a numpy array embedding
        self.numpy_embedding = np.array([0.5, 0.6, 0.7, 0.8])
        self.document_with_numpy_embedding = {
            "id": str(uuid.uuid4()),
            "content": "Document with numpy embedding.",
            "source": "numpy_embedding.txt",
            "document_type": "text",
            "embedding": self.numpy_embedding,
            "embedding_model": "test-model"
        }

    def test_validate_id_empty(self):
        """Test validate_id method with empty ID."""
        # Create document with empty ID
        doc_data = self.valid_document_data.copy()
        doc_data["id"] = ""
        
        # Validate should generate a new UUID
        document = DocumentSchema.model_validate(doc_data)
        self.assertNotEqual(document.id, "")
        self.assertTrue(len(document.id) > 0)
    
    def test_validate_id_none(self):
        """Test validate_id method with None ID."""
        # Test the validator directly
        result = DocumentSchema.validate_id(None)
        
        # Should generate a new UUID
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)
        
        # Verify it's a valid UUID string
        try:
            uuid_obj = uuid.UUID(result)
            self.assertTrue(isinstance(uuid_obj, uuid.UUID))
        except ValueError:
            self.fail("Result is not a valid UUID string")
    
    def test_to_dict_basic(self):
        """Test to_dict method with basic document."""
        document = DocumentSchema.model_validate(self.valid_document_data)
        dict_data = document.to_dict()
        
        # Check basic fields
        self.assertEqual(dict_data["id"], self.valid_document_data["id"])
        self.assertEqual(dict_data["content"], self.valid_document_data["content"])
        self.assertEqual(dict_data["source"], self.valid_document_data["source"])
        self.assertEqual(dict_data["document_type"], self.valid_document_data["document_type"])
        
        # Check timestamps are strings
        self.assertIsInstance(dict_data["created_at"], str)
        self.assertIsInstance(dict_data["updated_at"], str)
    
    def test_to_dict_with_numpy_embedding(self):
        """Test to_dict method with numpy embedding."""
        document = DocumentSchema.model_validate(self.document_with_numpy_embedding)
        dict_data = document.to_dict()
        
        # Check embedding is converted to list
        self.assertIsInstance(dict_data["embedding"], list)
        self.assertEqual(len(dict_data["embedding"]), 4)
        self.assertEqual(dict_data["embedding"], self.numpy_embedding.tolist())
    
    def test_from_ingest_document_minimal(self):
        """Test from_ingest_document method with minimal data."""
        # Create a mock IngestDocument
        class MockIngestDocument:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.content = "Mock document content"
                self.source = "mock.txt"
                self.document_type = "text"
        
        mock_doc = MockIngestDocument()
        
        # Convert to DocumentSchema
        document = DocumentSchema.from_ingest_document(mock_doc)
        
        # Check fields
        self.assertEqual(document.id, mock_doc.id)
        self.assertEqual(document.content, mock_doc.content)
        self.assertEqual(document.source, mock_doc.source)
        self.assertEqual(document.document_type, mock_doc.document_type)
    
    def test_from_ingest_document_with_metadata(self):
        """Test from_ingest_document method with metadata."""
        # Create a mock IngestDocument with metadata
        class MockIngestDocument:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.content = "Mock document content"
                self.source = "mock.txt"
                self.document_type = "text"
                self.metadata = {
                    "title": "Mock Title",
                    "author": "Mock Author",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "custom_field": "custom value"
                }
        
        mock_doc = MockIngestDocument()
        
        # Convert to DocumentSchema
        document = DocumentSchema.from_ingest_document(mock_doc)
        
        # Check metadata fields
        self.assertEqual(document.title, mock_doc.metadata["title"])
        self.assertEqual(document.author, mock_doc.metadata["author"])
        self.assertEqual(document.metadata["custom_field"], mock_doc.metadata["custom_field"])
    
    def test_from_ingest_document_with_embedding(self):
        """Test from_ingest_document method with embedding."""
        # Create a mock IngestDocument with embedding
        class MockIngestDocument:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.content = "Mock document content"
                self.source = "mock.txt"
                self.document_type = "text"
                self.embedding = [0.1, 0.2, 0.3, 0.4]
                self.embedding_model = "test-model"
        
        mock_doc = MockIngestDocument()
        
        # Convert to DocumentSchema
        document = DocumentSchema.from_ingest_document(mock_doc)
        
        # Check embedding fields
        self.assertEqual(document.embedding, mock_doc.embedding)
        self.assertEqual(document.embedding_model, mock_doc.embedding_model)
    
    def test_from_ingest_document_with_chunks(self):
        """Test from_ingest_document method with chunks."""
        # Create a mock IngestDocument with chunks
        class MockChunk:
            def __init__(self, start_offset, end_offset, content):
                self.start_offset = start_offset
                self.end_offset = end_offset
                self.content = content
                self.chunk_type = "text"
        
        class MockIngestDocument:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.content = "Mock document content"
                self.source = "mock.txt"
                self.document_type = "text"
                self.chunks = [
                    MockChunk(0, 10, "Chunk 1"),
                    MockChunk(11, 20, "Chunk 2")
                ]
        
        mock_doc = MockIngestDocument()
        
        # Convert to DocumentSchema
        document = DocumentSchema.from_ingest_document(mock_doc)
        
        # Check chunks
        self.assertEqual(len(document.chunks), 2)
        self.assertEqual(document.chunks[0].start_offset, 0)
        self.assertEqual(document.chunks[0].end_offset, 10)
        self.assertEqual(document.chunks[0].parent_id, mock_doc.id)
        self.assertEqual(document.chunks[0].chunk_type, "text")


class TestDatasetSchemaAdditional(unittest.TestCase):
    """Additional test cases for DatasetSchema model."""
    
    def setUp(self):
        """Set up test data."""
        self.document1 = DocumentSchema.model_validate({
            "id": "doc1",
            "content": "Document 1 content",
            "source": "source1.txt",
            "document_type": "text"
        })
        
        self.document2 = DocumentSchema.model_validate({
            "id": "doc2",
            "content": "Document 2 content",
            "source": "source2.txt",
            "document_type": "text"
        })
        
        self.relation = DocumentRelationSchema.model_validate({
            "source_id": "doc1",
            "target_id": "doc2",
            "relation_type": RelationType.REFERENCES,
            "weight": 0.5
        })
    
    def test_from_ingest_dataset_minimal(self):
        """Test from_ingest_dataset method with minimal data."""
        # Create a mock IngestDataset
        class MockIngestDataset:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.name = "Mock Dataset"
                self.description = "Mock dataset description"
                self.documents = {}
                self.relations = []
                self.metadata = {}
        
        mock_dataset = MockIngestDataset()
        
        # Convert to DatasetSchema
        dataset = DatasetSchema.from_ingest_dataset(mock_dataset)
        
        # Check fields
        self.assertEqual(dataset.id, mock_dataset.id)
        self.assertEqual(dataset.name, mock_dataset.name)
        self.assertEqual(dataset.description, mock_dataset.description)
        self.assertEqual(len(dataset.documents), 0)
        self.assertEqual(len(dataset.relations), 0)
    
    def test_from_ingest_dataset_with_documents(self):
        """Test from_ingest_dataset method with documents."""
        # Create mock IngestDocuments
        class MockIngestDocument:
            def __init__(self, doc_id, content, source):
                self.id = doc_id
                self.content = content
                self.source = source
                self.document_type = "text"
        
        # Create a mock IngestDataset with documents
        class MockIngestDataset:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.name = "Mock Dataset"
                self.description = "Mock dataset description"
                self.documents = {
                    "doc1": MockIngestDocument("doc1", "Document 1 content", "source1.txt"),
                    "doc2": MockIngestDocument("doc2", "Document 2 content", "source2.txt")
                }
                self.relations = []
                self.metadata = {}
        
        mock_dataset = MockIngestDataset()
        
        # Convert to DatasetSchema
        dataset = DatasetSchema.from_ingest_dataset(mock_dataset)
        
        # Check documents
        self.assertEqual(len(dataset.documents), 2)
        self.assertIn("doc1", dataset.documents)
        self.assertIn("doc2", dataset.documents)
        self.assertEqual(dataset.documents["doc1"].content, "Document 1 content")
        self.assertEqual(dataset.documents["doc2"].content, "Document 2 content")
    
    def test_from_ingest_dataset_with_relations(self):
        """Test from_ingest_dataset method with relations."""
        # Create mock IngestDocuments
        class MockIngestDocument:
            def __init__(self, doc_id, content, source):
                self.id = doc_id
                self.content = content
                self.source = source
                self.document_type = "text"
        
        # Create mock IngestRelation
        class MockIngestRelation:
            def __init__(self, source_id, target_id, relation_type, weight):
                self.source_id = source_id
                self.target_id = target_id
                self.relation_type = relation_type
                self.weight = weight
                self.bidirectional = False
        
        # Create a mock IngestDataset with documents and relations
        class MockIngestDataset:
            def __init__(self):
                self.id = str(uuid.uuid4())
                self.name = "Mock Dataset"
                self.description = "Mock dataset description"
                self.documents = {
                    "doc1": MockIngestDocument("doc1", "Document 1 content", "source1.txt"),
                    "doc2": MockIngestDocument("doc2", "Document 2 content", "source2.txt")
                }
                self.relations = [
                    MockIngestRelation("doc1", "doc2", RelationType.REFERENCES, 0.5)
                ]
                self.metadata = {}
        
        mock_dataset = MockIngestDataset()
        
        # Convert to DatasetSchema
        dataset = DatasetSchema.from_ingest_dataset(mock_dataset)
        
        # Check relations
        self.assertEqual(len(dataset.relations), 1)
        self.assertEqual(dataset.relations[0].source_id, "doc1")
        self.assertEqual(dataset.relations[0].target_id, "doc2")
        self.assertEqual(dataset.relations[0].relation_type, RelationType.REFERENCES)
        self.assertEqual(dataset.relations[0].weight, 0.5)


if __name__ == "__main__":
    unittest.main()
