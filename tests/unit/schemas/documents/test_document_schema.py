"""
Unit tests for the document schemas in the HADES-PathRAG system.

Tests document schema functionality including validation, relationships, and
conversion between different document formats.
"""

import unittest
from datetime import datetime
import uuid
import numpy as np
from pydantic import ValidationError

from src.schemas.documents.base import DocumentSchema, ChunkMetadata
from src.schemas.common.enums import DocumentType, SchemaVersion


class TestDocumentSchema(unittest.TestCase):
    """Test the DocumentSchema functionality."""
    
    def test_document_instantiation(self):
        """Test that DocumentSchema can be instantiated with required attributes."""
        # Test minimal document
        doc = DocumentSchema(
            id="test_doc_1",
            content="This is test content",
            source="test_file.txt",
            document_type=DocumentType.TEXT
        )
        
        self.assertEqual(doc.id, "test_doc_1")
        self.assertEqual(doc.content, "This is test content")
        self.assertEqual(doc.source, "test_file.txt")
        self.assertEqual(doc.document_type, DocumentType.TEXT)
        self.assertEqual(doc.schema_version, SchemaVersion.V2)
        
        # Check default values
        self.assertIsNone(doc.embedding)
        self.assertEqual(doc.chunks, [])
        self.assertEqual(doc.tags, [])
        self.assertIsNotNone(doc.created_at)
        self.assertIsNotNone(doc.updated_at)
        
        # Check title derived from source
        self.assertEqual(doc.title, "test_file.txt")
    
    def test_document_type_validation(self):
        """Test document type validation."""
        # Valid types
        for doc_type in DocumentType:
            doc = DocumentSchema(
                id="test_doc",
                content="Test content",
                source="test.txt",
                document_type=doc_type
            )
            self.assertEqual(doc.document_type, doc_type)
        
        # String values should be converted to enum
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type="text"
        )
        self.assertEqual(doc.document_type, DocumentType.TEXT)
        
        # Invalid type
        with self.assertRaises(ValidationError):
            DocumentSchema(
                id="test_doc",
                content="Test content",
                source="test.txt",
                document_type="invalid_type"
            )
    
    def test_id_validation(self):
        """Test ID validation and auto-generation."""
        # Explicit ID
        doc = DocumentSchema(
            id="test_doc_id",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        self.assertEqual(doc.id, "test_doc_id")
    
        # Auto-generated ID when not provided
        doc = DocumentSchema(
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        self.assertIsNotNone(doc.id)
        self.assertTrue(len(doc.id) > 0)
        
        # Auto-generated ID when empty string is provided
        doc = DocumentSchema(
            id="",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        self.assertIsNotNone(doc.id)
        self.assertTrue(len(doc.id) > 0)
    
    def test_timestamps(self):
        """Test timestamp handling."""
        # Default timestamps
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT
        )
        self.assertIsNotNone(doc.created_at)
        self.assertIsNotNone(doc.updated_at)
        self.assertEqual(doc.created_at, doc.updated_at)
        
        # Explicit timestamps
        now = datetime.now()
        earlier = datetime(2020, 1, 1, 12, 0, 0)
        
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT,
            created_at=earlier,
            updated_at=now
        )
        self.assertEqual(doc.created_at, earlier)
        self.assertEqual(doc.updated_at, now)
        
        # Only created_at provided
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT,
            created_at=earlier
        )
        self.assertEqual(doc.created_at, earlier)
        self.assertEqual(doc.updated_at, earlier)  # updated_at should match created_at
    
    def test_with_chunks(self):
        """Test document with chunks."""
        chunks = [
            ChunkMetadata(
                start_offset=0,
                end_offset=100,
                chunk_index=0,
                parent_id="test_doc",
                chunk_type="text"
            ),
            ChunkMetadata(
                start_offset=101,
                end_offset=200,
                chunk_index=1,
                parent_id="test_doc",
                chunk_type="text",
                context_before="Previous content"
            )
        ]
        
        doc = DocumentSchema(
            id="test_doc",
            content="This is test content that spans multiple chunks.",
            source="test_file.txt",
            document_type=DocumentType.TEXT,
            chunks=chunks
        )
        
        self.assertEqual(len(doc.chunks), 2)
        self.assertEqual(doc.chunks[0].start_offset, 0)
        self.assertEqual(doc.chunks[0].end_offset, 100)
        self.assertEqual(doc.chunks[1].start_offset, 101)
        self.assertEqual(doc.chunks[1].end_offset, 200)
        self.assertEqual(doc.chunks[1].context_before, "Previous content")
    
    def test_embedding(self):
        """Test document with embedding."""
        # List embedding
        embedding = [0.1, 0.2, 0.3, 0.4]
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT,
            embedding=embedding,
            embedding_model="test-model"
        )
        
        self.assertEqual(doc.embedding, embedding)
        self.assertEqual(doc.embedding_model, "test-model")
        
        # NumPy array embedding
        np_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT,
            embedding=np_embedding,
            embedding_model="test-model"
        )
        
        # The embedding should be stored as the original numpy array
        self.assertIsInstance(doc.embedding, np.ndarray)
        np.testing.assert_array_equal(doc.embedding, np_embedding)
        
        # But when converted to dict, it should be a list
        doc_dict = doc.model_dump_safe()
        self.assertIsInstance(doc_dict["embedding"], list)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = DocumentSchema(
            id="test_doc",
            content="Test content",
            source="test.txt",
            document_type=DocumentType.TEXT,
            title="Test Document",
            metadata={"key": "value"},
            tags=["tag1", "tag2"]
        )
        
        doc_dict = doc.to_dict()
        
        # Check all fields were preserved
        self.assertEqual(doc_dict["id"], "test_doc")
        self.assertEqual(doc_dict["content"], "Test content")
        self.assertEqual(doc_dict["source"], "test.txt")
        self.assertEqual(doc_dict["document_type"], "text")
        self.assertEqual(doc_dict["title"], "Test Document")
        self.assertEqual(doc_dict["metadata"], {"key": "value"})
        self.assertEqual(doc_dict["tags"], ["tag1", "tag2"])
    
    def test_from_ingest_document(self):
        """Test conversion from IngestDocument."""
        # Simulate an IngestDocument dict
        ingest_doc = {
            "id": "ingest_doc_1",
            "content": "Ingest document content",
            "source": "ingest.txt",
            "type": "markdown",
            "metadata": {"source_system": "test_ingestion"},
            "embeddings": [0.1, 0.2, 0.3]
        }
        
        # Convert to DocumentSchema
        doc = DocumentSchema.from_ingest_document(ingest_doc)
        
        # Check conversion
        self.assertEqual(doc.id, "ingest_doc_1")
        self.assertEqual(doc.content, "Ingest document content")
        self.assertEqual(doc.source, "ingest.txt")
        self.assertEqual(doc.document_type, DocumentType.MARKDOWN)
        self.assertEqual(doc.metadata, {"source_system": "test_ingestion"})


if __name__ == "__main__":
    unittest.main()
