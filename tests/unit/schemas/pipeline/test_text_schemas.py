"""
Unit tests for the text pipeline schemas in the HADES-PathRAG system.

Tests text pipeline configuration, chunking strategies, and processing result schemas.
"""

import unittest
from typing import Dict, Any

from pydantic import ValidationError

from src.schemas.common.enums import DocumentType, ProcessingStage
from src.schemas.pipeline.text import (
    ChunkingStrategy,
    TextPipelineConfigSchema,
    TextPipelineResultSchema
)


class TestChunkingStrategy(unittest.TestCase):
    """Test the ChunkingStrategy enumeration."""
    
    def test_chunking_strategies(self):
        """Test all defined chunking strategies."""
        expected_strategies = [
            "fixed_size", "semantic", "recursive", 
            "overlap", "code_aware", "chonky"
        ]
        
        # Check that all expected strategies are defined
        for strategy in expected_strategies:
            self.assertTrue(hasattr(ChunkingStrategy, strategy.upper()))
            self.assertEqual(getattr(ChunkingStrategy, strategy.upper()).value, strategy)
        
        # Check total number of strategies
        self.assertEqual(len(ChunkingStrategy), len(expected_strategies))


class TestTextPipelineConfigSchema(unittest.TestCase):
    """Test the TextPipelineConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that TextPipelineConfigSchema can be instantiated with required attributes."""
        # Test minimal config
        config = TextPipelineConfigSchema(
            name="text-pipeline",
            input_dir="/path/to/input"
        )
        
        self.assertEqual(config.name, "text-pipeline")
        self.assertEqual(config.input_dir, "/path/to/input")
        self.assertIsNone(config.output_dir)  # default value
        self.assertEqual(config.file_types, ["*.txt", "*.md", "*.pdf"])  # default value
        self.assertTrue(config.recursive)  # default value
        self.assertEqual(config.exclude_patterns, [])  # default value
        self.assertEqual(config.max_file_size_mb, 10.0)  # default value
        self.assertEqual(config.encoding, "utf-8")  # default value
        self.assertFalse(config.remove_stopwords)  # default value
        self.assertTrue(config.clean_html)  # default value
        self.assertIsNone(config.language)  # default value
        self.assertEqual(config.chunk_strategy, ChunkingStrategy.SEMANTIC)  # default value
        self.assertEqual(config.chunk_size, 1000)  # default value
        self.assertEqual(config.chunk_overlap, 200)  # default value
        self.assertEqual(config.embedding_model, "modernbert")  # default value
        self.assertTrue(config.normalize_embeddings)  # default value
        self.assertEqual(config.batch_size, 32)  # default value
        self.assertTrue(config.store_embeddings)  # default value
        self.assertTrue(config.store_raw_documents)  # default value
        self.assertTrue(config.store_chunks)  # default value
        self.assertIsNone(config.db_connection_string)  # default value
        self.assertIsNone(config.db_name)  # default value
        self.assertFalse(config.create_new_collections)  # default value
        
        # Test with all attributes
        config = TextPipelineConfigSchema(
            name="full-text-pipeline",
            input_dir="/path/to/input",
            output_dir="/path/to/output",
            file_types=["*.txt", "*.md"],
            recursive=False,
            exclude_patterns=["*.tmp", "*/temp/*"],
            max_file_size_mb=5.0,
            encoding="latin-1",
            remove_stopwords=True,
            clean_html=False,
            language="en",
            chunk_strategy=ChunkingStrategy.CODE_AWARE,
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="sentence-transformers",
            normalize_embeddings=False,
            batch_size=16,
            store_embeddings=True,
            store_raw_documents=False,
            store_chunks=True,
            db_connection_string="http://localhost:8529",
            db_name="hades_pathrag",
            create_new_collections=True
        )
        
        self.assertEqual(config.name, "full-text-pipeline")
        self.assertEqual(config.input_dir, "/path/to/input")
        self.assertEqual(config.output_dir, "/path/to/output")
        self.assertEqual(config.file_types, ["*.txt", "*.md"])
        self.assertFalse(config.recursive)
        self.assertEqual(config.exclude_patterns, ["*.tmp", "*/temp/*"])
        self.assertEqual(config.max_file_size_mb, 5.0)
        self.assertEqual(config.encoding, "latin-1")
        self.assertTrue(config.remove_stopwords)
        self.assertFalse(config.clean_html)
        self.assertEqual(config.language, "en")
        self.assertEqual(config.chunk_strategy, ChunkingStrategy.CODE_AWARE)
        self.assertEqual(config.chunk_size, 500)
        self.assertEqual(config.chunk_overlap, 100)
        self.assertEqual(config.embedding_model, "sentence-transformers")
        self.assertFalse(config.normalize_embeddings)
        self.assertEqual(config.batch_size, 16)
        self.assertTrue(config.store_embeddings)
        self.assertFalse(config.store_raw_documents)
        self.assertTrue(config.store_chunks)
        self.assertEqual(config.db_connection_string, "http://localhost:8529")
        self.assertEqual(config.db_name, "hades_pathrag")
        self.assertTrue(config.create_new_collections)
    
    def test_positive_int_validation(self):
        """Test validation of positive integer values."""
        # Test chunk_size validation
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                chunk_size=0
            )
        
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                chunk_size=-10
            )
        
        # Test chunk_overlap validation
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                chunk_overlap=0
            )
        
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                chunk_overlap=-10
            )
        
        # Test batch_size validation
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                batch_size=0
            )
        
        with self.assertRaises(ValidationError):
            TextPipelineConfigSchema(
                name="test",
                input_dir="/path/to/input",
                batch_size=-10
            )


class TestTextPipelineResultSchema(unittest.TestCase):
    """Test the TextPipelineResultSchema functionality."""
    
    def test_result_instantiation(self):
        """Test that TextPipelineResultSchema can be instantiated with required attributes."""
        # Test minimal result
        result = TextPipelineResultSchema(
            document_id="doc123",
            source="/path/to/document.txt",
            document_type=DocumentType.TEXT
        )
        
        self.assertEqual(result.document_id, "doc123")
        self.assertEqual(result.source, "/path/to/document.txt")
        self.assertEqual(result.document_type, DocumentType.TEXT)
        self.assertEqual(result.processing_stage, ProcessingStage.RAW)  # default value
        self.assertTrue(result.success)  # default value
        self.assertIsNone(result.error)  # default value
        self.assertEqual(result.chunks, [])  # default value
        self.assertEqual(result.metadata, {})  # default value
        
        # Test with all attributes
        chunk1: Dict[str, Any] = {
            "id": "chunk1",
            "content": "Chunk 1 content",
            "start_offset": 0,
            "end_offset": 100
        }
        
        chunk2: Dict[str, Any] = {
            "id": "chunk2",
            "content": "Chunk 2 content",
            "start_offset": 100,
            "end_offset": 200
        }
        
        result = TextPipelineResultSchema(
            document_id="doc456",
            source="/path/to/document.pdf",
            document_type=DocumentType.PDF,
            processing_stage=ProcessingStage.CHUNKED,
            success=True,
            chunks=[chunk1, chunk2],
            metadata={"pages": 5, "author": "Test Author"}
        )
        
        self.assertEqual(result.document_id, "doc456")
        self.assertEqual(result.source, "/path/to/document.pdf")
        self.assertEqual(result.document_type, DocumentType.PDF)
        self.assertEqual(result.processing_stage, ProcessingStage.CHUNKED)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.chunks[0], chunk1)
        self.assertEqual(result.chunks[1], chunk2)
        self.assertEqual(result.metadata, {"pages": 5, "author": "Test Author"})
    
    def test_error_consistency(self):
        """Test error state consistency validation."""
        # Test with error but success=True (should be auto-corrected to success=False)
        result = TextPipelineResultSchema(
            document_id="doc123",
            source="/path/to/document.txt",
            document_type=DocumentType.TEXT,
            error="Failed to process document",
            success=True
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Failed to process document")
        
        # Test with error and success=False (should remain the same)
        result = TextPipelineResultSchema(
            document_id="doc123",
            source="/path/to/document.txt",
            document_type=DocumentType.TEXT,
            error="Failed to process document",
            success=False
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Failed to process document")
        
        # Test without error and success=True (should remain the same)
        result = TextPipelineResultSchema(
            document_id="doc123",
            source="/path/to/document.txt",
            document_type=DocumentType.TEXT,
            success=True
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
