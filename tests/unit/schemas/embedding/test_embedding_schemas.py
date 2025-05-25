"""
Unit tests for the embedding schemas in the HADES-PathRAG system.

Tests embedding schema functionality including validation, configuration models,
and adapter interfaces.
"""

import unittest
import numpy as np
from pydantic import ValidationError

from src.schemas.embedding.models import (
    EmbeddingModelType, 
    EmbeddingConfig, 
    EmbeddingResult, 
    BatchEmbeddingRequest,
    BatchEmbeddingResult
)
from src.schemas.embedding.adapters import (
    AdapterType,
    BaseAdapterConfig,
    HuggingFaceAdapterConfig,
    SentenceTransformersAdapterConfig,
    ModernBERTAdapterConfig,
    AdapterResult
)


class TestEmbeddingConfigSchema(unittest.TestCase):
    """Test the EmbeddingConfig schema."""
    
    def test_embedding_config_instantiation(self):
        """Test that EmbeddingConfig can be instantiated with required attributes."""
        # Test minimal config
        config = EmbeddingConfig(
            model_name="test-model",
            model_type=EmbeddingModelType.TRANSFORMER
        )
        
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.model_type, EmbeddingModelType.TRANSFORMER)
        self.assertEqual(config.batch_size, 32)  # default value
        self.assertEqual(config.max_length, 512)  # default value
        self.assertTrue(config.normalize)  # default value
        
        # Test with custom values
        config = EmbeddingConfig(
            model_name="custom-model",
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            batch_size=16,
            max_length=256,
            normalize=False,
            device="cuda:0",
            cache_dir="/tmp/cache"
        )
        
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.model_type, EmbeddingModelType.SENTENCE_TRANSFORMER)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.max_length, 256)
        self.assertFalse(config.normalize)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.cache_dir, "/tmp/cache")
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch size
        config = EmbeddingConfig(
            model_name="test-model",
            model_type=EmbeddingModelType.TRANSFORMER,
            batch_size=10
        )
        self.assertEqual(config.batch_size, 10)
        
        # Invalid batch size
        with self.assertRaises(ValidationError):
            EmbeddingConfig(
                model_name="test-model",
                model_type=EmbeddingModelType.TRANSFORMER,
                batch_size=0
            )
        
        with self.assertRaises(ValidationError):
            EmbeddingConfig(
                model_name="test-model",
                model_type=EmbeddingModelType.TRANSFORMER,
                batch_size=-5
            )


class TestEmbeddingResultSchema(unittest.TestCase):
    """Test the EmbeddingResult schema."""
    
    def test_embedding_result_instantiation(self):
        """Test that EmbeddingResult can be instantiated with required attributes."""
        # Test with list embedding
        result = EmbeddingResult(
            text="This is a test",
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="test-model"
        )
        
        self.assertEqual(result.text, "This is a test")
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.metadata, {})  # default value
        
        # Test with numpy array embedding
        np_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        result = EmbeddingResult(
            text="This is a test",
            embedding=np_embedding,
            model_name="test-model",
            metadata={"source": "test"}
        )
        
        self.assertEqual(result.text, "This is a test")
        np.testing.assert_array_equal(result.embedding, np_embedding)
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.metadata, {"source": "test"})
    
    def test_empty_embedding_validation(self):
        """Test that empty embeddings are not allowed."""
        # Empty list
        with self.assertRaises(ValidationError):
            EmbeddingResult(
                text="This is a test",
                embedding=[],
                model_name="test-model"
            )
        
        # Empty numpy array
        with self.assertRaises(ValidationError):
            EmbeddingResult(
                text="This is a test",
                embedding=np.array([]),
                model_name="test-model"
            )


class TestBatchEmbeddingSchema(unittest.TestCase):
    """Test the BatchEmbeddingRequest and BatchEmbeddingResult schemas."""
    
    def test_batch_request_instantiation(self):
        """Test that BatchEmbeddingRequest can be instantiated with required attributes."""
        # Test minimal request
        request = BatchEmbeddingRequest(
            texts=["Text 1", "Text 2", "Text 3"]
        )
        
        self.assertEqual(request.texts, ["Text 1", "Text 2", "Text 3"])
        self.assertIsNone(request.model_name)
        self.assertIsNone(request.config)
        self.assertEqual(request.metadata, {})  # default value
        
        # Test with all attributes
        config = EmbeddingConfig(
            model_name="test-model",
            model_type=EmbeddingModelType.TRANSFORMER
        )
        
        request = BatchEmbeddingRequest(
            texts=["Text 1", "Text 2", "Text 3"],
            model_name="custom-model",
            config=config,
            metadata={"source": "test"}
        )
        
        self.assertEqual(request.texts, ["Text 1", "Text 2", "Text 3"])
        self.assertEqual(request.model_name, "custom-model")
        self.assertEqual(request.config, config)
        self.assertEqual(request.metadata, {"source": "test"})
    
    def test_empty_texts_validation(self):
        """Test that empty texts list is not allowed."""
        with self.assertRaises(ValidationError):
            BatchEmbeddingRequest(texts=[])
    
    def test_batch_result_instantiation(self):
        """Test that BatchEmbeddingResult can be instantiated with required attributes."""
        # Create embedding results
        result1 = EmbeddingResult(
            text="Text 1",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model"
        )
        
        result2 = EmbeddingResult(
            text="Text 2",
            embedding=[0.4, 0.5, 0.6],
            model_name="test-model"
        )
        
        # Test batch result
        batch_result = BatchEmbeddingResult(
            embeddings=[result1, result2],
            model_name="test-model"
        )
        
        self.assertEqual(len(batch_result.embeddings), 2)
        self.assertEqual(batch_result.embeddings[0], result1)
        self.assertEqual(batch_result.embeddings[1], result2)
        self.assertEqual(batch_result.model_name, "test-model")
        self.assertEqual(batch_result.metadata, {})  # default value


class TestAdapterConfigSchemas(unittest.TestCase):
    """Test the adapter configuration schemas."""
    
    def test_base_adapter_config(self):
        """Test the BaseAdapterConfig schema."""
        # Test instantiation
        config = BaseAdapterConfig(
            adapter_type=AdapterType.HUGGINGFACE,
            model_name="test-model"
        )
        
        self.assertEqual(config.adapter_type, AdapterType.HUGGINGFACE)
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.batch_size, 32)  # default value
        self.assertEqual(config.metadata, {})  # default value
        
        # Test batch size validation
        with self.assertRaises(ValidationError):
            BaseAdapterConfig(
                adapter_type=AdapterType.HUGGINGFACE,
                model_name="test-model",
                batch_size=0
            )
    
    def test_huggingface_adapter_config(self):
        """Test the HuggingFaceAdapterConfig schema."""
        config = HuggingFaceAdapterConfig(
            model_name="bert-base-uncased",
            device="cuda:0",
            cache_dir="/tmp/cache",
            normalize=True,
            model_kwargs={"use_auth_token": False},
            encode_kwargs={"padding": True}
        )
        
        self.assertEqual(config.adapter_type, AdapterType.HUGGINGFACE)
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.cache_dir, "/tmp/cache")
        self.assertTrue(config.normalize)
        self.assertEqual(config.model_kwargs, {"use_auth_token": False})
        self.assertEqual(config.encode_kwargs, {"padding": True})
    
    def test_sentence_transformers_adapter_config(self):
        """Test the SentenceTransformersAdapterConfig schema."""
        config = SentenceTransformersAdapterConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            normalize=False,
            encode_kwargs={"show_progress_bar": True}
        )
        
        self.assertEqual(config.adapter_type, AdapterType.SENTENCE_TRANSFORMERS)
        self.assertEqual(config.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.normalize)
        self.assertEqual(config.encode_kwargs, {"show_progress_bar": True})
    
    def test_modernbert_adapter_config(self):
        """Test the ModernBERTAdapterConfig schema."""
        config = ModernBERTAdapterConfig(
            model_name="modernbert-large",
            api_key="test-api-key",
            api_base="https://custom-api.com",
            timeout=30.0
        )
        
        self.assertEqual(config.adapter_type, AdapterType.MODERNBERT)
        self.assertEqual(config.model_name, "modernbert-large")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.api_base, "https://custom-api.com")
        self.assertEqual(config.timeout, 30.0)
        self.assertTrue(config.normalize)  # default value


class TestAdapterResultSchema(unittest.TestCase):
    """Test the AdapterResult schema."""
    
    def test_adapter_result_instantiation(self):
        """Test that AdapterResult can be instantiated with required attributes."""
        # Test with minimal attributes
        result = AdapterResult(
            text_inputs=["Text 1", "Text 2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model_name="test-model",
            adapter_type=AdapterType.HUGGINGFACE
        )
        
        self.assertEqual(result.text_inputs, ["Text 1", "Text 2"])
        self.assertEqual(result.embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.adapter_type, AdapterType.HUGGINGFACE)
        self.assertEqual(result.metadata, {})  # default value
        
        # Test with numpy arrays
        np_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        result = AdapterResult(
            text_inputs=["Text 1", "Text 2"],
            embeddings=np_embeddings,
            model_name="test-model",
            adapter_type=AdapterType.SENTENCE_TRANSFORMERS,
            metadata={"source": "test"}
        )
        
        self.assertEqual(result.text_inputs, ["Text 1", "Text 2"])
        self.assertEqual(len(result.embeddings), 2)
        np.testing.assert_array_equal(result.embeddings[0], np_embeddings[0])
        np.testing.assert_array_equal(result.embeddings[1], np_embeddings[1])
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.adapter_type, AdapterType.SENTENCE_TRANSFORMERS)
        self.assertEqual(result.metadata, {"source": "test"})
    
    def test_embedding_length_validation(self):
        """Test that embeddings length must match text_inputs length."""
        # Valid case (lengths match)
        result = AdapterResult(
            text_inputs=["Text 1", "Text 2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model_name="test-model",
            adapter_type=AdapterType.HUGGINGFACE
        )
        
        # Invalid case (lengths don't match)
        with self.assertRaises(ValidationError):
            AdapterResult(
                text_inputs=["Text 1", "Text 2", "Text 3"],  # 3 inputs
                embeddings=[[0.1, 0.2], [0.3, 0.4]],  # but only 2 embeddings
                model_name="test-model",
                adapter_type=AdapterType.HUGGINGFACE
            )


if __name__ == "__main__":
    unittest.main()
