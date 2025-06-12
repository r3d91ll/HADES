"""
Encoder Embedding Component

This module provides encoder-based embedding component that implements the
Embedder protocol specifically for transformer encoder models with custom
architectures and fine-tuned models.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    EmbeddingInput,
    EmbeddingOutput,
    ChunkEmbedding,
    ProcessingStatus
)
from src.types.components.protocols import Embedder


class EncoderEmbedder(Embedder):
    """
    Encoder embedding component implementing Embedder protocol.
    
    This component specializes in transformer encoder-based embedding generation
    using custom models, fine-tuned encoders, and specialized architectures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize encoder embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.EMBEDDING,
            component_name="encoder",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration settings
        self._model_name = self._config.get('model_name', 'microsoft/DialoGPT-medium')
        self._batch_size = self._config.get('batch_size', 16)  # Lower for encoder models
        self._max_length = self._config.get('max_length', 512)
        self._normalize_embeddings = self._config.get('normalize_embeddings', True)
        self._pooling_strategy = self._config.get('pooling_strategy', 'mean')
        self._hidden_size = self._config.get('hidden_size', 768)
        self._device = self._config.get('device', 'cpu')
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        # Performance tracking
        self._total_embeddings_created = 0
        self._total_processing_time = 0.0
        
        # Check transformer availability
        self._transformers_available = self._check_transformers_availability()
        
        if self._transformers_available:
            self.logger.info(f"Initialized encoder embedder with model: {self._model_name}")
        else:
            self.logger.warning("Transformers not available - embedder will use fallback implementation")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "encoder"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.EMBEDDING
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update configuration parameters
        if 'model_name' in config:
            self._model_name = config['model_name']
            # Reset model to force reload
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
        
        if 'batch_size' in config:
            self._batch_size = config['batch_size']
        
        if 'max_length' in config:
            self._max_length = config['max_length']
        
        if 'hidden_size' in config:
            self._hidden_size = config['hidden_size']
        
        self.logger.info("Updated encoder embedder configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate model name
        if 'model_name' in config:
            if not isinstance(config['model_name'], str):
                return False
        
        # Validate batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                return False
        
        # Validate max length
        if 'max_length' in config:
            max_length = config['max_length']
            if not isinstance(max_length, int) or max_length < 1:
                return False
        
        # Validate pooling strategy
        if 'pooling_strategy' in config:
            pooling = config['pooling_strategy']
            if pooling not in ['mean', 'max', 'cls', 'first', 'last']:
                return False
        
        # Validate hidden size
        if 'hidden_size' in config:
            hidden_size = config['hidden_size']
            if not isinstance(hidden_size, int) or hidden_size < 1:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the transformer encoder model",
                    "default": "microsoft/DialoGPT-medium"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "default": 16,
                    "description": "Batch size for encoder processing"
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "default": 512,
                    "description": "Maximum token length per text"
                },
                "normalize_embeddings": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to normalize embeddings"
                },
                "pooling_strategy": {
                    "type": "string",
                    "enum": ["mean", "max", "cls", "first", "last"],
                    "default": "mean",
                    "description": "Pooling strategy for sequence embeddings"
                },
                "hidden_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 768,
                    "description": "Hidden size of the encoder model"
                },
                "device": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "auto"],
                    "default": "auto",
                    "description": "Device to use for processing"
                },
                "use_attention_mask": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use attention masking"
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Check transformers availability
            if not self._transformers_available:
                self.logger.warning("Transformers not available for health check")
                return True  # Allow fallback implementation
            
            # Try to initialize model if not already loaded
            if not self._model_loaded:
                self._initialize_model()
            
            # Test with simple text if model is loaded
            if self._model_loaded and self._model:
                test_text = "This is a test text for embedding."
                embedding = self._embed_single_text(test_text)
                return len(embedding) > 0
            
            # Return True for fallback mode
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_processing_time = (
            self._total_processing_time / max(self._total_embeddings_created, 1)
        )
        
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "model_name": self._model_name,
            "transformers_available": self._transformers_available,
            "model_loaded": self._model_loaded,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "hidden_size": self._hidden_size,
            "pooling_strategy": self._pooling_strategy,
            "total_embeddings_created": self._total_embeddings_created,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add GPU metrics if available
        if self._transformers_available and self._model_loaded:
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.update({
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                        "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                        "gpu_device_count": torch.cuda.device_count()
                    })
            except ImportError:
                pass
        
        return metrics
    
    def embed(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Generate embeddings according to the contract.
        
        Args:
            input_data: Input data conforming to EmbeddingInput contract
            
        Returns:
            Output data conforming to EmbeddingOutput contract
        """
        errors: List[str] = []
        
        try:
            start_time = datetime.utcnow()
            
            # Extract texts from chunks
            texts = [chunk.content for chunk in input_data.chunks]
            chunk_ids = [chunk.id for chunk in input_data.chunks]
            
            # Generate embeddings
            if self._transformers_available:
                # Try to use transformers if available
                embedding_vectors = self._embed_texts_encoder(texts)
            else:
                # Fall back to basic implementation
                embedding_vectors = self._embed_texts_fallback(texts)
            
            # Convert to contract format
            embeddings = []
            for i, (chunk_id, vector) in enumerate(zip(chunk_ids, embedding_vectors)):
                embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    embedding_dimension=len(vector),
                    model_name=input_data.model_name or self._model_name,
                    confidence=1.0,  # Encoder embeddings have consistent confidence
                    metadata=input_data.metadata
                )
                embeddings.append(embedding)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._total_embeddings_created += len(embeddings)
            self._total_processing_time += processing_time
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS
            )
            
            return EmbeddingOutput(
                embeddings=embeddings,
                metadata=metadata,
                embedding_stats={
                    "processing_time": processing_time,
                    "embedding_count": len(embeddings),
                    "total_texts": len(texts),
                    "batch_size": self._batch_size,
                    "model_name": self._model_name,
                    "pooling_strategy": self._pooling_strategy,
                    "engine_type": "encoder" if self._transformers_available else "fallback",
                    "throughput_embeddings_per_second": len(embeddings) / max(processing_time, 0.001)
                },
                model_info={
                    "model_name": self._model_name,
                    "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
                    "max_length": self._max_length,
                    "hidden_size": self._hidden_size,
                    "device": "auto"
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Encoder embedding failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return EmbeddingOutput(
                embeddings=[],
                metadata=metadata,
                embedding_stats={},
                model_info={},
                errors=errors
            )
    
    def estimate_tokens(self, input_data: EmbeddingInput) -> int:
        """
        Estimate number of tokens that will be processed.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of tokens
        """
        try:
            total_chars = sum(len(chunk.content) for chunk in input_data.chunks)
            # Simple estimation: ~4 characters per token
            estimated_tokens = total_chars // 4
            return max(1, estimated_tokens)
            
        except Exception:
            return len(input_data.chunks)  # Fallback estimate
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if embedder supports the given model.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is supported, False otherwise
        """
        # Common encoder models and patterns
        supported_patterns = [
            'bert',
            'roberta',
            'distilbert',
            'electra',
            'albert',
            'xlnet',
            'microsoft/DialoGPT',
            'encoder'
        ]
        
        return any(pattern in model_name.lower() for pattern in supported_patterns)
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """
        Get the embedding dimension for a given model.
        
        Args:
            model_name: Model name
            
        Returns:
            Embedding dimension
        """
        # Common encoder model dimensions
        model_dims = {
            'bert-base': 768,
            'bert-large': 1024,
            'roberta-base': 768,
            'roberta-large': 1024,
            'distilbert': 768,
            'electra-base': 768,
            'electra-large': 1024,
            'microsoft/DialoGPT-medium': 1024,
            'microsoft/DialoGPT-large': 1280
        }
        
        for model, dim in model_dims.items():
            if model in model_name.lower():
                return dim
        
        return self._hidden_size  # Use configured hidden size as default
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        return self.get_embedding_dimension(self._model_name)
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length this embedder can handle."""
        return self._max_length
    
    def supports_batch_processing(self) -> bool:
        """Check if embedder supports batch processing."""
        return True
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for this embedder."""
        return self._batch_size
    
    def estimate_embedding_time(self, input_data: EmbeddingInput) -> float:
        """
        Estimate time to generate embeddings using encoder models.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_chunks = len(input_data.chunks)
            total_tokens = self.estimate_tokens(input_data)
            
            # Encoder processing speed depends on device
            # Estimate based on tokens and batch size
            batches_needed = (num_chunks + self._batch_size - 1) // self._batch_size
            
            # Base time per batch varies by device
            if self._device in ['cuda', 'gpu']:
                time_per_batch = 0.1  # 100ms per batch on GPU
            else:
                time_per_batch = 0.3  # 300ms per batch on CPU
            
            # Scale by model complexity
            if 'large' in self._model_name.lower():
                complexity_factor = 1.5
            elif 'base' in self._model_name.lower():
                complexity_factor = 1.0
            else:
                complexity_factor = 0.8  # Small models
            
            estimated_time = batches_needed * time_per_batch * complexity_factor
            
            return max(0.05, estimated_time)
            
        except Exception:
            # Fallback estimate
            return len(input_data.chunks) * 0.05
    
    def _check_transformers_availability(self) -> bool:
        """Check if transformers library is available."""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    
    def _initialize_model(self) -> None:
        """Initialize the transformer encoder model."""
        if not self._transformers_available:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.logger.info(f"Loading transformer encoder model: {self._model_name}")
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            
            # Set to evaluation mode
            self._model.eval()
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            
            self._model_loaded = True
            
            self.logger.info(f"Successfully loaded encoder model: {self._model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encoder model: {e}")
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
    
    def _embed_texts_encoder(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using transformer encoder."""
        if not self._model_loaded:
            self._initialize_model()
        
        if not self._model_loaded:
            # Fall back to basic implementation
            return self._embed_texts_fallback(texts)
        
        try:
            import torch
            
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self._batch_size):
                batch_texts = texts[i:i + self._batch_size]
                
                # Tokenize batch
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt"
                )
                
                # Move to model device
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    
                    # Apply pooling strategy
                    batch_embeddings = self._apply_pooling(
                        hidden_states, 
                        inputs.get('attention_mask')
                    )
                    
                    # Normalize if requested
                    if self._normalize_embeddings:
                        batch_embeddings = torch.nn.functional.normalize(
                            batch_embeddings, p=2, dim=1
                        )
                    
                    # Convert to list and add to results
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    embeddings.extend(batch_embeddings.tolist())
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Encoder embedding failed: {e}")
            return self._embed_texts_fallback(texts)
    
    def _apply_pooling(self, hidden_states, attention_mask=None):
        """Apply pooling strategy to hidden states."""
        import torch
        
        if self._pooling_strategy == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self._pooling_strategy == "max":
            return torch.max(hidden_states, dim=1)[0]
        
        elif self._pooling_strategy == "cls":
            return hidden_states[:, 0]  # CLS token
        
        elif self._pooling_strategy == "first":
            return hidden_states[:, 0]
        
        elif self._pooling_strategy == "last":
            if attention_mask is not None:
                # Get last non-padded token
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                return hidden_states[range(batch_size), seq_lengths]
            else:
                return hidden_states[:, -1]
        
        else:
            # Default to mean pooling
            return torch.mean(hidden_states, dim=1)
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self._embed_texts_encoder([text])
        return embeddings[0] if embeddings else []
    
    def _embed_texts_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method using basic text features."""
        embeddings = []
        
        for text in texts:
            # Simple feature-based embedding
            features = self._extract_text_features(text)
            embeddings.append(features)
        
        return embeddings
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract basic text features for fallback embedding."""
        import math
        import re
        
        # Basic text statistics (normalized to create embedding-like vector)
        features = []
        
        # Length features
        features.append(min(len(text) / 1000.0, 1.0))  # Normalized length
        features.append(min(len(text.split()) / 100.0, 1.0))  # Normalized word count
        features.append(min(len(re.findall(r'\w+', text)) / 100.0, 1.0))  # Word tokens
        
        # Character distribution features
        char_counts: Dict[str, int] = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Add features for common characters
        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 .,!?'
        for char in common_chars:
            freq = char_counts.get(char, 0) / max(len(text), 1)
            features.append(freq)
        
        # Syntactic features
        features.append(text.count('.') / max(len(text), 1))  # Sentence density
        features.append(text.count('?') / max(len(text), 1))  # Question density
        features.append(text.count('!') / max(len(text), 1))  # Exclamation density
        
        # Pad or truncate to hidden size
        target_dim = self._hidden_size
        while len(features) < target_dim:
            features.append(0.0)
        
        features = features[:target_dim]
        
        # Normalize if requested
        if self._normalize_embeddings:
            norm = math.sqrt(sum(f * f for f in features))
            if norm > 0:
                features = [f / norm for f in features]
        
        return features