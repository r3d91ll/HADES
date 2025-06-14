"""
Chonky Chunking Component

This module provides the GPU-accelerated "chonky" chunking component that implements
the Chunker protocol. It uses modern transformer models and GPU acceleration for
high-throughput chunking operations.
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)
from src.types.components.protocols import Chunker


class ChonkyChunker(Chunker):
    """
    GPU-accelerated "chonky" chunking component implementing Chunker protocol.
    
    This component provides high-throughput chunking using transformer tokenizers
    and GPU acceleration for processing large volumes of text efficiently.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chonky chunker component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.CHUNKING,
            component_name="chonky",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._device = self._config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = self._config.get('batch_size', 32)
        self._tokenizer_model = self._config.get('tokenizer_model', 'bert-base-uncased')
        self._max_token_length = self._config.get('max_token_length', 512)
        self._chunk_overlap_tokens = self._config.get('chunk_overlap_tokens', 50)
        self._use_fast_tokenizer = self._config.get('use_fast_tokenizer', True)
        
        # Model components
        self._tokenizer = None
        self._model_loaded = False
        
        # Performance tracking
        self._total_chunks_created = 0
        self._total_processing_time = 0.0
        self._gpu_memory_peak = 0.0
        
        self.logger.info(f"Initialized chonky chunker with device: {self._device}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "chonky"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.CHUNKING
    
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
        if 'device' in config:
            self._device = config['device']
        
        if 'batch_size' in config:
            self._batch_size = config['batch_size']
        
        if 'tokenizer_model' in config:
            self._tokenizer_model = config['tokenizer_model']
            # Reset tokenizer to force reload with new model
            self._tokenizer = None
            self._model_loaded = False
        
        if 'max_token_length' in config:
            self._max_token_length = config['max_token_length']
        
        self.logger.info("Updated chonky chunker configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate device
        if 'device' in config:
            device = config['device']
            if not isinstance(device, str) or device not in ['cuda', 'cpu', 'auto']:
                return False
        
        # Validate batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                return False
        
        # Validate max token length
        if 'max_token_length' in config:
            max_tokens = config['max_token_length']
            if not isinstance(max_tokens, int) or max_tokens < 1:
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
                "device": {
                    "type": "string",
                    "enum": ["cuda", "cpu", "auto"],
                    "default": "auto",
                    "description": "Device to use for processing"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 256,
                    "default": 32,
                    "description": "Batch size for GPU processing"
                },
                "tokenizer_model": {
                    "type": "string",
                    "default": "bert-base-uncased",
                    "description": "Transformer model for tokenization"
                },
                "max_token_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "default": 512,
                    "description": "Maximum token length per chunk"
                },
                "chunk_overlap_tokens": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 50,
                    "description": "Token overlap between chunks"
                },
                "use_fast_tokenizer": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use fast tokenizer implementation"
                },
                "gpu_memory_fraction": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "Fraction of GPU memory to use"
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
            # Check if transformers library is available
            try:
                from transformers import AutoTokenizer
            except ImportError:
                self.logger.error("Transformers library not available")
                return False
            
            # Check device availability
            if self._device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available")
                return False
            
            # Initialize tokenizer if needed
            if not self._model_loaded:
                self._initialize_tokenizer()
            
            return self._tokenizer is not None
            
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
            float(self._total_processing_time) / max(int(self._total_chunks_created), 1)
        )
        
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "device": self._device,
            "batch_size": self._batch_size,
            "tokenizer_model": self._tokenizer_model,
            "max_token_length": self._max_token_length,
            "model_loaded": self._model_loaded,
            "total_chunks_created": self._total_chunks_created,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add GPU metrics if using CUDA
        if self._device == 'cuda' and torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "gpu_memory_peak_mb": self._gpu_memory_peak / (1024 * 1024),
                "gpu_device_name": torch.cuda.get_device_name()
            })
        
        return metrics
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk text according to the contract using GPU acceleration.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
            
        Raises:
            ChunkingError: If chunking fails
        """
        errors = []
        chunks: List[TextChunk] = []
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize tokenizer if needed
            if not self._model_loaded:
                self._initialize_tokenizer()
            
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            # Perform chunking
            chunks = self._perform_chunking(input_data)
            
            # Track GPU memory usage
            if self._device == 'cuda' and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                self._gpu_memory_peak = max(self._gpu_memory_peak, current_memory)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._total_chunks_created += len(chunks)
            self._total_processing_time += processing_time
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return ChunkingOutput(
                chunks=chunks,
                metadata=metadata,
                processing_stats={
                    "processing_time": processing_time,
                    "chunks_created": len(chunks),
                    "tokens_processed": sum(len(chunk.text.split()) for chunk in chunks),
                    "device_used": self._device,
                    "batch_size": self._batch_size,
                    "tokenizer_model": self._tokenizer_model,
                    "throughput_chunks_per_second": len(chunks) / max(processing_time, 0.001)
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Chonky chunking failed: {str(e)}"
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
            
            return ChunkingOutput(
                chunks=[],
                metadata=metadata,
                processing_stats={},
                errors=errors
            )
    
    def estimate_chunks(self, input_data: ChunkingInput) -> int:
        """
        Estimate number of chunks that will be generated.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of chunks
        """
        try:
            # Simple estimation based on text length and token limit
            text_length = len(input_data.text)
            
            # Rough estimation: 4 characters per token on average
            estimated_tokens = text_length // 4
            
            # Account for overlap
            effective_chunk_size = self._max_token_length - self._chunk_overlap_tokens
            
            if effective_chunk_size <= 0:
                return 1
            
            estimated_chunks = max(1, estimated_tokens // effective_chunk_size)
            
            return int(estimated_chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to estimate chunks: {e}")
            return 1
    
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if content type is supported, False otherwise
        """
        supported_types = ['text', 'markdown', 'code', 'document']
        return content_type.lower() in supported_types
    
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        # Convert token limit to approximate character count
        base_chars = self._max_token_length * 4  # 4 chars per token average
        
        # Adjust based on content type
        content_multipliers = {
            'text': 1.0,
            'markdown': 0.9,  # Slightly smaller due to formatting
            'code': 0.8,      # Smaller for code to preserve structure
            'document': 1.1    # Slightly larger for documents
        }
        
        multiplier = content_multipliers.get(content_type.lower(), 1.0)
        return int(base_chars * multiplier)
    
    def _initialize_tokenizer(self) -> None:
        """Initialize the transformer tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            self.logger.info(f"Loading tokenizer: {self._tokenizer_model}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_model,
                use_fast=self._use_fast_tokenizer
            )
            
            self._model_loaded = True
            self.logger.info(f"Successfully loaded tokenizer: {self._tokenizer_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            self._tokenizer = None
            self._model_loaded = False
    
    def _perform_chunking(self, input_data: ChunkingInput) -> List[TextChunk]:
        """Perform the actual chunking with GPU acceleration."""
        try:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            # Tokenize the input text
            tokens = self._tokenizer.encode(
                input_data.text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            if self._device == 'cuda':
                tokens = tokens.to(self._device)
            
            chunks: List[TextChunk] = []
            chunk_id = 0
            
            # Calculate effective chunk size (accounting for overlap)
            effective_chunk_size = self._max_token_length - self._chunk_overlap_tokens
            
            # Process tokens in overlapping windows
            start_idx = 0
            while start_idx < tokens.size(1):
                end_idx = min(start_idx + self._max_token_length, tokens.size(1))
                
                # Extract chunk tokens
                chunk_tokens = tokens[:, start_idx:end_idx]
                
                # Decode back to text
                chunk_text = self._tokenizer.decode(
                    chunk_tokens.squeeze().cpu(),
                    skip_special_tokens=True
                )
                
                # Create chunk
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    chunk_index=chunk_id,
                    metadata={
                        "tokenizer_model": self._tokenizer_model,
                        "device_used": self._device,
                        "token_count": chunk_tokens.size(1),
                        "processing_method": "gpu_accelerated"
                    }
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Move to next chunk with overlap
                start_idx += effective_chunk_size
                
                # Break if we've reached the end
                if end_idx >= tokens.size(1):
                    break
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            # Fallback to simple character-based chunking
            return self._fallback_chunking(input_data)
    
    def _fallback_chunking(self, input_data: ChunkingInput) -> List[TextChunk]:
        """Fallback chunking method using simple character-based approach."""
        try:
            text = input_data.text
            chunk_size = self.get_optimal_chunk_size('text')
            overlap = min(input_data.chunk_overlap, chunk_size // 4)
            
            chunks: List[TextChunk] = []
            chunk_id = 0
            start_idx = 0
            
            while start_idx < len(text):
                end_idx = min(start_idx + chunk_size, len(text))
                chunk_text = text[start_idx:end_idx]
                
                chunk = TextChunk(
                    id=f"{input_data.document_id}_fallback_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    chunk_index=chunk_id,
                    metadata={
                        "processing_method": "fallback_character_based",
                        "chunk_size": chunk_size,
                        "overlap": overlap
                    }
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Move to next chunk with overlap
                start_idx += (chunk_size - overlap)
                
                if end_idx >= len(text):
                    break
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Fallback chunking failed: {e}")
            return []