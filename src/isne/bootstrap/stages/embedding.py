"""
Embedding Stage for ISNE Bootstrap Pipeline

Handles generation of vector embeddings for chunks using
the configured embedding model and strategy.
"""

import logging
import traceback
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.components.embedding.factory import create_embedding_component
from src.types.components.contracts import EmbeddingInput
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding stage."""
    success: bool
    embeddings: List[Any]  # List of embedding objects
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class EmbeddingStage(BaseBootstrapStage):
    """Embedding stage for bootstrap pipeline."""
    
    def __init__(self):
        """Initialize embedding stage."""
        super().__init__("embedding")
        
    def execute(self, chunks: List[Any], config: Any) -> EmbeddingResult:
        """
        Execute embedding stage.
        
        Args:
            chunks: List of document chunks from chunking stage
            config: EmbeddingConfig object
            
        Returns:
            EmbeddingResult with embeddings and stats
        """
        logger.info(f"Starting embedding stage with {len(chunks)} chunks")
        
        try:
            # Create embedder
            embedder = create_embedding_component(config.embedder_type)
            
            # Prepare embedding input
            embedding_input = EmbeddingInput(
                chunks=chunks,
                model_name=config.model_name,
                embedding_options={
                    "normalize": config.normalize,
                    "batch_size": config.batch_size,
                    "device": config.device,
                    **config.options
                },
                batch_size=config.batch_size,
                metadata={
                    "bootstrap_stage": "embedding",
                    "pipeline": "isne_bootstrap",
                    "embedder_type": config.embedder_type,
                    "model_name": config.model_name
                }
            )
            
            logger.info(f"Generating embeddings using {config.embedder_type} embedder with {config.model_name}")
            logger.info(f"Batch size: {config.batch_size}, Device: {config.device}")
            
            # Generate embeddings
            embedding_start = time.time()
            output = embedder.embed(embedding_input)
            embedding_duration = time.time() - embedding_start
            
            # Validate embeddings quality
            valid_embeddings = [emb for emb in output.embeddings if len(emb.embedding) > 0]
            success_rate = len(valid_embeddings) / len(chunks) if chunks else 0.0
            
            # Calculate stats
            stats = {
                "input_chunks": len(chunks),
                "output_embeddings": len(output.embeddings),
                "valid_embeddings": len(valid_embeddings),
                "success_rate": success_rate,
                "embedding_dimension": valid_embeddings[0].embedding_dimension if valid_embeddings else 0,
                "processing_time_seconds": getattr(output.metadata, 'processing_time', embedding_duration),
                "embedding_duration_seconds": embedding_duration,
                "embeddings_per_second": len(valid_embeddings) / max(embedding_duration, 0.001),
                "model_info": {
                    "model_name": config.model_name,
                    "embedder_type": config.embedder_type,
                    "batch_size": config.batch_size,
                    "device": config.device,
                    "normalize": config.normalize
                },
                "chunk_details": []
            }
            
            # Add per-chunk details for analysis
            for i, (chunk, embedding) in enumerate(zip(chunks, output.embeddings)):
                chunk_detail = {
                    "chunk_id": chunk.id,
                    "document_id": getattr(chunk, 'document_id', f'unknown_doc_{i}'),
                    "chunk_length": len(chunk.content),
                    "embedding_dimension": len(embedding.embedding),
                    "embedding_valid": len(embedding.embedding) > 0
                }
                stats["chunk_details"].append(chunk_detail)
            
            # Check embedding quality
            if success_rate < 0.9:  # Less than 90% success rate
                warning_msg = (f"Low embedding success rate: {len(valid_embeddings)}/{len(chunks)} "
                             f"({success_rate*100:.1f}%)")
                logger.warning(warning_msg)
                stats["quality_warnings"] = [warning_msg]
            
            # Check for dimension consistency
            if valid_embeddings:
                dimensions = [emb.embedding_dimension for emb in valid_embeddings]
                unique_dimensions = set(dimensions)
                if len(unique_dimensions) > 1:
                    warning_msg = f"Inconsistent embedding dimensions: {unique_dimensions}"
                    logger.warning(warning_msg)
                    stats.setdefault("quality_warnings", []).append(warning_msg)
            
            logger.info(f"Embedding completed: {len(valid_embeddings)} valid embeddings generated")
            logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")
            logger.info(f"  Success rate: {success_rate*100:.1f}%")
            logger.info(f"  Throughput: {stats['embeddings_per_second']:.1f} embeddings/second")
            
            return EmbeddingResult(
                success=True,
                embeddings=output.embeddings,
                stats=stats
            )
            
        except Exception as e:
            error_msg = f"Embedding stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return EmbeddingResult(
                success=False,
                embeddings=[],
                stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def validate_inputs(self, chunks: List[Any], config: Any) -> List[str]:
        """
        Validate inputs for embedding stage.
        
        Args:
            chunks: List of document chunks
            config: EmbeddingConfig object
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not chunks:
            errors.append("No chunks provided for embedding")
            return errors
        
        # Check chunk structure
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'content'):
                errors.append(f"Chunk {i} missing 'content' attribute")
            elif not isinstance(chunk.content, str):
                errors.append(f"Chunk {i} content is not a string")
            elif not chunk.content.strip():
                errors.append(f"Chunk {i} has empty content")
            
            if not hasattr(chunk, 'id'):
                errors.append(f"Chunk {i} missing 'id' attribute")
        
        # Validate embedding config
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.batch_size > 1000:
            errors.append("batch_size is very large (>1000), this may cause memory issues")
        
        # Validate embedder type
        valid_embedders = ["cpu", "gpu", "core", "encoder"]
        if config.embedder_type not in valid_embedders:
            errors.append(f"Invalid embedder type: {config.embedder_type}. "
                         f"Valid options: {valid_embedders}")
        
        # Validate model name format
        if not config.model_name or not isinstance(config.model_name, str):
            errors.append("model_name must be a non-empty string")
        
        # Validate device
        valid_devices = ["cpu", "cuda", "auto"]
        if config.device not in valid_devices:
            # Allow specific CUDA devices like "cuda:0"
            if not (config.device.startswith("cuda:") and config.device[5:].isdigit()):
                errors.append(f"Invalid device: {config.device}. "
                             f"Valid options: {valid_devices} or 'cuda:N'")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["embeddings", "stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Number of chunks
            
        Returns:
            Estimated duration in seconds
        """
        # Embedding is typically the most time-consuming stage
        # Estimate based on chunks and typical processing speed
        base_time = 60  # Base overhead
        chunk_time = input_size * 0.1  # 0.1 seconds per chunk (conservative)
        return max(base_time, chunk_time)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements.
        
        Args:
            input_size: Number of chunks
            
        Returns:
            Dictionary with resource estimates
        """
        # Memory requirements depend on model size and batch size
        # Conservative estimates for sentence-transformers models
        base_memory = 500  # Base model loading
        batch_memory = input_size * 2  # 2MB per chunk in batch
        
        return {
            "memory_mb": max(base_memory, base_memory + batch_memory),
            "cpu_cores": 2,  # Embedding can benefit from multiple cores
            "disk_mb": 50,  # Minimal disk usage for embeddings
            "network_required": True,  # May need to download models
            "gpu_memory_mb": 2048 if input_size > 1000 else 1024  # If using GPU
        }
    
    def pre_execute_checks(self, chunks: List[Any], config: Any) -> List[str]:
        """
        Perform pre-execution checks specific to embedding stage.
        
        Args:
            chunks: List of document chunks
            config: EmbeddingConfig object
            
        Returns:
            List of check failure messages
        """
        checks = []
        
        # Check if GPU is available when requested
        if config.device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    checks.append("CUDA device requested but CUDA is not available")
                elif config.device != "cuda" and not torch.cuda.device_count() > int(config.device.split(":")[1]):
                    checks.append(f"CUDA device {config.device} not available")
            except ImportError:
                checks.append("PyTorch not available for CUDA check")
        
        # Check model accessibility (basic check)
        if "/" not in config.model_name and not config.model_name.startswith("sentence-transformers"):
            checks.append(f"Model name '{config.model_name}' may not be a valid model identifier")
        
        # Check memory requirements vs available memory
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
            required_memory = self.get_resource_requirements(len(chunks))["memory_mb"]
            
            if required_memory > available_memory_mb * 0.8:  # Use max 80% of available memory
                checks.append(f"Insufficient memory: need {required_memory}MB, "
                             f"available {available_memory_mb:.0f}MB")
        except ImportError:
            pass  # Skip memory check if psutil not available
        
        return checks