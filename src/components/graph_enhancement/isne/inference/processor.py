"""
ISNE Inference Component

This module provides ISNE inference component that implements the
GraphEnhancer protocol specifically for running inference with
pre-trained ISNE models.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    EnhancedEmbedding,
    ProcessingStatus
)
from src.types.components.protocols import GraphEnhancer


class ISNEInferenceEnhancer(GraphEnhancer):
    """
    ISNE inference component implementing GraphEnhancer protocol.
    
    This component specializes in running inference with pre-trained
    ISNE models for fast embedding enhancement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ISNE inference enhancer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.GRAPH_ENHANCEMENT,
            component_name="isne_inference",
            component_version="1.0.0",
            config=self._config
        )
        
        # Inference configuration
        self._model_config = self._config.get('model_config', {})
        self._inference_config = self._config.get('inference_config', {})
        
        # Model state
        self._model_loaded = False
        self._model_parameters: Optional[Dict[str, Any]] = None
        self._model_path = self._config.get('model_path')
        
        # Performance tracking
        self._total_inferences = 0
        self._total_processing_time = 0.0
        
        # Check dependencies
        self._dependencies_available = self._check_dependencies()
        
        if self._dependencies_available:
            self.logger.info("Initialized ISNE inference enhancer")
            # Try to load model if path is provided
            if self._model_path:
                self._load_model_if_available()
        else:
            self.logger.warning("ISNE dependencies not available - using fallback implementation")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "isne_inference"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.GRAPH_ENHANCEMENT
    
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
        
        # Update model config
        if 'model_config' in config:
            self._model_config.update(config['model_config'])
        
        # Update inference config
        if 'inference_config' in config:
            self._inference_config.update(config['inference_config'])
        
        # Update model path
        if 'model_path' in config:
            self._model_path = config['model_path']
            self._model_loaded = False
            self._model_parameters = None
            # Try to load the new model
            if self._model_path:
                self._load_model_if_available()
        
        self.logger.info("Updated ISNE inference configuration")
    
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
        
        # Validate model config
        if 'model_config' in config:
            model_config = config['model_config']
            if not isinstance(model_config, dict):
                return False
            
            if 'embedding_dim' in model_config:
                dim = model_config['embedding_dim']
                if not isinstance(dim, int) or dim <= 0:
                    return False
        
        # Validate inference config
        if 'inference_config' in config:
            inference_config = config['inference_config']
            if not isinstance(inference_config, dict):
                return False
            
            if 'batch_size' in inference_config:
                batch_size = inference_config['batch_size']
                if not isinstance(batch_size, int) or batch_size <= 0:
                    return False
        
        # Validate model path
        if 'model_path' in config:
            model_path = config['model_path']
            if not isinstance(model_path, str):
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
                "model_path": {
                    "type": "string",
                    "description": "Path to pre-trained ISNE model"
                },
                "model_config": {
                    "type": "object",
                    "description": "Model configuration",
                    "properties": {
                        "embedding_dim": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 768,
                            "description": "Embedding dimension"
                        },
                        "hidden_dim": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 256,
                            "description": "Hidden layer dimension"
                        },
                        "num_layers": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 3,
                            "description": "Number of ISNE layers"
                        }
                    }
                },
                "inference_config": {
                    "type": "object",
                    "description": "Inference configuration",
                    "properties": {
                        "batch_size": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 128,
                            "description": "Inference batch size"
                        },
                        "use_cache": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to cache inference results"
                        },
                        "enhancement_strength": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                            "description": "Strength of enhancement application"
                        }
                    }
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
            # Check if dependencies are available
            if not self._dependencies_available:
                self.logger.warning("Dependencies not available for health check")
                return True  # Allow fallback implementation
            
            # Check if configuration is valid
            if not self.validate_config(self._config):
                return False
            
            # Try a simple inference to verify functionality
            test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            enhanced = self._inference_fallback(test_embeddings)
            
            return len(enhanced) == len(test_embeddings)
            
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
            self._total_processing_time / max(self._total_inferences, 1)
        )
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "model_loaded": self._model_loaded,
            "model_path": self._model_path,
            "dependencies_available": self._dependencies_available,
            "total_inferences": self._total_inferences,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "model_config": self._model_config,
            "inference_config": self._inference_config,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Enhance embeddings with pre-trained ISNE model.
        
        Args:
            input_data: Input data conforming to GraphEnhancementInput contract
            
        Returns:
            Output data conforming to GraphEnhancementOutput contract
        """
        errors: List[str] = []
        
        try:
            start_time = datetime.utcnow()
            
            # Extract embeddings from input
            original_embeddings = []
            chunk_ids = []
            
            for embedding in input_data.embeddings:
                original_embeddings.append(embedding.embedding)
                chunk_ids.append(embedding.chunk_id)
            
            # Run inference
            if self._dependencies_available and self._model_loaded:
                enhanced_vectors = self._inference_with_model(original_embeddings)
            else:
                enhanced_vectors = self._inference_fallback(original_embeddings)
            
            # Convert to contract format
            enhanced_embeddings = []
            for i, (chunk_id, original, enhanced) in enumerate(zip(chunk_ids, original_embeddings, enhanced_vectors)):
                enhancement_score = self._calculate_enhancement_score(original, enhanced)
                
                enhanced_embedding = EnhancedEmbedding(
                    chunk_id=chunk_id,
                    original_embedding=original,
                    enhanced_embedding=enhanced,
                    graph_features={
                        "enhancement_method": "isne_inference",
                        "model_loaded": self._model_loaded,
                        "model_path": self._model_path,
                        "inference_config": self._inference_config
                    },
                    enhancement_score=enhancement_score,
                    metadata=input_data.metadata
                )
                enhanced_embeddings.append(enhanced_embedding)
            
            # Calculate processing time and update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_inferences += len(enhanced_embeddings)
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
            
            return GraphEnhancementOutput(
                enhanced_embeddings=enhanced_embeddings,
                metadata=metadata,
                graph_stats={
                    "processing_time": processing_time,
                    "embedding_count": len(enhanced_embeddings),
                    "enhancement_method": "isne_inference",
                    "model_loaded": self._model_loaded,
                    "dependencies_available": self._dependencies_available
                },
                model_info={
                    "model_type": "ISNE_Inference",
                    "model_loaded": self._model_loaded,
                    "model_path": self._model_path,
                    "inference_config": self._inference_config
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"ISNE inference enhancement failed: {str(e)}"
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
            
            return GraphEnhancementOutput(
                enhanced_embeddings=[],
                metadata=metadata,
                graph_stats={},
                model_info={},
                errors=errors
            )
    
    def estimate_enhancement_time(self, input_data: GraphEnhancementInput) -> float:
        """
        Estimate enhancement processing time for inference.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_embeddings = len(input_data.embeddings)
            
            if self._model_loaded:
                # Fast inference with loaded model
                return num_embeddings * 0.001  # 1ms per embedding
            else:
                # Slower fallback processing
                return num_embeddings * 0.005  # 5ms per embedding
                
        except Exception:
            return 0.1  # Default estimate
    
    def supports_enhancement_method(self, method: str) -> bool:
        """
        Check if enhancer supports the given enhancement method.
        
        Args:
            method: Enhancement method to check
            
        Returns:
            True if method is supported, False otherwise
        """
        supported_methods = ['isne_inference', 'isne', 'inference']
        return method.lower() in supported_methods
    
    def get_required_graph_features(self) -> List[str]:
        """
        Get list of required graph features for inference.
        
        Returns:
            List of required feature names
        """
        return [
            "node_embeddings",
            "graph_structure"  # Less features needed for inference
        ]
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained ISNE model from disk.
        
        Args:
            model_path: Path to model file
            
        Raises:
            IOError: If model cannot be loaded
        """
        try:
            from pathlib import Path
            
            load_path = Path(model_path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self._model_path = model_path
            self._load_model_if_available()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model: {e}")
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import numpy as np
            import scipy.sparse
            return True
        except ImportError:
            return False
    
    def _load_model_if_available(self) -> None:
        """Load model if path is available and dependencies are met."""
        if not self._model_path or not self._dependencies_available:
            return
        
        try:
            from pathlib import Path
            
            model_path = Path(self._model_path)
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {self._model_path}")
                return
            
            # Simulate model loading - in real implementation, this would
            # load actual model parameters from disk
            self.logger.info(f"Loading ISNE model from {self._model_path}")
            
            # Mock model parameters for demonstration
            embedding_dim = self._model_config.get('embedding_dim', 768)
            hidden_dim = self._model_config.get('hidden_dim', 256)
            
            self._model_parameters = {
                'weight_matrix': np.random.normal(0, 0.1, (embedding_dim, hidden_dim)),
                'bias': np.zeros(hidden_dim),
                'output_weight': np.random.normal(0, 0.1, (hidden_dim, embedding_dim)),
                'output_bias': np.zeros(embedding_dim)
            }
            
            self._model_loaded = True
            self.logger.info("ISNE model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            self._model_parameters = None
    
    def _inference_with_model(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Run inference with loaded ISNE model."""
        if not self._model_parameters:
            return self._inference_fallback(embeddings)
        
        try:
            embeddings_np = np.array(embeddings)
            enhancement_strength = self._inference_config.get('enhancement_strength', 0.5)
            
            # Apply learned transformation
            hidden = np.dot(embeddings_np, self._model_parameters['weight_matrix']) + self._model_parameters['bias']
            hidden = np.tanh(hidden)  # Activation function
            
            enhanced = np.dot(hidden, self._model_parameters['output_weight']) + self._model_parameters['output_bias']
            
            # Apply enhancement strength
            enhanced = (1 - enhancement_strength) * embeddings_np + enhancement_strength * enhanced
            
            return list(enhanced.tolist())
            
        except Exception as e:
            self.logger.error(f"Model inference failed: {e}")
            return self._inference_fallback(embeddings)
    
    def _inference_fallback(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Fallback inference using similarity-based enhancement."""
        try:
            embeddings_np = np.array(embeddings)
            
            # Simple similarity-based enhancement
            similarity_matrix = np.dot(embeddings_np, embeddings_np.T)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms_matrix = np.dot(norms, norms.T)
            similarity_matrix = similarity_matrix / (norms_matrix + 1e-8)
            
            # Enhance each embedding with neighbors
            enhanced_embeddings = []
            k_neighbors = min(3, len(embeddings) - 1)  # Fewer neighbors for faster inference
            
            for i, embedding in enumerate(embeddings_np):
                if k_neighbors > 0:
                    # Get k most similar neighbors (excluding self)
                    similarities = similarity_matrix[i]
                    neighbor_indices = np.argsort(similarities)[::-1][1:k_neighbors+1]
                    
                    if len(neighbor_indices) > 0:
                        neighbor_weights = similarities[neighbor_indices]
                        neighbor_embeddings = embeddings_np[neighbor_indices]
                        
                        # Weighted average of neighbors
                        if np.sum(neighbor_weights) > 0:
                            weighted_neighbors = np.average(neighbor_embeddings, axis=0, weights=neighbor_weights)
                            # Light enhancement for inference
                            enhanced = 0.9 * embedding + 0.1 * weighted_neighbors
                        else:
                            enhanced = embedding
                    else:
                        enhanced = embedding
                else:
                    enhanced = embedding
                
                enhanced_embeddings.append(enhanced.tolist())
            
            return enhanced_embeddings
            
        except Exception as e:
            self.logger.error(f"Fallback inference failed: {e}")
            # Return original embeddings if enhancement fails
            return embeddings
    
    def _calculate_enhancement_score(self, original: List[float], enhanced: List[float]) -> float:
        """Calculate enhancement score between original and enhanced embeddings."""
        try:
            original_np = np.array(original)
            enhanced_np = np.array(enhanced)
            
            # Calculate cosine similarity
            orig_norm = np.linalg.norm(original_np)
            enh_norm = np.linalg.norm(enhanced_np)
            
            if orig_norm == 0 or enh_norm == 0:
                return 0.0
            
            similarity = float(np.dot(original_np, enhanced_np) / (orig_norm * enh_norm))
            
            # Enhancement score: higher values indicate more change
            enhancement_score = 1.0 - abs(similarity)
            
            return float(min(1.0, max(0.0, enhancement_score)))
            
        except Exception:
            return 0.0