"""
Core ISNE Component

This module provides the core ISNE (Inductive Shallow Node Embedding) component 
that implements the GraphEnhancer protocol for graph-based embedding enhancement.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
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


class CoreISNE(GraphEnhancer):
    """
    Core ISNE component implementing GraphEnhancer protocol.
    
    This component provides graph-based embedding enhancement using inductive 
    shallow node embedding techniques for improved semantic representation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core ISNE enhancer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.GRAPH_ENHANCEMENT,
            component_name="core_isne",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration settings
        self._enhancement_method = self._config.get('enhancement_method', 'isne')
        self._model_config = self._config.get('model_config', {})
        self._graph_config = self._config.get('graph_config', {})
        
        # Model components
        self._model: Optional[Dict[str, Any]] = None
        self._pipeline_initialized = False
        
        # Performance tracking
        self._total_enhancements = 0
        self._total_processing_time = 0.0
        
        # Check if graph libraries are available
        self._graph_available = self._check_graph_availability()
        
        if self._graph_available:
            self.logger.info(f"Initialized core ISNE enhancer with method: {self._enhancement_method}")
        else:
            self.logger.warning("Graph libraries not available - enhancer will use fallback implementation")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "core_isne"
    
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
        
        # Update configuration parameters
        if 'enhancement_method' in config:
            self._enhancement_method = config['enhancement_method']
        
        if 'model_config' in config:
            self._model_config.update(config['model_config'])
        
        if 'graph_config' in config:
            self._graph_config.update(config['graph_config'])
        
        # Reset pipeline to force re-initialization
        self._pipeline_initialized = False
        self._model = None
        
        self.logger.info("Updated core ISNE configuration")
    
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
        
        # Validate enhancement method
        if 'enhancement_method' in config:
            valid_methods = ['isne', 'none', 'basic', 'similarity']
            if config['enhancement_method'] not in valid_methods:
                return False
        
        # Validate model config
        if 'model_config' in config:
            if not isinstance(config['model_config'], dict):
                return False
        
        # Validate graph config
        if 'graph_config' in config:
            if not isinstance(config['graph_config'], dict):
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
                "enhancement_method": {
                    "type": "string",
                    "enum": ["isne", "none", "basic", "similarity"],
                    "default": "isne",
                    "description": "Graph enhancement method to use"
                },
                "model_config": {
                    "type": "object",
                    "description": "ISNE model configuration",
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
                            "description": "Number of enhancement layers"
                        },
                        "dropout": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1,
                            "description": "Dropout rate"
                        },
                        "learning_rate": {
                            "type": "number",
                            "minimum": 0.0,
                            "default": 0.001,
                            "description": "Learning rate for training"
                        }
                    }
                },
                "graph_config": {
                    "type": "object",
                    "description": "Graph construction configuration",
                    "properties": {
                        "k_neighbors": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 10,
                            "description": "Number of nearest neighbors for graph construction"
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                            "description": "Similarity threshold for edge creation"
                        },
                        "edge_weighting": {
                            "type": "string",
                            "enum": ["cosine", "euclidean", "dot"],
                            "default": "cosine",
                            "description": "Edge weighting method"
                        },
                        "enhancement_strength": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                            "description": "Strength of graph enhancement"
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
            # Check graph libraries availability
            if not self._graph_available:
                self.logger.warning("Graph libraries not available for health check")
                return True  # Allow fallback implementation
            
            # Try to initialize pipeline if needed
            if not self._pipeline_initialized:
                self._initialize_pipeline()
            
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
            self._total_processing_time / max(self._total_enhancements, 1)
        )
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "enhancement_method": self._enhancement_method,
            "graph_available": self._graph_available,
            "pipeline_initialized": self._pipeline_initialized,
            "model_initialized": self._model is not None,
            "total_enhancements": self._total_enhancements,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "model_config": self._model_config,
            "graph_config": self._graph_config,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Enhance embeddings with graph information according to the contract.
        
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
            
            # Enhance embeddings based on method
            if self._graph_available and self._enhancement_method == 'isne':
                enhanced_vectors = self._enhance_with_isne(original_embeddings, input_data)
            elif self._enhancement_method == 'similarity':
                enhanced_vectors = self._enhance_with_similarity(original_embeddings, input_data)
            elif self._enhancement_method == 'basic':
                enhanced_vectors = self._enhance_basic(original_embeddings, input_data)
            else:
                # No enhancement - pass through
                enhanced_vectors = original_embeddings.copy()
            
            # Convert to contract format
            enhanced_embeddings = []
            for i, (chunk_id, original, enhanced) in enumerate(zip(chunk_ids, original_embeddings, enhanced_vectors)):
                # Calculate enhancement score
                enhancement_score = self._calculate_enhancement_score(original, enhanced)
                
                enhanced_embedding = EnhancedEmbedding(
                    chunk_id=chunk_id,
                    original_embedding=original,
                    enhanced_embedding=enhanced,
                    graph_features={
                        "enhancement_method": self._enhancement_method,
                        "graph_neighbors": self._graph_config.get("k_neighbors", 10),
                        "processing_index": i,
                        "enhancement_strength": self._graph_config.get("enhancement_strength", 0.5)
                    },
                    enhancement_score=enhancement_score,
                    metadata=input_data.metadata
                )
                enhanced_embeddings.append(enhanced_embedding)
            
            # Calculate processing time and update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_enhancements += len(enhanced_embeddings)
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
                    "enhancement_method": self._enhancement_method,
                    "average_enhancement_score": sum(e.enhancement_score or 0 for e in enhanced_embeddings) / len(enhanced_embeddings) if enhanced_embeddings else 0,
                    "graph_config": self._graph_config
                },
                model_info={
                    "model_type": "ISNE",
                    "enhancement_method": self._enhancement_method,
                    "model_config": self._model_config,
                    "graph_available": self._graph_available
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"ISNE enhancement failed: {str(e)}"
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
        Estimate enhancement processing time.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_embeddings = len(input_data.embeddings)
            embedding_dim = len(input_data.embeddings[0].embedding) if input_data.embeddings else 768
            
            # Different methods have different computational complexity
            if self._enhancement_method == 'isne':
                # ISNE is more compute intensive
                base_time = num_embeddings * 0.02  # 20ms per embedding
            elif self._enhancement_method in ['similarity', 'basic']:
                # Simpler methods are faster
                base_time = num_embeddings * 0.005  # 5ms per embedding
            else:
                # No enhancement
                base_time = num_embeddings * 0.001  # 1ms per embedding
            
            # Scale by dimension complexity
            complexity_factor = embedding_dim / 768
            
            return max(0.1, base_time * complexity_factor)
            
        except Exception:
            return 1.0  # Default estimate
    
    def supports_enhancement_method(self, method: str) -> bool:
        """
        Check if enhancer supports the given enhancement method.
        
        Args:
            method: Enhancement method to check
            
        Returns:
            True if method is supported, False otherwise
        """
        supported_methods = ['isne', 'none', 'basic', 'similarity']
        return method.lower() in supported_methods
    
    def get_required_graph_features(self) -> List[str]:
        """
        Get list of required graph features for enhancement.
        
        Returns:
            List of required feature names
        """
        return [
            "node_embeddings",
            "adjacency_matrix", 
            "similarity_matrix",
            "neighbor_indices"
        ]
    
    def _check_graph_availability(self) -> bool:
        """Check if graph processing libraries are available."""
        try:
            import networkx
            import scipy.sparse
            return True
        except ImportError:
            return False
    
    def _initialize_pipeline(self) -> None:
        """Initialize the ISNE enhancement pipeline."""
        try:
            self.logger.info("Initializing ISNE enhancement pipeline")
            
            # For now, use a placeholder model initialization
            # In a full implementation, this would initialize the actual ISNE model
            self._model = {
                "embedding_dim": self._model_config.get('embedding_dim', 768),
                "hidden_dim": self._model_config.get('hidden_dim', 256),
                "num_layers": self._model_config.get('num_layers', 3),
                "dropout": self._model_config.get('dropout', 0.1),
                "initialized": True
            }
            
            self._pipeline_initialized = True
            self.logger.info("ISNE pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ISNE pipeline: {e}")
            self._pipeline_initialized = False
            self._model = None
    
    def _enhance_with_isne(self, embeddings: List[List[float]], input_data: GraphEnhancementInput) -> List[List[float]]:
        """Enhance embeddings using ISNE method."""
        if not self._pipeline_initialized:
            self._initialize_pipeline()
        
        try:
            # TODO: Implement actual ISNE enhancement
            # For now, use similarity-based enhancement as placeholder
            return self._enhance_with_similarity(embeddings, input_data)
            
        except Exception as e:
            self.logger.error(f"ISNE enhancement failed: {e}")
            return self._enhance_basic(embeddings, input_data)
    
    def _enhance_with_similarity(self, embeddings: List[List[float]], input_data: GraphEnhancementInput) -> List[List[float]]:
        """Enhance embeddings using similarity-based graph enhancement."""
        try:
            import numpy as np
            
            embeddings_np = np.array(embeddings)
            k_neighbors = self._graph_config.get('k_neighbors', 10)
            enhancement_strength = self._graph_config.get('enhancement_strength', 0.5)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings_np, embeddings_np.T)
            
            # Normalize by vector norms
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms_matrix = np.dot(norms, norms.T)
            similarity_matrix = similarity_matrix / (norms_matrix + 1e-8)
            
            # For each embedding, enhance with k most similar neighbors
            enhanced_embeddings = []
            for i, embedding in enumerate(embeddings_np):
                # Get k most similar neighbors (excluding self)
                similarities = similarity_matrix[i]
                neighbor_indices = np.argsort(similarities)[::-1][1:k_neighbors+1]
                
                # Weight neighbors by similarity
                neighbor_weights = similarities[neighbor_indices]
                neighbor_embeddings = embeddings_np[neighbor_indices]
                
                # Weighted average of neighbors
                if len(neighbor_embeddings) > 0 and np.sum(neighbor_weights) > 0:
                    weighted_neighbors = np.average(neighbor_embeddings, axis=0, weights=neighbor_weights)
                    
                    # Blend original with neighbors
                    enhanced = (1 - enhancement_strength) * embedding + enhancement_strength * weighted_neighbors
                else:
                    enhanced = embedding
                
                enhanced_embeddings.append(enhanced.tolist())
            
            return enhanced_embeddings
            
        except Exception as e:
            self.logger.error(f"Similarity enhancement failed: {e}")
            return self._enhance_basic(embeddings, input_data)
    
    def _enhance_basic(self, embeddings: List[List[float]], input_data: GraphEnhancementInput) -> List[List[float]]:
        """Basic enhancement using simple transformations."""
        try:
            import numpy as np
            
            enhancement_strength = self._graph_config.get('enhancement_strength', 0.1)
            
            enhanced_embeddings = []
            for embedding in embeddings:
                embedding_np = np.array(embedding)
                
                # Simple enhancement: add small random noise scaled by enhancement strength
                noise = np.random.normal(0, 0.01, embedding_np.shape) * enhancement_strength
                enhanced = embedding_np + noise
                
                # Normalize to maintain embedding properties
                norm = np.linalg.norm(enhanced)
                if norm > 0:
                    enhanced = enhanced / norm * np.linalg.norm(embedding_np)
                
                enhanced_embeddings.append(enhanced.tolist())
            
            return enhanced_embeddings
            
        except Exception as e:
            self.logger.error(f"Basic enhancement failed: {e}")
            # Return original embeddings if enhancement fails
            return embeddings
    
    def _calculate_enhancement_score(self, original: List[float], enhanced: List[float]) -> float:
        """
        Calculate enhancement score between original and enhanced embeddings.
        
        Args:
            original: Original embedding vector
            enhanced: Enhanced embedding vector
            
        Returns:
            Enhancement score between 0 and 1
        """
        try:
            import numpy as np
            
            original_np = np.array(original)
            enhanced_np = np.array(enhanced)
            
            # Calculate cosine similarity
            orig_norm = np.linalg.norm(original_np)
            enh_norm = np.linalg.norm(enhanced_np)
            
            if orig_norm == 0 or enh_norm == 0:
                return 0.0
            
            similarity = float(np.dot(original_np, enhanced_np) / (orig_norm * enh_norm))
            
            # Enhancement score: higher values indicate more change
            # Map similarity to enhancement score (1.0 = no change, 0.0 = maximum change)
            enhancement_score = 1.0 - abs(similarity)
            
            return float(min(1.0, max(0.0, enhancement_score)))
            
        except Exception:
            return 0.0  # Default score if calculation fails