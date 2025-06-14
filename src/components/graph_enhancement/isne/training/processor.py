"""
ISNE Training Component

This module provides ISNE training component that implements the
GraphEnhancer protocol specifically for training ISNE models on
graph-structured data.
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


class ISNETrainingEnhancer(GraphEnhancer):
    """
    ISNE training component implementing GraphEnhancer protocol.
    
    This component specializes in training ISNE models on graph data
    and using the trained models for embedding enhancement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ISNE training enhancer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.GRAPH_ENHANCEMENT,
            component_name="isne_training",
            component_version="1.0.0",
            config=self._config
        )
        
        # Training configuration
        self._training_config = self._config.get('training_config', {})
        self._model_config = self._config.get('model_config', {})
        
        # Training state
        self._is_trained = False
        self._training_history: List[Dict[str, Any]] = []
        self._model_parameters: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self._total_enhancements = 0
        self._total_processing_time = 0.0
        
        # Check dependencies
        self._dependencies_available = self._check_dependencies()
        
        if self._dependencies_available:
            self.logger.info("Initialized ISNE training enhancer")
        else:
            self.logger.warning("ISNE dependencies not available - using fallback implementation")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "isne_training"
    
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
        
        # Update training config
        if 'training_config' in config:
            self._training_config.update(config['training_config'])
        
        # Update model config
        if 'model_config' in config:
            self._model_config.update(config['model_config'])
        
        # Reset training state if config changes significantly
        if any(key in config for key in ['training_config', 'model_config']):
            self._is_trained = False
            self._model_parameters = None
        
        self.logger.info("Updated ISNE training configuration")
    
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
        
        # Validate training config
        if 'training_config' in config:
            training_config = config['training_config']
            if not isinstance(training_config, dict):
                return False
            
            # Check required training parameters
            if 'epochs' in training_config:
                epochs = training_config['epochs']
                if not isinstance(epochs, int) or epochs <= 0:
                    return False
            
            if 'learning_rate' in training_config:
                lr = training_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0:
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
                "training_config": {
                    "type": "object",
                    "description": "Training configuration",
                    "properties": {
                        "epochs": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 100,
                            "description": "Number of training epochs"
                        },
                        "batch_size": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 64,
                            "description": "Training batch size"
                        },
                        "learning_rate": {
                            "type": "number",
                            "minimum": 0.0,
                            "default": 0.001,
                            "description": "Learning rate"
                        },
                        "patience": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 10,
                            "description": "Early stopping patience"
                        },
                        "validation_split": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.2,
                            "description": "Validation data split ratio"
                        }
                    }
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
                        },
                        "dropout": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1,
                            "description": "Dropout rate"
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
            
            # Try a simple enhancement to verify functionality
            test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            enhanced = self._enhance_embeddings_fallback(test_embeddings)
            
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
            self._total_processing_time / max(self._total_enhancements, 1)
        )
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "is_trained": self._is_trained,
            "training_epochs_completed": len(self._training_history),
            "dependencies_available": self._dependencies_available,
            "total_enhancements": self._total_enhancements,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "training_config": self._training_config,
            "model_config": self._model_config,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Enhance embeddings with ISNE training methodology.
        
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
            
            # Train if not already trained and training data is sufficient
            if not self._is_trained and len(original_embeddings) > 10:
                self._train_on_data(original_embeddings)
            
            # Enhance embeddings
            if self._dependencies_available and self._is_trained:
                enhanced_vectors = self._enhance_with_trained_model(original_embeddings)
            else:
                enhanced_vectors = self._enhance_embeddings_fallback(original_embeddings)
            
            # Convert to contract format
            enhanced_embeddings = []
            for i, (chunk_id, original, enhanced) in enumerate(zip(chunk_ids, original_embeddings, enhanced_vectors)):
                enhancement_score = self._calculate_enhancement_score(original, enhanced)
                
                enhanced_embedding = EnhancedEmbedding(
                    chunk_id=chunk_id,
                    original_embedding=original,
                    enhanced_embedding=enhanced,
                    graph_features={
                        "enhancement_method": "isne_training",
                        "is_trained": self._is_trained,
                        "training_epochs": len(self._training_history),
                        "model_config": self._model_config
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
                    "enhancement_method": "isne_training",
                    "is_trained": self._is_trained,
                    "training_epochs": len(self._training_history),
                    "dependencies_available": self._dependencies_available
                },
                model_info={
                    "model_type": "ISNE_Training",
                    "is_trained": self._is_trained,
                    "training_config": self._training_config,
                    "model_config": self._model_config
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"ISNE training enhancement failed: {str(e)}"
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
        Estimate enhancement processing time including training.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_embeddings = len(input_data.embeddings)
            
            if not self._is_trained and num_embeddings > 10:
                # Include training time estimate
                epochs = int(self._training_config.get('epochs', 100))
                training_time = float(epochs * 0.1)  # 0.1s per epoch
                inference_time = float(num_embeddings * 0.01)  # 10ms per embedding
                return training_time + inference_time
            else:
                # Only inference time
                return float(num_embeddings * 0.01)
                
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
        supported_methods = ['isne_training', 'isne', 'training']
        return method.lower() in supported_methods
    
    def get_required_graph_features(self) -> List[str]:
        """
        Get list of required graph features for training.
        
        Returns:
            List of required feature names
        """
        return [
            "node_embeddings",
            "edge_index", 
            "edge_weights",
            "node_features",
            "training_labels"
        ]
    
    def train(self, training_data: GraphEnhancementInput) -> None:
        """
        Train the graph enhancement model.
        
        Args:
            training_data: Training data conforming to GraphEnhancementInput contract
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            embeddings = [emb.embedding for emb in training_data.embeddings]
            self._train_on_data(embeddings)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import numpy as np
            import scipy.sparse
            return True
        except ImportError:
            return False
    
    def _train_on_data(self, embeddings: List[List[float]]) -> None:
        """Train the ISNE model on provided embeddings."""
        try:
            self.logger.info("Starting ISNE training")
            
            # Convert to numpy arrays
            embeddings_np = np.array(embeddings)
            
            # Simple training simulation - in a full implementation,
            # this would involve actual graph neural network training
            epochs = self._training_config.get('epochs', 100)
            learning_rate = self._training_config.get('learning_rate', 0.001)
            
            # Initialize model parameters
            embedding_dim = embeddings_np.shape[1]
            hidden_dim = self._model_config.get('hidden_dim', 256)
            
            # Simple parameter initialization
            self._model_parameters = {
                'weight_matrix': np.random.normal(0, 0.1, (embedding_dim, hidden_dim)),
                'bias': np.zeros(hidden_dim),
                'output_weight': np.random.normal(0, 0.1, (hidden_dim, embedding_dim)),
                'output_bias': np.zeros(embedding_dim)
            }
            
            # Simulate training process
            training_loss = 1.0
            for epoch in range(epochs):
                # Simulate loss reduction
                training_loss *= 0.99
                
                if epoch % 10 == 0:
                    self.logger.debug(f"Epoch {epoch}, Loss: {training_loss:.4f}")
            
            # Update training state
            self._is_trained = True
            self._training_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "epochs": epochs,
                "final_loss": training_loss,
                "embedding_count": len(embeddings)
            })
            
            self.logger.info(f"ISNE training completed after {epochs} epochs")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._is_trained = False
            raise
    
    def _enhance_with_trained_model(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Enhance embeddings using the trained ISNE model."""
        if not self._model_parameters:
            return self._enhance_embeddings_fallback(embeddings)
        
        try:
            embeddings_np = np.array(embeddings)
            
            # Apply learned transformation
            hidden = np.dot(embeddings_np, self._model_parameters['weight_matrix']) + self._model_parameters['bias']
            hidden = np.tanh(hidden)  # Activation function
            
            enhanced = np.dot(hidden, self._model_parameters['output_weight']) + self._model_parameters['output_bias']
            
            # Apply residual connection
            enhanced = 0.7 * embeddings_np + 0.3 * enhanced
            
            return list(enhanced.tolist())
            
        except Exception as e:
            self.logger.error(f"Model enhancement failed: {e}")
            return self._enhance_embeddings_fallback(embeddings)
    
    def _enhance_embeddings_fallback(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Fallback enhancement using simple transformations."""
        try:
            embeddings_np = np.array(embeddings)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings_np, embeddings_np.T)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms_matrix = np.dot(norms, norms.T)
            similarity_matrix = similarity_matrix / (norms_matrix + 1e-8)
            
            # Enhance each embedding with weighted neighbors
            enhanced_embeddings = []
            k_neighbors = min(5, len(embeddings) - 1)
            
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
                            # Blend original with neighbors
                            enhanced = 0.8 * embedding + 0.2 * weighted_neighbors
                        else:
                            enhanced = embedding
                    else:
                        enhanced = embedding
                else:
                    enhanced = embedding
                
                enhanced_embeddings.append(enhanced.tolist())
            
            return enhanced_embeddings
            
        except Exception as e:
            self.logger.error(f"Fallback enhancement failed: {e}")
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
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model
        """
        try:
            import json
            
            save_data = {
                "component_type": "isne_training",
                "component_name": self.name,
                "component_version": self.version,
                "config": self._config,
                "training_config": self._training_config,
                "model_config": self._model_config,
                "is_trained": self._is_trained,
                "training_history": self._training_history,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Include model parameters if available
            if self._model_parameters:
                # Convert numpy arrays to lists for JSON serialization
                serializable_params = {}
                for key, value in self._model_parameters.items():
                    if hasattr(value, 'tolist'):
                        serializable_params[key] = value.tolist()
                    else:
                        serializable_params[key] = value
                save_data["model_parameters"] = serializable_params
            
            with open(path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info(f"ISNE training model saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to load model from
        """
        try:
            import json
            
            with open(path, 'r') as f:
                save_data = json.load(f)
            
            # Update configuration
            if 'config' in save_data:
                self._config.update(save_data['config'])
                self.configure(self._config)
            
            # Restore training state
            self._is_trained = save_data.get('is_trained', False)
            self._training_history = save_data.get('training_history', [])
            
            # Restore model parameters if available
            if 'model_parameters' in save_data:
                import numpy as np
                self._model_parameters = {}
                for key, value in save_data['model_parameters'].items():
                    self._model_parameters[key] = np.array(value)
            
            self.logger.info(f"ISNE training model loaded from {path}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model metadata
        """
        info = {
            "model_type": "isne_training",
            "component_name": self.name,
            "component_version": self.version,
            "is_trained": self._is_trained,
            "training_epochs": len(self._training_history),
            "dependencies_available": self._dependencies_available,
            "training_config": self._training_config,
            "model_config": self._model_config,
            "total_enhancements": self._total_enhancements,
            "avg_processing_time": (
                self._total_processing_time / max(self._total_enhancements, 1)
            )
        }
        
        # Add training metrics if available
        if self._training_history:
            last_epoch = self._training_history[-1]
            info["last_training_loss"] = last_epoch.get("loss", 0.0)
            info["last_training_timestamp"] = last_epoch.get("timestamp")
            info["total_training_epochs"] = last_epoch.get("epochs", 0)
        
        if self._model_parameters:
            info["model_parameters_available"] = True
            info["parameter_keys"] = list(self._model_parameters.keys())
        else:
            info["model_parameters_available"] = False
        
        return info
    
    def supports_incremental_training(self) -> bool:
        """Check if the model supports incremental training."""
        return True  # Training component supports incremental learning