"""
Inductive ISNE Component

This module provides the inductive ISNE component that implements the
GraphEnhancer protocol. It provides inductive learning capabilities for
enhancing embeddings of new nodes that weren't seen during training.
"""

import logging
import torch
import torch.nn as nn
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


class InductiveISNE(GraphEnhancer):
    """
    Inductive ISNE component implementing GraphEnhancer protocol.
    
    This component provides inductive learning capabilities for enhancing
    embeddings of new nodes using pre-trained graph neural network models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize inductive ISNE component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.GRAPH_ENHANCEMENT,
            component_name="inductive",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._hidden_dim = self._config.get('hidden_dim', 256)
        self._num_layers = self._config.get('num_layers', 2)
        self._dropout = self._config.get('dropout', 0.1)
        self._device = self._config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self._model: Optional[Any] = None
        self._is_trained = False
        self._training_history: List[Dict[str, Any]] = []
        
        # Training configuration
        self._learning_rate = self._config.get('learning_rate', 0.001)
        self._max_epochs = self._config.get('max_epochs', 100)
        self._early_stopping_patience = self._config.get('early_stopping_patience', 10)
        
        self.logger.info(f"Initialized inductive ISNE with hidden_dim={self._hidden_dim}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "inductive"
    
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
        if 'hidden_dim' in config:
            self._hidden_dim = config['hidden_dim']
        
        if 'num_layers' in config:
            self._num_layers = config['num_layers']
        
        if 'device' in config:
            self._device = config['device']
        
        # Reinitialize model if configuration changed
        if self._model is not None:
            self._initialize_model()
        
        self.logger.info("Updated inductive ISNE configuration")
    
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
        
        # Validate hidden dimension
        if 'hidden_dim' in config:
            if not isinstance(config['hidden_dim'], int) or config['hidden_dim'] < 1:
                return False
        
        # Validate number of layers
        if 'num_layers' in config:
            if not isinstance(config['num_layers'], int) or config['num_layers'] < 1:
                return False
        
        # Validate dropout
        if 'dropout' in config:
            dropout = config['dropout']
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
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
                "hidden_dim": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 256,
                    "description": "Hidden dimension size for neural network layers"
                },
                "num_layers": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                    "description": "Number of neural network layers"
                },
                "dropout": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.99,
                    "default": 0.1,
                    "description": "Dropout rate for regularization"
                },
                "device": {
                    "type": "string",
                    "enum": ["cuda", "cpu", "auto"],
                    "default": "auto",
                    "description": "Device to use for computation"
                },
                "learning_rate": {
                    "type": "number",
                    "minimum": 0.0001,
                    "maximum": 1.0,
                    "default": 0.001,
                    "description": "Learning rate for training"
                },
                "max_epochs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Maximum training epochs"
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
            # Check if PyTorch is available
            import torch
            
            # Check device availability
            if self._device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available")
                return False
            
            # Check if model can be initialized
            if self._model is None:
                self._initialize_model()
            
            return self._model is not None
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "device": self._device,
            "is_trained": self._is_trained,
            "training_epochs": len(self._training_history),
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add training metrics if available
        if self._training_history:
            last_epoch = self._training_history[-1]
            metrics.update({
                "last_training_loss": last_epoch.get("loss", 0.0),
                "best_validation_score": max([epoch.get("val_score", 0.0) for epoch in self._training_history])
            })
        
        return metrics
    
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Enhance embeddings using inductive methods according to the contract.
        
        Args:
            input_data: Input data conforming to GraphEnhancementInput contract
            
        Returns:
            Output data conforming to GraphEnhancementOutput contract
            
        Raises:
            EnhancementError: If enhancement fails
        """
        errors = []
        enhanced_embeddings = []
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize model if needed
            if self._model is None:
                self._initialize_model()
            
            if not self._is_trained:
                self.logger.warning("Model not trained, using random initialization")
            
            # Process embeddings
            for chunk_embedding in input_data.embeddings:
                try:
                    enhanced_emb = self._enhance_single_embedding(
                        chunk_embedding,
                        input_data.enhancement_options
                    )
                    enhanced_embeddings.append(enhanced_emb)
                    
                except Exception as e:
                    error_msg = f"Failed to enhance embedding {chunk_embedding.chunk_id}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate enhancement quality score
            quality_score = self._calculate_quality_score(enhanced_embeddings)
            
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
            
            return GraphEnhancementOutput(
                enhanced_embeddings=enhanced_embeddings,
                metadata=metadata,
                graph_stats={
                    "processing_time": processing_time,
                    "embeddings_enhanced": len(enhanced_embeddings),
                    "embeddings_failed": len(errors),
                    "inductive_method": "neural_network",
                    "model_trained": self._is_trained
                },
                model_info={
                    "model_type": "inductive_isne",
                    "hidden_dim": self._hidden_dim,
                    "num_layers": self._num_layers,
                    "device": self._device,
                    "parameters": self._count_parameters()
                },
                errors=errors,
                enhancement_quality_score=quality_score
            )
            
        except Exception as e:
            error_msg = f"Inductive ISNE enhancement failed: {str(e)}"
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
    
    def train(self, training_data: GraphEnhancementInput) -> None:
        """
        Train the inductive enhancement model.
        
        Args:
            training_data: Training data conforming to GraphEnhancementInput contract
            
        Raises:
            TrainingError: If training fails
        """
        try:
            self.logger.info("Starting inductive ISNE training")
            
            # Initialize model if needed
            if self._model is None:
                self._initialize_model()
            
            # Prepare training data
            embeddings = []
            targets = []
            
            for chunk_emb in training_data.embeddings:
                embeddings.append(torch.tensor(chunk_emb.embedding, dtype=torch.float32))
                # Use the embedding itself as target for autoencoder-style training
                targets.append(torch.tensor(chunk_emb.embedding, dtype=torch.float32))
            
            if not embeddings:
                raise ValueError("No training data provided")
            
            # Convert to tensors
            X = torch.stack(embeddings).to(self._device)
            y = torch.stack(targets).to(self._device)
            
            # Setup training
            if self._model is None:
                raise ValueError("Model not initialized")
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            self._training_history = []
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self._max_epochs):
                self._model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self._model(X)
                loss = criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Record training metrics
                epoch_info = {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "timestamp": datetime.utcnow()
                }
                self._training_history.append(epoch_info)
                
                # Early stopping check
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self._early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self._is_trained = True
            self.logger.info(f"Training completed after {len(self._training_history)} epochs")
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model
            
        Raises:
            IOError: If model cannot be saved
        """
        try:
            if self._model is None:
                raise ValueError("No model to save")
            
            save_dict = {
                'model_state_dict': self._model.state_dict(),
                'config': self._config,
                'is_trained': self._is_trained,
                'training_history': self._training_history
            }
            
            torch.save(save_dict, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to load model from
            
        Raises:
            IOError: If model cannot be loaded
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)
            
            # Update configuration from saved model
            self._config.update(checkpoint.get('config', {}))
            
            # Initialize model with loaded config
            self._initialize_model()
            
            # Load model state
            if self._model is None:
                raise ValueError("Model not initialized")
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._is_trained = checkpoint.get('is_trained', False)
            self._training_history = checkpoint.get('training_history', [])
            
            self.logger.info(f"Model loaded from {path}")
            
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
            "model_type": "inductive_isne",
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "device": self._device,
            "is_trained": self._is_trained,
            "parameter_count": self._count_parameters(),
            "training_epochs": len(self._training_history)
        }
        
        if self._training_history:
            info["last_training_loss"] = self._training_history[-1].get("loss", 0.0)
        
        return info
    
    def supports_incremental_training(self) -> bool:
        """Check if the model supports incremental training."""
        return True
    
    def _initialize_model(self) -> None:
        """Initialize the neural network model."""
        try:
            # Simple feedforward network for inductive enhancement
            layers: List[nn.Module] = []
            
            # Input layer (we'll set this dynamically based on first input)
            input_dim = self._config.get('input_dim', 384)  # Default embedding size
            
            layers.append(nn.Linear(input_dim, self._hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self._dropout))
            
            # Hidden layers
            for _ in range(self._num_layers - 1):
                layers.append(nn.Linear(self._hidden_dim, self._hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self._dropout))
            
            # Output layer (same size as input for autoencoder-style enhancement)
            layers.append(nn.Linear(self._hidden_dim, input_dim))
            
            self._model = nn.Sequential(*layers).to(self._device)
            
            self.logger.info(f"Initialized inductive ISNE model with {self._count_parameters()} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            self._model = None
    
    def _enhance_single_embedding(self, chunk_embedding: Any, options: Dict[str, Any]) -> EnhancedEmbedding:
        """Enhance a single embedding using the inductive model."""
        try:
            # Convert to tensor
            embedding_tensor = torch.tensor(
                chunk_embedding.embedding, 
                dtype=torch.float32
            ).unsqueeze(0).to(self._device)
            
            # Forward pass through model
            if self._model is None:
                raise ValueError("Model not initialized")
            self._model.eval()
            with torch.no_grad():
                enhanced_tensor = self._model(embedding_tensor)
            
            # Convert back to list
            enhanced_embedding = enhanced_tensor.squeeze(0).cpu().tolist()
            
            # Calculate enhancement score (similarity to original)
            enhancement_score = self._calculate_enhancement_score(
                chunk_embedding.embedding,
                enhanced_embedding
            )
            
            return EnhancedEmbedding(
                chunk_id=chunk_embedding.chunk_id,
                original_embedding=chunk_embedding.embedding,
                enhanced_embedding=enhanced_embedding,
                graph_features={
                    "enhancement_method": "inductive_neural",
                    "model_confidence": float(enhancement_score),
                    "layers_used": self._num_layers
                },
                enhancement_score=enhancement_score,
                metadata=chunk_embedding.metadata or {}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to enhance embedding: {e}")
            # Return original embedding if enhancement fails
            return EnhancedEmbedding(
                chunk_id=chunk_embedding.chunk_id,
                original_embedding=chunk_embedding.embedding,
                enhanced_embedding=chunk_embedding.embedding,
                graph_features={"enhancement_method": "passthrough", "error": str(e)},
                enhancement_score=0.0,
                metadata=chunk_embedding.metadata or {}
            )
    
    def _calculate_enhancement_score(self, original: List[float], enhanced: List[float]) -> float:
        """Calculate enhancement quality score."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            orig_array = np.array(original).reshape(1, -1)
            enh_array = np.array(enhanced).reshape(1, -1)
            
            # Higher cosine similarity indicates better preservation of semantic meaning
            similarity = cosine_similarity(orig_array, enh_array)[0][0]
            return float(similarity)
            
        except Exception:
            return 0.5  # Default score
    
    def _calculate_quality_score(self, enhanced_embeddings: List[EnhancedEmbedding]) -> float:
        """Calculate overall enhancement quality score."""
        if not enhanced_embeddings:
            return 0.0
        
        scores = [emb.enhancement_score or 0.0 for emb in enhanced_embeddings]
        return sum(scores) / len(scores)
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        if self._model is None:
            return 0
        
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)