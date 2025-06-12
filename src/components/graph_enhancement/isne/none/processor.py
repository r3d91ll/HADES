"""
None/Passthrough ISNE Component

This module provides a passthrough ISNE component that implements the
GraphEnhancer protocol but performs no actual enhancement. It simply
passes through the original embeddings unchanged.
"""

import logging
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


class NoneISNE(GraphEnhancer):
    """
    None/Passthrough ISNE component implementing GraphEnhancer protocol.
    
    This component provides a no-op implementation that passes through
    original embeddings without any enhancement. Useful for A/B testing
    and performance baselines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize none/passthrough ISNE component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.GRAPH_ENHANCEMENT,
            component_name="none",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._preserve_original = self._config.get('preserve_original_embeddings', True)
        self._add_metadata = self._config.get('add_metadata', True)
        self._validate_inputs = self._config.get('validate_inputs', True)
        self._validate_outputs = self._config.get('validate_outputs', True)
        self._copy_tensors = self._config.get('copy_tensors', False)
        
        # Statistics
        self._total_processed = 0
        self._total_processing_time = 0.0
        
        self.logger.info("Initialized none/passthrough ISNE component")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "none"
    
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
        if 'preserve_original_embeddings' in config:
            self._preserve_original = config['preserve_original_embeddings']
        
        if 'add_metadata' in config:
            self._add_metadata = config['add_metadata']
        
        if 'validate_inputs' in config:
            self._validate_inputs = config['validate_inputs']
        
        if 'validate_outputs' in config:
            self._validate_outputs = config['validate_outputs']
        
        if 'copy_tensors' in config:
            self._copy_tensors = config['copy_tensors']
        
        self.logger.info("Updated none/passthrough ISNE configuration")
    
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
        
        # Validate boolean parameters
        bool_params = [
            'preserve_original_embeddings',
            'add_metadata',
            'validate_inputs',
            'validate_outputs',
            'copy_tensors'
        ]
        
        for param in bool_params:
            if param in config and not isinstance(config[param], bool):
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
                "preserve_original_embeddings": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve original embeddings in output"
                },
                "add_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Add passthrough metadata to embeddings"
                },
                "validate_inputs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate input data"
                },
                "validate_outputs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate output data"
                },
                "copy_tensors": {
                    "type": "boolean",
                    "default": False,
                    "description": "Create copies of embedding tensors"
                },
                "log_operations": {
                    "type": "boolean",
                    "default": False,
                    "description": "Log passthrough operations"
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
            # Passthrough component is always healthy
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
            self._total_processing_time / self._total_processed 
            if self._total_processed > 0 else 0.0
        )
        
        return {
            "component_name": self.name,
            "component_version": self.version,
            "enhancement_method": "passthrough",
            "total_processed": self._total_processed,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "preserve_original": self._preserve_original,
            "add_metadata": self._add_metadata,
            "validate_inputs": self._validate_inputs,
            "validate_outputs": self._validate_outputs,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def enhance(self, input_data: GraphEnhancementInput) -> GraphEnhancementOutput:
        """
        Pass through embeddings without enhancement according to the contract.
        
        Args:
            input_data: Input data conforming to GraphEnhancementInput contract
            
        Returns:
            Output data conforming to GraphEnhancementOutput contract
        """
        errors = []
        enhanced_embeddings = []
        
        try:
            start_time = datetime.utcnow()
            
            # Validate inputs if enabled
            if self._validate_inputs:
                validation_errors = self._validate_input_data(input_data)
                if validation_errors:
                    errors.extend(validation_errors)
            
            # Process embeddings (passthrough)
            for chunk_embedding in input_data.embeddings:
                try:
                    enhanced_emb = self._passthrough_embedding(chunk_embedding)
                    enhanced_embeddings.append(enhanced_emb)
                    
                except Exception as e:
                    error_msg = f"Failed to process embedding {chunk_embedding.chunk_id}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Validate outputs if enabled
            if self._validate_outputs:
                output_validation_errors = self._validate_output_data(enhanced_embeddings)
                if output_validation_errors:
                    errors.extend(output_validation_errors)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._total_processed += len(input_data.embeddings)
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
            
            return GraphEnhancementOutput(
                enhanced_embeddings=enhanced_embeddings,
                metadata=metadata,
                graph_stats={
                    "processing_time": processing_time,
                    "embeddings_processed": len(enhanced_embeddings),
                    "embeddings_failed": len(errors),
                    "enhancement_method": "passthrough",
                    "preserve_original": self._preserve_original,
                    "throughput_embeddings_per_second": len(enhanced_embeddings) / max(processing_time, 0.001)
                },
                model_info={
                    "model_type": "passthrough",
                    "enhancement_method": "none",
                    "preserve_original": self._preserve_original,
                    "validation_enabled": self._validate_inputs and self._validate_outputs
                },
                errors=errors,
                enhancement_quality_score=1.0  # Perfect preservation
            )
            
        except Exception as e:
            error_msg = f"None/passthrough ISNE processing failed: {str(e)}"
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
        No-op training for passthrough component.
        
        Args:
            training_data: Training data (ignored)
        """
        self.logger.info("No training required for passthrough component")
        # No actual training needed for passthrough
        pass
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return True  # Passthrough is always "trained"
    
    def save_model(self, path: str) -> None:
        """
        Save component configuration to disk.
        
        Args:
            path: Path to save configuration
        """
        try:
            import json
            
            save_data = {
                "component_type": "none_isne",
                "component_name": self.name,
                "component_version": self.version,
                "config": self._config,
                "statistics": {
                    "total_processed": self._total_processed,
                    "total_processing_time": self._total_processing_time
                },
                "saved_at": datetime.utcnow().isoformat()
            }
            
            with open(path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info(f"Component configuration saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def load_model(self, path: str) -> None:
        """
        Load component configuration from disk.
        
        Args:
            path: Path to load configuration from
        """
        try:
            import json
            
            with open(path, 'r') as f:
                save_data = json.load(f)
            
            # Update configuration
            if 'config' in save_data:
                self._config.update(save_data['config'])
                self.configure(self._config)
            
            # Restore statistics if available
            if 'statistics' in save_data:
                stats = save_data['statistics']
                self._total_processed = stats.get('total_processed', 0)
                self._total_processing_time = stats.get('total_processing_time', 0.0)
            
            self.logger.info(f"Component configuration loaded from {path}")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            self.logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the passthrough component.
        
        Returns:
            Dictionary containing component metadata
        """
        return {
            "model_type": "passthrough",
            "component_name": self.name,
            "component_version": self.version,
            "enhancement_method": "none",
            "preserve_original": self._preserve_original,
            "add_metadata": self._add_metadata,
            "validate_inputs": self._validate_inputs,
            "validate_outputs": self._validate_outputs,
            "total_processed": self._total_processed,
            "avg_processing_time": (
                self._total_processing_time / self._total_processed 
                if self._total_processed > 0 else 0.0
            )
        }
    
    def supports_incremental_training(self) -> bool:
        """Check if the model supports incremental training."""
        return False  # No training needed for passthrough
    
    def _passthrough_embedding(self, chunk_embedding: Any) -> EnhancedEmbedding:
        """Create a passthrough enhanced embedding."""
        try:
            # Copy or reference original embedding
            if self._copy_tensors:
                enhanced_embedding = chunk_embedding.embedding.copy()
            else:
                enhanced_embedding = chunk_embedding.embedding
            
            # Add metadata if enabled
            metadata = chunk_embedding.metadata.copy() if chunk_embedding.metadata else {}
            if self._add_metadata:
                metadata.update({
                    "enhancement_method": "none",
                    "enhancement_timestamp": datetime.utcnow().isoformat(),
                    "original_embedding_preserved": self._preserve_original
                })
            
            return EnhancedEmbedding(
                chunk_id=chunk_embedding.chunk_id,
                original_embedding=chunk_embedding.embedding if self._preserve_original else [],
                enhanced_embedding=enhanced_embedding,
                graph_features={
                    "enhancement_method": "passthrough",
                    "preserve_original": self._preserve_original,
                    "processing_time": 0.0  # Minimal processing time
                },
                enhancement_score=1.0,  # Perfect preservation
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create passthrough embedding: {e}")
            # Return minimal valid embedding on error
            return EnhancedEmbedding(
                chunk_id=chunk_embedding.chunk_id,
                original_embedding=chunk_embedding.embedding,
                enhanced_embedding=chunk_embedding.embedding,
                graph_features={"enhancement_method": "error_fallback", "error": str(e)},
                enhancement_score=0.0,
                metadata=chunk_embedding.metadata or {}
            )
    
    def _validate_input_data(self, input_data: GraphEnhancementInput) -> List[str]:
        """Validate input data."""
        errors = []
        
        try:
            if not input_data.embeddings:
                errors.append("No embeddings provided in input data")
            
            for i, embedding in enumerate(input_data.embeddings):
                if not embedding.chunk_id:
                    errors.append(f"Embedding {i} missing chunk_id")
                
                if not embedding.embedding or not isinstance(embedding.embedding, list):
                    errors.append(f"Embedding {i} has invalid embedding data")
                
                if len(embedding.embedding) == 0:
                    errors.append(f"Embedding {i} has empty embedding vector")
            
        except Exception as e:
            errors.append(f"Input validation error: {str(e)}")
        
        return errors
    
    def _validate_output_data(self, enhanced_embeddings: List[EnhancedEmbedding]) -> List[str]:
        """Validate output data."""
        errors = []
        
        try:
            if not enhanced_embeddings:
                errors.append("No enhanced embeddings generated")
            
            for i, embedding in enumerate(enhanced_embeddings):
                if not embedding.chunk_id:
                    errors.append(f"Enhanced embedding {i} missing chunk_id")
                
                if not embedding.enhanced_embedding or not isinstance(embedding.enhanced_embedding, list):
                    errors.append(f"Enhanced embedding {i} has invalid enhanced_embedding data")
                
                if len(embedding.enhanced_embedding) == 0:
                    errors.append(f"Enhanced embedding {i} has empty enhanced_embedding vector")
                
                if embedding.enhancement_score is None or embedding.enhancement_score < 0:
                    errors.append(f"Enhanced embedding {i} has invalid enhancement_score")
            
        except Exception as e:
            errors.append(f"Output validation error: {str(e)}")
        
        return errors