"""
ISNE Model Updater for Incremental Storage

Handles expanding ISNE model capacity and performing incremental training
with new data while preserving existing model knowledge.
"""

import logging
import asyncio
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from arango.database import StandardDatabase

from .types import (
    IncrementalConfig, 
    ModelUpdateResult, 
    VersionInfo,
    ModelExpansionStrategy
)
from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer
from src.alerts import AlertManager, AlertLevel

logger = logging.getLogger(__name__)


class ModelUpdater:
    """
    Manages ISNE model updates for incremental storage.
    
    Handles:
    1. Model capacity expansion for new nodes
    2. Incremental training with new graph data
    3. Model versioning and rollback capabilities
    4. Performance monitoring and validation
    """
    
    def __init__(
        self, 
        db: StandardDatabase, 
        config: IncrementalConfig,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Initialize model updater.
        
        Args:
            db: ArangoDB database connection
            config: Incremental storage configuration
            alert_manager: Alert manager for notifications
        """
        self.db = db
        self.config = config
        self.alert_manager = alert_manager or AlertManager()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.expansion_strategy = config.model_expansion_strategy
        self.expansion_factor = config.expansion_factor
        
        # Current model state
        self.current_model: Optional[ISNEModel] = None
        self.current_version: Optional[str] = None
        self.current_node_count: int = 0
        
    async def update_incremental(
        self, 
        new_nodes: int, 
        batch_id: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> ModelUpdateResult:
        """
        Update ISNE model incrementally with new nodes.
        
        Args:
            new_nodes: Number of new nodes to add
            batch_id: Batch identifier for tracking
            training_config: Optional training configuration overrides
            
        Returns:
            Model update results
        """
        logger.info(f"Starting incremental model update: +{new_nodes} nodes")
        
        start_time = datetime.now()
        training_job_id = f"incremental_{batch_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Load current model
            await self._load_current_model()
            
            if self.current_model is None:
                logger.warning("No current model found, creating new model")
                return await self._create_initial_model(new_nodes, training_job_id)
            
            # Check if expansion is needed
            expansion_needed = await self._check_expansion_needed(new_nodes)
            
            if expansion_needed:
                # Expand model capacity
                expanded_model = await self._expand_model_capacity(new_nodes)
            else:
                expanded_model = self.current_model
            
            # Perform incremental training
            training_result = await self._train_incremental(
                expanded_model, 
                training_job_id,
                training_config
            )
            
            # Save updated model
            new_version = await self._save_model_version(
                expanded_model,
                training_result,
                training_job_id,
                new_nodes
            )
            
            # Update model metadata
            await self._update_model_metadata(new_version, training_result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ModelUpdateResult(
                success=True,
                previous_version=self.current_version or "none",
                new_version=new_version,
                previous_node_count=self.current_node_count,
                new_node_count=self.current_node_count + new_nodes,
                nodes_added=new_nodes,
                training_epochs=training_result.get('epochs', 0),
                final_loss=training_result.get('final_loss', 0.0),
                convergence_achieved=training_result.get('converged', False),
                training_time_seconds=processing_time,
                model_size_mb=training_result.get('model_size_mb', 0.0),
                model_path=training_result.get('model_path', ''),
                backup_path=training_result.get('backup_path'),
                metadata={
                    'expansion_strategy': self.expansion_strategy.value,
                    'training_job_id': training_job_id,
                    'batch_id': batch_id
                }
            )
            
            # Send success alert
            self.alert_manager.alert(
                message=f"ISNE model updated: +{new_nodes} nodes, version {new_version}",
                level=AlertLevel.LOW,
                source="model_updater",
                context={
                    'new_version': new_version,
                    'nodes_added': new_nodes,
                    'training_time': processing_time
                }
            )
            
            logger.info(f"Model update completed: version {new_version}")
            return result
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            
            # Log training failure
            await self._log_training_failure(training_job_id, str(e))
            
            # Send error alert
            self.alert_manager.alert(
                message=f"ISNE model update failed: {e}",
                level=AlertLevel.HIGH,
                source="model_updater",
                context={
                    'training_job_id': training_job_id,
                    'error': str(e),
                    'new_nodes': new_nodes
                }
            )
            
            raise
    
    async def _load_current_model(self) -> bool:
        """
        Load the current ISNE model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            # Get current model info from database
            models_collection = self.db.collection('models')
            
            query = """
            FOR model IN models
            FILTER model.is_current == true AND model.model_type == 'isne'
            LIMIT 1
            RETURN model
            """
            
            results = list(self.db.query(query))
            
            if not results:
                logger.info("No current ISNE model found")
                return False
            
            model_info = results[0]
            model_path = model_info['model_path']
            
            # Load model from file
            if Path(model_path).exists():
                self.current_model = ISNEModel.load(model_path)
                self.current_version = model_info['version_id']
                self.current_node_count = model_info['node_count']
                
                logger.info(f"Loaded current model: {self.current_version} ({self.current_node_count} nodes)")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load current model: {e}")
            return False
    
    async def _check_expansion_needed(self, new_nodes: int) -> bool:
        """
        Check if model capacity expansion is needed.
        
        Args:
            new_nodes: Number of new nodes to add
            
        Returns:
            True if expansion needed
        """
        if self.current_model is None:
            return True
        
        current_capacity = self.current_model.num_nodes
        required_capacity = self.current_node_count + new_nodes
        
        # Check if current capacity can accommodate new nodes
        if required_capacity > current_capacity:
            logger.info(f"Model expansion needed: {required_capacity} > {current_capacity}")
            return True
        
        # Check expansion strategy
        if self.expansion_strategy == ModelExpansionStrategy.GRADUAL:
            # Expand if utilization > 80%
            utilization = required_capacity / current_capacity
            if utilization > 0.8:
                logger.info(f"Gradual expansion triggered: {utilization:.2%} utilization")
                return True
        
        return False
    
    async def _expand_model_capacity(self, new_nodes: int) -> ISNEModel:
        """
        Expand model capacity to accommodate new nodes.
        
        Args:
            new_nodes: Number of new nodes to add
            
        Returns:
            Expanded model
        """
        logger.info(f"Expanding model capacity for {new_nodes} new nodes")
        
        # Calculate new capacity based on strategy
        current_capacity = self.current_model.num_nodes
        required_capacity = self.current_node_count + new_nodes
        
        if self.expansion_strategy == ModelExpansionStrategy.IMMEDIATE:
            new_capacity = required_capacity
        elif self.expansion_strategy == ModelExpansionStrategy.GRADUAL:
            new_capacity = max(
                required_capacity,
                int(current_capacity * self.expansion_factor)
            )
        else:  # ADAPTIVE
            # Adaptive expansion based on growth rate
            growth_rate = new_nodes / self.current_node_count if self.current_node_count > 0 else 1.0
            expansion_multiplier = min(2.0, 1.0 + growth_rate)
            new_capacity = int(required_capacity * expansion_multiplier)
        
        logger.info(f"Expanding model capacity: {current_capacity} -> {new_capacity}")
        
        # Create new model with expanded capacity
        expanded_model = ISNEModel(
            num_nodes=new_capacity,
            embedding_dim=self.current_model.embedding_dim,
            learning_rate=self.current_model.learning_rate,
            negative_samples=self.current_model.negative_samples,
            context_window=self.current_model.context_window,
            device=self.device
        )
        
        # Copy existing parameters
        with torch.no_grad():
            # Copy theta parameters for existing nodes
            existing_nodes = min(current_capacity, self.current_node_count)
            if existing_nodes > 0:
                expanded_model.theta.data[:existing_nodes] = self.current_model.theta.data[:existing_nodes]
            
            # Initialize new node parameters
            if new_capacity > existing_nodes:
                # Initialize new parameters with small random values
                expanded_model.theta.data[existing_nodes:].normal_(0, 0.01)
        
        logger.info(f"Model capacity expanded successfully: {new_capacity} nodes")
        return expanded_model
    
    async def _train_incremental(
        self,
        model: ISNEModel,
        training_job_id: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform incremental training on the model.
        
        Args:
            model: Model to train
            training_job_id: Training job identifier
            training_config: Training configuration
            
        Returns:
            Training results
        """
        logger.info("Starting incremental training")
        
        # Log training start
        await self._log_training_start(training_job_id, model)
        
        # Get training data
        graph_data = await self._get_graph_data_for_training()
        
        if graph_data is None:
            logger.warning("No graph data available for training")
            return {
                'epochs': 0,
                'final_loss': 0.0,
                'converged': False,
                'model_path': '',
                'model_size_mb': 0.0
            }
        
        # Configure training
        default_config = {
            'epochs': 20,  # Fewer epochs for incremental updates
            'learning_rate': 0.001,
            'batch_size': 128,
            'patience': 10,
            'min_delta': 0.001
        }
        
        if training_config:
            default_config.update(training_config)
        
        # Initialize trainer
        trainer = ISNETrainer(
            embedding_dim=model.embedding_dim,
            hidden_dim=128,  # Use same as model
            num_layers=3,
            learning_rate=default_config['learning_rate'],
            device=self.device
        )
        
        # Replace trainer's model with our expanded model
        trainer.model = model.to(self.device)
        
        # Perform training
        training_results = trainer.train_isne_simple(
            edge_index=graph_data['edge_index'],
            epochs=default_config['epochs'],
            batch_size=default_config['batch_size'],
            verbose=True
        )
        
        # Save model
        model_dir = Path("models/incremental")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{training_job_id}_model.pth"
        
        model.save(str(model_path))
        
        # Calculate model size
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        result = {
            'epochs': len(training_results.get('loss_history', [])),
            'final_loss': training_results.get('final_loss', 0.0),
            'best_loss': training_results.get('best_loss', 0.0),
            'converged': training_results.get('converged', False),
            'model_path': str(model_path),
            'model_size_mb': model_size_mb,
            'loss_history': training_results.get('loss_history', [])
        }
        
        logger.info(f"Incremental training completed: {result['epochs']} epochs, final loss: {result['final_loss']:.6f}")
        return result
    
    async def _get_graph_data_for_training(self) -> Optional[Dict[str, Any]]:
        """
        Get graph data for training.
        
        Returns:
            Graph data dictionary or None if no data
        """
        try:
            # Get recent edges for training
            query = """
            FOR edge IN edges
            RETURN {
                from_node: PARSE_IDENTIFIER(edge._from).key,
                to_node: PARSE_IDENTIFIER(edge._to).key,
                weight: edge.weight
            }
            """
            
            edges = list(self.db.query(query))
            
            if not edges:
                return None
            
            # Convert to PyTorch format
            edge_list = []
            for edge in edges:
                try:
                    from_idx = int(edge['from_node'])
                    to_idx = int(edge['to_node'])
                    edge_list.append([from_idx, to_idx])
                except (ValueError, KeyError):
                    continue
            
            if not edge_list:
                return None
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            
            return {
                'edge_index': edge_index,
                'num_edges': len(edge_list)
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph data for training: {e}")
            return None
    
    async def _save_model_version(
        self,
        model: ISNEModel,
        training_result: Dict[str, Any],
        training_job_id: str,
        new_nodes: int
    ) -> str:
        """
        Save new model version to database.
        
        Args:
            model: Trained model
            training_result: Training results
            training_job_id: Training job ID
            new_nodes: Number of new nodes added
            
        Returns:
            New version ID
        """
        models_collection = self.db.collection('models')
        
        # Generate version ID
        version_number = await self._get_next_version_number()
        version_id = f"v{version_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mark previous model as not current
        if self.current_version:
            models_collection.update(
                {'version_id': self.current_version},
                {'is_current': False}
            )
        
        # Create new model record
        model_doc = {
            '_key': version_id,
            'version_id': version_id,
            'version_number': version_number,
            'model_type': 'isne',
            'node_count': self.current_node_count + new_nodes,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': 128,  # From training config
            'num_layers': 3,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_path': training_result['model_path'],
            'training_config': {
                'training_job_id': training_job_id,
                'incremental_update': True,
                'nodes_added': new_nodes
            },
            'performance_metrics': {
                'final_loss': training_result['final_loss'],
                'best_loss': training_result.get('best_loss', 0.0),
                'training_epochs': training_result['epochs'],
                'converged': training_result['converged']
            },
            'created_at': datetime.now().isoformat(),
            'created_by': 'incremental_updater',
            'description': f'Incremental update: +{new_nodes} nodes',
            'parent_version': self.current_version,
            'is_current': True,
            'metadata': {
                'expansion_strategy': self.expansion_strategy.value,
                'model_size_mb': training_result['model_size_mb']
            }
        }
        
        models_collection.insert(model_doc)
        
        logger.info(f"Saved new model version: {version_id}")
        return version_id
    
    async def _get_next_version_number(self) -> str:
        """
        Get next version number.
        
        Returns:
            Next version number as string
        """
        query = """
        FOR model IN models
        FILTER model.model_type == 'isne'
        SORT model.version_number DESC
        LIMIT 1
        RETURN model.version_number
        """
        
        results = list(self.db.query(query))
        
        if results:
            try:
                last_version = int(results[0])
                return str(last_version + 1)
            except (ValueError, TypeError):
                pass
        
        return "1"
    
    async def _update_model_metadata(
        self, 
        version_id: str, 
        training_result: Dict[str, Any]
    ) -> None:
        """
        Update model metadata after training.
        
        Args:
            version_id: Model version ID
            training_result: Training results
        """
        # Update current model state
        self.current_version = version_id
        self.current_node_count = self.current_model.num_nodes if self.current_model else 0
        
        # Log performance metrics
        logger.info(f"Model {version_id} performance:")
        logger.info(f"  Final loss: {training_result.get('final_loss', 0.0):.6f}")
        logger.info(f"  Epochs: {training_result.get('epochs', 0)}")
        logger.info(f"  Converged: {training_result.get('converged', False)}")
    
    async def _create_initial_model(
        self, 
        node_count: int, 
        training_job_id: str
    ) -> ModelUpdateResult:
        """
        Create initial model when no current model exists.
        
        Args:
            node_count: Number of nodes for initial model
            training_job_id: Training job ID
            
        Returns:
            Model update result
        """
        logger.info(f"Creating initial ISNE model with {node_count} nodes")
        
        # Create new model
        model = ISNEModel(
            num_nodes=node_count,
            embedding_dim=128,  # Default dimension
            device=self.device
        )
        
        # Perform initial training
        training_result = await self._train_incremental(model, training_job_id)
        
        # Save as first version
        version_id = await self._save_model_version(
            model, training_result, training_job_id, node_count
        )
        
        return ModelUpdateResult(
            success=True,
            previous_version="none",
            new_version=version_id,
            previous_node_count=0,
            new_node_count=node_count,
            nodes_added=node_count,
            training_epochs=training_result['epochs'],
            final_loss=training_result['final_loss'],
            convergence_achieved=training_result['converged'],
            training_time_seconds=0.0,  # Would need to track separately
            model_size_mb=training_result['model_size_mb'],
            model_path=training_result['model_path'],
            metadata={'initial_model': True}
        )
    
    async def _log_training_start(
        self, 
        training_job_id: str, 
        model: ISNEModel
    ) -> None:
        """Log training start."""
        collection = self.db.collection('model_versions')
        
        log_entry = {
            '_key': training_job_id,
            'model_id': self.current_version or 'initial',
            'training_job_id': training_job_id,
            'start_time': datetime.now().isoformat(),
            'training_data_hash': 'incremental',
            'data_version': 'current',
            'training_config': {
                'incremental_update': True,
                'model_capacity': model.num_nodes,
                'embedding_dim': model.embedding_dim
            },
            'status': 'training',
            'metadata': {}
        }
        
        collection.insert(log_entry)
    
    async def _log_training_failure(
        self, 
        training_job_id: str, 
        error: str
    ) -> None:
        """Log training failure."""
        collection = self.db.collection('model_versions')
        
        try:
            collection.update(training_job_id, {
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error_message': error
            })
        except Exception as e:
            logger.warning(f"Failed to log training failure: {e}")
    
    async def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            models_collection = self.db.collection('models')
            
            # Get target version
            target_model = models_collection.get(version_id)
            if not target_model:
                logger.error(f"Version not found: {version_id}")
                return False
            
            # Mark current model as not current
            if self.current_version:
                models_collection.update(
                    {'version_id': self.current_version},
                    {'is_current': False}
                )
            
            # Mark target as current
            models_collection.update(version_id, {'is_current': True})
            
            # Update internal state
            self.current_version = version_id
            self.current_node_count = target_model['node_count']
            self.current_model = None  # Will be loaded on next use
            
            logger.info(f"Rolled back to model version: {version_id}")
            
            # Send rollback alert
            self.alert_manager.alert(
                message=f"Model rolled back to version: {version_id}",
                level=AlertLevel.MEDIUM,
                source="model_updater",
                context={'version_id': version_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False