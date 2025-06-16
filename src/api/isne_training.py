"""
ISNE Training Pipeline Implementation

This module implements the production ISNE training pipeline for the API.
It provides a clean interface for training ISNE models without the bootstrap process,
suitable for periodic model updates and production use.
"""

import asyncio
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import torch
import traceback

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer
from src.isne.bootstrap.config import WandBConfig
from src.isne.bootstrap.wandb_logger import WandBLogger
from src.alerts.alert_manager import AlertManager
from src.alerts import AlertLevel
from src.config.config_loader import load_config
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class ISNEProductionTrainer:
    """Production ISNE training pipeline."""
    
    def __init__(self, alert_manager: Optional[AlertManager] = None):
        """
        Initialize the production trainer.
        
        Args:
            alert_manager: Optional alert manager for notifications
        """
        self.alert_manager = alert_manager or AlertManager()
        self.wandb_logger = None
        self.wandb_run = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_wandb(self, training_config: Dict[str, Any], model_name: str) -> bool:
        """
        Initialize Weights & Biases logging.
        
        Args:
            training_config: Training configuration
            model_name: Name for this training run
            
        Returns:
            True if initialized successfully
        """
        try:
            # Try to initialize wandb directly
            if HAS_WANDB:
                import wandb
                
                # Initialize W&B run directly
                run_config = {
                    "training_config": training_config,
                    "device": str(self.device),
                    "production_training": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.wandb_run = wandb.init(
                    project="hades-isne-training",
                    name=f"{model_name}_production_training",
                    config=run_config,
                    tags=["isne", "production", "hades"],
                    notes="ISNE model training for HADES graph-RAG system",
                    reinit=True
                )
                
                logger.info(f"Started W&B run: {self.wandb_run.name}")
                return True
            else:
                logger.warning("W&B not available")
                return False
            
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            return False
    
    def load_graph_data(self, graph_data_path: str) -> Tuple[Dict[str, Any], Data]:
        """
        Load graph data from file.
        
        Args:
            graph_data_path: Path to graph data JSON file
            
        Returns:
            Tuple of (graph_data_dict, torch_geometric_data)
        """
        logger.info(f"Loading graph data from: {graph_data_path}")
        
        with open(graph_data_path, 'r') as f:
            graph_data = json.load(f)
        
        # Extract nodes and edges
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        logger.info(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")
        
        # Convert to PyTorch Geometric format
        embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32)
        
        # Build edge index
        edge_index = []
        edge_weights = []
        
        for edge in edges:
            edge_index.append([edge['source'], edge['target']])
            edge_weights.append(edge['weight'])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        # Create PyTorch Geometric data object
        data = Data(x=embeddings, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph_data, data
    
    def load_existing_model(self, model_path: Optional[str], num_nodes: int, embedding_dim: int) -> ISNEModel:
        """
        Load existing model or create new one.
        
        Args:
            model_path: Optional path to existing model
            num_nodes: Number of nodes in graph
            embedding_dim: Embedding dimension
            
        Returns:
            ISNE model instance
        """
        if model_path and Path(model_path).exists():
            logger.info(f"Loading existing model from: {model_path}")
            model = ISNEModel.load(model_path)
            
            # Verify model compatibility
            if model.num_nodes != num_nodes:
                logger.warning(f"Model node count mismatch: {model.num_nodes} vs {num_nodes}")
                logger.info("Creating new model instead")
                model = ISNEModel(num_nodes, embedding_dim)
        else:
            logger.info(f"Creating new ISNE model: {num_nodes} nodes, {embedding_dim}D")
            model = ISNEModel(num_nodes, embedding_dim)
        
        return model
    
    async def train(
        self,
        job_id: str,
        job_data: Dict[str, Any],
        graph_data_path: str,
        output_dir: str,
        training_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute ISNE training.
        
        Args:
            job_id: Training job ID
            job_data: Job tracking data
            graph_data_path: Path to graph data
            output_dir: Output directory for model
            training_config: Training configuration overrides
            model_path: Optional path to existing model for continued training
            model_name: Name for this training run
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        model_name = model_name or f"isne_model_{job_id}"
        
        try:
            # Update job status
            job_data["status"] = "running"
            job_data["stage"] = "initialization"
            job_data["progress_percent"] = 5.0
            
            # Initialize W&B if configured
            wandb_enabled = self.initialize_wandb(training_config or {}, model_name)
            
            # Load training configuration
            default_config = {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 4,
                "dropout": 0.1,
                "weight_decay": 1e-5,
                "patience": 10,
                "min_delta": 0.001
            }
            
            if training_config:
                default_config.update(training_config)
            
            config = default_config
            
            # Load graph data
            job_data["stage"] = "data_loading"
            job_data["progress_percent"] = 10.0
            
            graph_dict, graph_data = self.load_graph_data(graph_data_path)
            num_nodes = len(graph_dict['nodes'])
            num_edges = len(graph_dict['edges'])
            embedding_dim = len(graph_dict['nodes'][0]['embedding'])
            
            # Log data statistics
            data_stats = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "embedding_dim": embedding_dim,
                "graph_density": (2 * num_edges) / (num_nodes * (num_nodes - 1))
            }
            
            job_data["data_stats"] = data_stats
            
            if self.wandb_run and HAS_WANDB:
                import wandb
                wandb.log({f"data_loading/{k}": v for k, v in data_stats.items() if isinstance(v, (int, float))})
            
            # Initialize trainer
            job_data["stage"] = "model_initialization"
            job_data["progress_percent"] = 15.0
            
            trainer = ISNETrainer(
                embedding_dim=embedding_dim,
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config["dropout"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                device=self.device
            )
            
            # Initialize the model
            trainer._initialize_model(num_nodes)
            
            # Load existing model if provided
            if model_path and Path(model_path).exists():
                logger.info(f"Loading existing model from: {model_path}")
                loaded_model = ISNEModel.load(model_path)
                trainer.model = loaded_model.to(self.device)
            
            # Log model info
            model_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {model_params:,}")
            
            if self.wandb_run and HAS_WANDB:
                import wandb
                wandb.log({
                    "model_init/parameters": model_params,
                    "model_init/layers": config["num_layers"],
                    "model_init/hidden_dim": config["hidden_dim"]
                })
            
            # Training
            job_data["stage"] = "training"
            job_data["progress_percent"] = 20.0
            
            # Use the simple ISNE training method
            training_start = time.time()
            training_results = trainer.train_isne_simple(
                edge_index=graph_data.edge_index,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                verbose=True
            )
            training_time = time.time() - training_start
            
            # Extract results
            final_loss = training_results.get("final_loss", 0.0)
            best_loss = training_results.get("best_loss", final_loss)
            training_history = training_results.get("loss_history", [])
            
            # Log to W&B
            if self.wandb_run and HAS_WANDB:
                import wandb
                for epoch, loss in enumerate(training_history):
                    wandb.log({
                        "training/loss": loss,
                        "training/duration_seconds": training_time / len(training_history),
                        "epoch": epoch + 1
                    }, step=epoch + 1)
            
            # Save final model
            job_data["stage"] = "model_saving"
            job_data["progress_percent"] = 90.0
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            final_model_path = output_path / f"{model_name}_final.pth"
            trainer.model.save(str(final_model_path))
            
            # Run validation
            job_data["stage"] = "validation"
            job_data["progress_percent"] = 95.0
            
            validation_results = self.validate_model(trainer.model, graph_data)
            
            # Prepare results
            training_time = time.time() - start_time
            results = {
                "success": True,
                "model_path": str(final_model_path),
                "best_model_path": str(final_model_path),  # For now, same as final
                "training_epochs": len(training_history),
                "best_epoch": len(training_history),
                "final_loss": final_loss,
                "best_loss": best_loss,
                "training_time_seconds": training_time,
                "model_parameters": model_params,
                "data_stats": data_stats,
                "validation_results": validation_results,
                "training_history": training_history
            }
            
            # Log final results to W&B
            if self.wandb_run and HAS_WANDB:
                import wandb
                
                # Log summary metrics
                wandb.log({
                    "final/num_nodes": num_nodes,
                    "final/num_edges": num_edges,
                    "final/embedding_dimension": embedding_dim,
                    "final/model_parameters": model_params,
                    "final/graph_density": data_stats["graph_density"],
                    "final/total_epochs": len(training_history),
                    "final/final_loss": results["final_loss"],
                    "final/best_loss": best_loss,
                    "final/training_time": training_time,
                    "final/validation_similarity": validation_results.get("similarity_preservation", 0)
                })
                
                # Log model as artifact
                model_artifact = wandb.Artifact(
                    name=f"isne_model_{model_name}",
                    type="model",
                    description=f"ISNE model: {num_nodes} nodes, {num_edges} edges"
                )
                model_artifact.add_file(str(final_model_path))
                wandb.log_artifact(model_artifact)
            
            # Send success alert
            self.alert_manager.alert(
                message=f"ISNE training completed successfully: {model_name}",
                level=AlertLevel.LOW,
                source="isne_training",
                context={
                    "job_id": job_id,
                    "epochs": len(training_history),
                    "final_loss": results["final_loss"],
                    "training_time_minutes": training_time / 60
                }
            )
            
            # Update job completion
            job_data["status"] = "completed"
            job_data["progress_percent"] = 100.0
            job_data["stage"] = "completed"
            job_data["results"] = results
            
            return results
            
        except Exception as e:
            error_msg = f"ISNE training failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            # Send failure alert
            self.alert_manager.alert(
                message=error_msg,
                level=AlertLevel.HIGH,
                source="isne_training",
                context={
                    "job_id": job_id,
                    "error": str(e),
                    "stage": job_data.get("stage", "unknown")
                }
            )
            
            # Update job failure
            job_data["status"] = "failed"
            job_data["error_message"] = error_msg
            job_data["results"] = {
                "success": False,
                "error": str(e),
                "training_time_seconds": time.time() - start_time
            }
            
            raise
            
        finally:
            # Finish W&B run
            if self.wandb_run and HAS_WANDB:
                import wandb
                wandb.finish(exit_code=0 if job_data["status"] == "completed" else 1)
    
    def validate_model(self, model: ISNEModel, graph_data: Data) -> Dict[str, Any]:
        """
        Validate trained model performance.
        
        Args:
            model: Trained ISNE model
            graph_data: Graph data
            
        Returns:
            Validation metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Get enhanced embeddings
            enhanced = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            
            # Calculate basic metrics
            embedding_norm = torch.norm(enhanced, dim=1).mean().item()
            embedding_std = enhanced.std().item()
            
            # Calculate cosine similarity preservation
            original_sim = torch.mm(graph_data.x, graph_data.x.t())
            enhanced_sim = torch.mm(enhanced, enhanced.t())
            
            # Normalize similarities
            original_sim = original_sim / (torch.norm(graph_data.x, dim=1, keepdim=True) @ 
                                         torch.norm(graph_data.x, dim=1, keepdim=True).t())
            enhanced_sim = enhanced_sim / (torch.norm(enhanced, dim=1, keepdim=True) @ 
                                         torch.norm(enhanced, dim=1, keepdim=True).t())
            
            similarity_correlation = torch.corrcoef(
                torch.stack([original_sim.flatten(), enhanced_sim.flatten()])
            )[0, 1].item()
        
        return {
            "embedding_norm_mean": embedding_norm,
            "embedding_std": embedding_std,
            "similarity_preservation": similarity_correlation,
            "graph_density": graph_data.edge_index.shape[1] / (graph_data.x.shape[0] ** 2)
        }


# Export the trainer class
__all__ = ['ISNEProductionTrainer']