"""
ISNE Pipeline with integrated validation.

This module provides a production-ready ISNE pipeline implementation with
built-in validation to ensure embedding consistency and quality.
"""

import os
import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast, Tuple, Type
from datetime import datetime
from torch import Tensor
import torch.nn as nn

# Fix the import paths for model classes
from src.isne.models.isne_model import ISNEModel
from src.isne.models.simplified_isne_model import SimplifiedISNEModel
from typing import Protocol

# Define a Protocol for models to ensure correct type checking
class ISNEModelProtocol(Protocol):
    """Protocol defining the interface for ISNE model types."""
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor: ...
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any: ...
    def to(self, device: Union[str, torch.device]) -> Any: ...
    def eval(self) -> Any: ...
    def __call__(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor: ...

# Set up logging
logger = logging.getLogger(__name__)

# Import validation utilities
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)
from src.isne.utils.geometric_utils import create_graph_from_documents
from src.alerts import AlertManager, AlertLevel


class ISNEPipeline:
    """
    Production-ready ISNE pipeline with integrated validation.
    
    This class provides a robust implementation for applying ISNE embeddings to
    documents with comprehensive validation and alerting to ensure data quality.
    """
    
    def __init__(
            self, 
            model_path: Optional[str] = None,
            validate: bool = True,
            alert_threshold: str = "high",
            device: Optional[str] = None,
            alert_dir: str = "./alerts",
            alert_manager: Optional[AlertManager] = None
        ):
        """
        Initialize the ISNE pipeline.
        
        Args:
            model_path: Path to the trained ISNE model
            validate: Whether to perform validation during processing
            alert_threshold: Threshold for alerting on validation issues ("low", "medium", "high")
            device: Device to use for processing ("cpu", "cuda:0", etc.)
            alert_dir: Directory to store alert logs
            alert_manager: Optional pre-configured AlertManager instance
        """
        self.model_path = model_path
        self.validate = validate
        self.alert_threshold = alert_threshold
        self.device = device or self._determine_device()
        # Initialize model variable with a specific type
        self.model: Optional[nn.Module] = None
        
        # Alert configuration
        self.alert_levels = {
            "low": 1,     # Alert on any discrepancy
            "medium": 5,  # Alert on 5+ discrepancies
            "high": 10    # Alert on 10+ discrepancies
        }
        self.alert_threshold_value = self.alert_levels.get(alert_threshold, 10)
        
        # Initialize or use provided alert manager
        if alert_manager:
            self.alert_manager = alert_manager
        else:
            # Create alert directory
            Path(alert_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize alert manager with appropriate minimum level
            min_level = AlertLevel.LOW
            if alert_threshold == "medium":
                min_level = AlertLevel.MEDIUM
            elif alert_threshold == "high":
                min_level = AlertLevel.HIGH
                
            self.alert_manager = AlertManager(
                alert_dir=alert_dir,
                min_level=min_level
            )
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _determine_device(self) -> str:
        """
        Determine the best device to use for processing.
        
        Returns:
            Device string ("cpu" or "cuda:X")
        """
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    
    def load_model(self, model_path: str) -> None:
        """
        Load ISNE model from a file.
        
        Args:
            model_path: Path to the trained model
        """
        try:
            # Load model checkpoint
            model_data = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            model_config = model_data.get("model_config", {})
            model_type = model_config.get("type", "standard")
            embedding_dim = model_config.get("embedding_dim", 768)
            hidden_dim = model_config.get("hidden_dim", 256)
            output_dim = model_config.get("output_dim", 128)
            num_layers = model_config.get("num_layers", 2)
            
            # Initialize model based on type - use nn.Module to ensure compatibility
            if model_type == "simplified":
                logger.info("Using SimplifiedISNEModel")
                # Create a specific type instance that is a subclass of nn.Module
                simplified_model: nn.Module = SimplifiedISNEModel(
                    in_features=embedding_dim,
                    hidden_features=hidden_dim,
                    out_features=output_dim
                )
                self.model = simplified_model
            else:
                logger.info("Using full ISNEModel")
                # Create a specific type instance that is a subclass of nn.Module
                full_model: nn.Module = ISNEModel(
                    in_features=embedding_dim,
                    hidden_features=hidden_dim,
                    out_features=output_dim,
                    num_layers=num_layers
                )
                self.model = full_model
                
            # Ensure model is not None at this point
            if self.model is not None:
                # Load model weights - since self.model is nn.Module, it has load_state_dict
                self.model.load_state_dict(model_data["model_state_dict"])
                # Move model to device - self.model.to() returns an nn.Module
                self.model = self.model.to(self.device)
                # Set model to evaluation mode
                _ = self.model.eval()  # This returns self, but we don't need to reassign
            else:
                raise ValueError("Model initialization failed")
            
            logger.info("ISNE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ISNE model: {str(e)}")
            raise
    
    def process_documents(
            self, 
            documents: List[Dict[str, Any]],
            save_report: bool = True,
            output_dir: Optional[str] = None
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process documents through the ISNE pipeline with validation.
        
        Args:
            documents: List of processed documents with base embeddings
            save_report: Whether to save the validation report to disk
            output_dir: Directory to save validation report and enhanced documents
            
        Returns:
            Tuple of (enhanced_documents, processing_stats)
        """
        # Ensure we have documents
        if not documents:
            logger.warning("No documents provided for processing")
            return [], {"status": "error", "message": "No documents provided"}
        
        # Ensure model is loaded
        if self.model is None and self.model_path:
            logger.info("Model not loaded, attempting to load from path")
            self.load_model(self.model_path)
        
        # Check if we still don't have a model
        if self.model is None:
            error_msg = "No ISNE model available for processing"
            logger.error(error_msg)
            return [], {"status": "error", "message": error_msg}
        
        # Extract embeddings from documents
        embeddings = []
        doc_ids: List[str] = []
        
        for doc in documents:
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
                doc_ids.append(doc.get("id", f"doc_{len(doc_ids)}"))
            else:
                logger.warning(f"Document {doc.get('id', 'unknown')} has no embedding, skipping")
        
        if not embeddings:
            error_msg = "No valid embeddings found in documents"
            logger.error(error_msg)
            return [], {"status": "error", "message": error_msg}
        
        # Convert to tensor
        x = torch.tensor(embeddings, dtype=torch.float32)
        
        # Validate input embeddings if requested
        validation_summary: Dict[str, Any] = {}
        if self.validate:
            logger.info("Validating input embeddings")
            # Use the correct function signature
            validation_summary = validate_embeddings_before_isne(documents)
        
        try:
            # Generate enhanced embeddings with ISNE
            logger.info(f"Generating enhanced embeddings for {len(embeddings)} documents")
            
            # Move data to the appropriate device
            x = x.to(self.device)
            
            # Create graph if needed (some models require it, others don't)
            edge_index = None
            
            # Get enhanced embeddings - model is guaranteed to be not None at this point
            assert isinstance(self.model, nn.Module), "Model must be an instance of nn.Module"
            
            # Call the model to get enhanced embeddings
            enhanced_embeddings = self.model(x, edge_index)
            
            # Move back to CPU for further processing
            enhanced_embeddings_np = enhanced_embeddings.cpu().numpy()
            
            logger.info(f"Successfully generated {len(enhanced_embeddings_np)} enhanced embeddings")
            
            # Validate enhanced embeddings if requested
            if self.validate:
                logger.info("Validating enhanced embeddings")
                # Use the correct function signature
                post_validation = validate_embeddings_after_isne(documents, validation_summary)
                # Update validation summary
                validation_summary = create_validation_summary(validation_summary, post_validation)
                
                # Check for discrepancies and trigger alerts if needed
                self._check_validation_discrepancies(validation_summary)
            
            # Update documents with enhanced embeddings
            enhanced_documents = []
            for i, doc in enumerate(documents):
                if i < len(enhanced_embeddings_np):
                    # Create a copy of the document to avoid modifying the original
                    enhanced_doc = doc.copy()
                    
                    # Store both embeddings
                    enhanced_doc["original_embedding"] = doc.get("embedding")
                    enhanced_doc["enhanced_embedding"] = enhanced_embeddings_np[i].tolist()
                    
                    # Set the main embedding to the enhanced one
                    enhanced_doc["embedding"] = enhanced_embeddings_np[i].tolist()
                    
                    # Add validation info to the document if available
                    if self.validate:
                        enhanced_doc["validation_info"] = {
                            "timestamp": datetime.now().isoformat(),
                            "quality_score": validation_summary.get("quality_score", 0)
                        }
                    
                    enhanced_documents.append(enhanced_doc)
                else:
                    logger.warning(f"Document at index {i} has no corresponding enhanced embedding")
                    enhanced_documents.append(doc)
            
            # Properly attach validation summary to the enhanced documents list
            if self.validate:
                enhanced_documents = attach_validation_summary(enhanced_documents, validation_summary)
            
            # Save validation report if requested
            if save_report and self.validate and output_dir:
                self._save_validation_report(validation_summary, output_dir)
            
            # Prepare processing stats
            processing_stats = {
                "status": "success",
                "documents_processed": len(enhanced_documents),
                "timestamp": datetime.now().isoformat(),
                "validation_performed": self.validate
            }
            
            # Include validation summary in stats if available
            if self.validate:
                processing_stats["validation_summary"] = {
                    k: v for k, v in validation_summary.items()
                    if k in ["overall_quality", "anomalies_detected", "consistency_score"]
                }
            
            return enhanced_documents, processing_stats
            
        except Exception as e:
            error_msg = f"Error during ISNE processing: {str(e)}"
            logger.error(error_msg)
            return [], {"status": "error", "message": error_msg}
    
    def _check_validation_discrepancies(self, validation_summary: Dict[str, Any]) -> None:
        """
        Check validation summary for discrepancies and trigger alerts if needed.
        
        Args:
            validation_summary: Validation summary dictionary
        """
        # Check for anomalies and trigger alerts if needed
        anomalies = validation_summary.get("anomalies", [])
        consistency_issues = validation_summary.get("consistency_issues", [])
        
        if len(anomalies) > self.alert_threshold_value:
            self._trigger_alert(
                f"ISNE validation detected {len(anomalies)} anomalies in embeddings",
                "high"
            )
        elif len(anomalies) > 0:
            self._trigger_alert(
                f"ISNE validation detected {len(anomalies)} anomalies in embeddings",
                "medium"
            )
        
        if len(consistency_issues) > self.alert_threshold_value:
            self._trigger_alert(
                f"ISNE validation detected {len(consistency_issues)} consistency issues",
                "high"
            )
        elif len(consistency_issues) > 0:
            self._trigger_alert(
                f"ISNE validation detected {len(consistency_issues)} consistency issues",
                "medium"
            )
    
    def _trigger_alert(self, message: str, level: str = "medium") -> None:
        """
        Trigger an alert for validation issues.
        
        Args:
            message: Alert message
            level: Alert level ("low", "medium", "high")
        """
        alert_level = AlertLevel.MEDIUM  # Default
        
        if level == "low":
            alert_level = AlertLevel.LOW
        elif level == "high":
            alert_level = AlertLevel.HIGH
        
        # Log the alert message
        logger.warning(f"ISNE Alert ({level}): {message}")
        
        # Send alert through alert manager with proper parameters
        context = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path
        }
        self.alert_manager.alert(message, alert_level, "ISNEPipeline", context)
    
    def configure_alert_email(self, config: Dict[str, Any]) -> None:
        """
        Configure email alerts for ISNE validation issues.
        
        Args:
            config: Email configuration dictionary with keys:
                   smtp_server, smtp_port, username, password,
                   from_addr, to_addrs
        """
        # Validate configuration
        required_keys = ["smtp_server", "smtp_port", "username", "password", "from_addr", "to_addrs"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required email configuration keys: {', '.join(missing_keys)}")
        
        # Configure email alerts in alert manager by passing the config dictionary
        self.alert_manager.configure_email(config)
        
        logger.info("Email alerts configured for ISNE pipeline")
    
    def _save_validation_report(self, validation_summary: Dict[str, Any], output_dir: str) -> None:
        """
        Save validation report to disk.
        
        Args:
            validation_summary: Validation summary dictionary
            output_dir: Directory to save report
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"isne_validation_{timestamp}.json")
        
        # Save report to file
        with open(report_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def save_enhanced_documents(self, documents: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """
        Save enhanced documents to disk.
        
        Args:
            documents: Enhanced documents
            output_dir: Directory to save documents
            
        Returns:
            Dictionary with output file paths
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save enhanced documents
        docs_path = os.path.join(output_dir, f"isne_enhanced_docs_{timestamp}.json")
        with open(docs_path, 'w') as f:
            json.dump(documents, f, indent=2)
        
        logger.info(f"Enhanced documents saved to {docs_path}")
        
        # Save embeddings separately for easier loading
        embeddings = [doc.get("embedding", []) for doc in documents]
        embeddings_path = os.path.join(output_dir, f"isne_embeddings_{timestamp}.npy")
        np.save(embeddings_path, np.array(embeddings))
        
        logger.info(f"Enhanced embeddings saved to {embeddings_path}")
        
        return {
            "documents": docs_path,
            "embeddings": embeddings_path
        }
