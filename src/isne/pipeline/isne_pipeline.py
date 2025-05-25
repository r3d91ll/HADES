"""
ISNE Pipeline with integrated validation.

This module provides a production-ready ISNE pipeline implementation with
built-in validation to ensure embedding consistency and quality.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, cast

import torch
import numpy as np

from src.isne.models.isne_model import ISNEModel
from src.isne.models.simplified_isne_model import SimplifiedISNEModel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)
from src.isne.utils.geometric_utils import create_graph_from_documents
from src.alerts import AlertManager, AlertLevel

# Configure logging
logger = logging.getLogger(__name__)

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
        self.model: Optional[Union[ISNEModel, SimplifiedISNEModel]] = None
        
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
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    
    def load_model(self, model_path: str) -> None:
        """
        Load ISNE model from a file.
        
        Args:
            model_path: Path to the trained model
        """
        logger.info(f"Loading ISNE model from {model_path}")
        
        try:
            # Check if model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"ISNE model file not found: {model_path}")
            
            # Load model data
            model_data = torch.load(model_path, map_location=self.device)
            logger.info(f"Successfully loaded model with keys: {list(model_data.keys())}")
            
            # Get model configuration
            model_config = model_data.get("config", {})
            if not model_config:
                logger.warning("Model config not found in saved model, using defaults")
            
            # Extract model parameters
            embedding_dim = model_config.get("embedding_dim", 768)  # Default ModernBERT dimension
            hidden_dim = model_config.get("hidden_dim", 256)
            output_dim = model_config.get("output_dim", 768)
            num_layers = model_config.get("num_layers", 2)
            
            logger.info(f"Model dimensions: {embedding_dim} → {hidden_dim} → {output_dim}")
            
            # Determine model type (full or simplified)
            model_type = model_config.get("model_type", "full")
            
            # Create the appropriate model
            if model_type == "simplified":
                logger.info("Using SimplifiedISNEModel")
                model = SimplifiedISNEModel(
                    in_features=embedding_dim,
                    hidden_features=hidden_dim,  # Required parameter even though not used internally
                    out_features=output_dim
                )
                self.model = model
            else:
                logger.info("Using full ISNEModel")
                model = ISNEModel(
                    in_features=embedding_dim,
                    hidden_features=hidden_dim,
                    out_features=output_dim,
                    num_layers=num_layers
                )
                self.model = model
            
            # Load model weights
            if self.model is not None:
                # Use explicit type casting to help mypy understand the object is a torch.nn.Module
                model = cast(torch.nn.Module, self.model)
                model.load_state_dict(model_data["model_state_dict"])
                model.to(self.device)
                model.eval()  # Set to evaluation mode
                # Update self.model to ensure it keeps the reference
                self.model = model
            else:
                raise ValueError("Model initialization failed")
            
            logger.info("ISNE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ISNE model: {e}")
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
        start_time = time.time()
        
        # Ensure model is loaded
        if self.model is None and self.model_path:
            self.load_model(self.model_path)
        elif self.model is None:
            raise ValueError("No ISNE model loaded. Either provide a model_path during initialization or call load_model().")
        
        # Pre-validation
        pre_validation = {}
        if self.validate:
            logger.info("Running pre-ISNE validation...")
            pre_validation = validate_embeddings_before_isne(documents)
            logger.info(f"Pre-validation complete: {len(documents)} documents with {pre_validation.get('total_chunks', 0)} chunks")
        
        # Create graph from documents
        logger.info("Creating document graph for ISNE inference...")
        graph, node_metadata, node_idx_map = create_graph_from_documents(documents)
        
        if graph is None or graph.num_nodes == 0:
            logger.error("Failed to build valid graph from documents")
            return documents, {"error": "Failed to build valid graph"}
        
        # Apply the ISNE model
        logger.info(f"Applying ISNE model to enhance {graph.num_nodes} embeddings...")
        enhanced_embeddings = None
        
        try:
            with torch.no_grad():
                # Move graph to appropriate device
                graph = graph.to(self.device)
                
                # Make sure all tensors are properly shaped and on the correct device
                x = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                
                # Handle edge attributes
                edge_attr = None
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    edge_attr = graph.edge_attr
                    # Convert to float tensor if needed
                    if not isinstance(edge_attr, torch.FloatTensor) and not isinstance(edge_attr, torch.cuda.FloatTensor):
                        edge_attr = edge_attr.float()
                    
                    # Reshape if needed
                    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                        edge_attr = edge_attr.squeeze(1)
                    
                    edge_attr = edge_attr.to(self.device)
                
                # Get enhanced embeddings
                if self.model is None:
                    raise ValueError("No ISNE model loaded. Call load_model() first.")
                    
                enhanced_embeddings = self.model(x, edge_index)
                
                # Move back to CPU for further processing
                enhanced_embeddings = enhanced_embeddings.cpu().numpy()
                
                logger.info(f"Successfully generated {len(enhanced_embeddings)} enhanced embeddings")
                
        except Exception as e:
            logger.error(f"Error during ISNE inference: {e}")
            if self.validate:
                self._trigger_alert(f"ISNE inference failed: {e}", "high")
            return documents, {"error": f"ISNE inference failed: {e}"}
        
        # Update documents with enhanced embeddings
        logger.info("Updating documents with enhanced embeddings...")
        
        # Create deep copy to avoid modifying originals
        import copy
        enhanced_documents = copy.deepcopy(documents)
        
        # Update embeddings in the document structure
        for doc_idx, doc in enumerate(enhanced_documents):
            if "chunks" not in doc or not doc["chunks"]:
                continue
                
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                
                if chunk_id in node_idx_map:
                    node_idx = node_idx_map[chunk_id]
                    # Add the enhanced embedding to the chunk
                    chunk["isne_embedding"] = enhanced_embeddings[node_idx].tolist()
        
        logger.info("Document enhancement with ISNE embeddings completed")
        
        # Post-validation
        validation_summary = None
        if self.validate and pre_validation:
            logger.info("Running post-ISNE validation...")
            post_validation = validate_embeddings_after_isne(enhanced_documents, pre_validation)
            logger.info(f"Post-validation complete: found {post_validation.get('chunks_with_isne', 0)} chunks with ISNE embeddings")
            
            # Create validation summary
            validation_summary = create_validation_summary(pre_validation, post_validation)
            
            # Check for discrepancies and trigger alerts if needed
            self._check_validation_discrepancies(validation_summary)
            
            # Attach validation summary to documents
            enhanced_documents = attach_validation_summary(enhanced_documents, validation_summary)
            
            # Save validation report if requested
            if save_report and output_dir:
                self._save_validation_report(validation_summary, output_dir)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compile statistics
        stats = {
            "total_documents": len(enhanced_documents),
            "total_chunks": pre_validation.get("total_chunks", 0) if pre_validation else sum(len(doc.get("chunks", [])) for doc in documents),
            "processing_time": processing_time,
            "validation_summary": validation_summary
        }
        
        return enhanced_documents, stats
    
    def _check_validation_discrepancies(self, validation_summary: Dict[str, Any]) -> None:
        """
        Check validation summary for discrepancies and trigger alerts if needed.
        
        Args:
            validation_summary: Validation summary dictionary
        """
        discrepancies = validation_summary.get("discrepancies", {})
        
        # Calculate total discrepancies
        total_discrepancies = sum(abs(value) for value in discrepancies.values())
        
        # Trigger alerts based on discrepancy count
        if total_discrepancies > 0:
            # Determine alert level
            alert_level = AlertLevel.LOW
            title_prefix = "NOTICE"
            
            if total_discrepancies >= self.alert_threshold_value:
                alert_level = AlertLevel.HIGH
                title_prefix = "CRITICAL"
            elif total_discrepancies >= self.alert_threshold_value / 2:
                alert_level = AlertLevel.MEDIUM
                title_prefix = "WARNING"
            
            # Create detailed alert message
            message = f"{title_prefix}: Found {total_discrepancies} total embedding discrepancies"
            
            details = []
            for key, value in discrepancies.items():
                if value != 0:
                    details.append(f"{key}: {value}")
            
            if details:
                message += f" - {', '.join(details)}"
            
            # Create detailed context for the alert
            context = {
                "discrepancies": discrepancies,
                "total_discrepancies": total_discrepancies,
                "threshold": self.alert_threshold_value,
                "expected_counts": validation_summary.get("expected_counts", {}),
                "actual_counts": validation_summary.get("actual_counts", {})
            }
            
            # Trigger the alert using the alert manager
            self.alert_manager.alert(
                message=message,
                level=alert_level,
                source="isne_pipeline",
                context=context
            )
    
    def _trigger_alert(self, message: str, level: str = "medium") -> None:
        """
        Trigger an alert for validation issues.
        
        Args:
            message: Alert message
            level: Alert level ("low", "medium", "high")
        """
        # Convert string level to AlertLevel enum
        alert_level = AlertLevel.MEDIUM  # Default
        if level == "low":
            alert_level = AlertLevel.LOW
        elif level == "medium":
            alert_level = AlertLevel.MEDIUM
        elif level == "high":
            alert_level = AlertLevel.HIGH
        
        # Create the alert
        self.alert_manager.alert(
            message=message,
            level=alert_level,
            source="isne_pipeline"
        )
    
    def configure_alert_email(self, config: Dict[str, Any]) -> None:
        """
        Configure email alerts for ISNE validation issues.
        
        Args:
            config: Email configuration dictionary with keys:
                   smtp_server, smtp_port, username, password,
                   from_addr, to_addrs
        """
        self.alert_manager.configure_email(config)
    
    def _save_validation_report(self, validation_summary: Dict[str, Any], output_dir: str) -> None:
        """
        Save validation report to disk.
        
        Args:
            validation_summary: Validation summary dictionary
            output_dir: Directory to save report
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_path = output_path / "isne_validation_report.json"
            
            with open(report_path, "w") as f:
                json.dump(validation_summary, f, indent=2)
                
            logger.info(f"Saved validation report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    def save_enhanced_documents(self, documents: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
        """
        Save enhanced documents to disk.
        
        Args:
            documents: Enhanced documents
            output_dir: Directory to save documents
            
        Returns:
            Dictionary with output file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save all documents
            full_path = output_path / "isne_enhanced_documents.json"
            with open(full_path, "w") as f:
                json.dump(documents, f, indent=2)
            
            # Save a small sample for quick review
            sample_size = min(2, len(documents))
            sample_path = output_path / "isne_enhanced_sample.json"
            with open(sample_path, "w") as f:
                json.dump(documents[:sample_size], f, indent=2)
                
            logger.info(f"Saved all enhanced documents to {full_path}")
            logger.info(f"Saved sample of {sample_size} documents to {sample_path}")
            
            return {
                "full_path": str(full_path),
                "sample_path": str(sample_path)
            }
        except Exception as e:
            logger.error(f"Error saving enhanced documents: {e}")
            return {"error": str(e)}
