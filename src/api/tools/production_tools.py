"""
MCP Tools for HADES production pipeline operations.

These tools provide MCP interface to the production pipeline endpoints,
allowing external systems to trigger and monitor pipeline operations.
"""

from typing import Dict, Any, List, Optional
import requests
import json
import time
from pathlib import Path

class ProductionPipelineTools:
    """MCP tools for production pipeline operations."""
    
    def __init__(self, base_url: str = "http://localhost:8595"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1/production"
    
    def bootstrap_isne_dataset(
        self,
        input_dir: str,
        database_name: str = "isne_training_database",
        similarity_threshold: float = 0.8,
        debug_logging: bool = False
    ) -> Dict[str, Any]:
        """
        Bootstrap ISNE test dataset into ArangoDB.
        
        Args:
            input_dir: Path to ISNE test dataset directory
            database_name: Target database name
            similarity_threshold: Similarity threshold for edge creation
            debug_logging: Enable debug logging
            
        Returns:
            Operation details including operation ID for status tracking
        """
        payload = {
            "input_dir": input_dir,
            "database_name": database_name,
            "similarity_threshold": similarity_threshold,
            "debug_logging": debug_logging
        }
        
        response = requests.post(f"{self.api_base}/bootstrap", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def train_isne_model(
        self,
        database_name: str = "isne_training_database",
        output_dir: str = "./output/isne_training_efficient",
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        hidden_dim: int = 256,
        num_layers: int = 4
    ) -> Dict[str, Any]:
        """
        Train ISNE model on graph data.
        
        Args:
            database_name: Source database name
            output_dir: Output directory for model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            hidden_dim: Hidden dimension size
            num_layers: Number of model layers
            
        Returns:
            Operation details including operation ID for status tracking
        """
        payload = {
            "database_name": database_name,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers
        }
        
        response = requests.post(f"{self.api_base}/train", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def apply_isne_model(
        self,
        model_path: str,
        source_db: str = "isne_training_database",
        target_db: str = "isne_production_database",
        similarity_threshold: float = 0.85,
        max_edges_per_node: int = 10,
        create_new_db: bool = True
    ) -> Dict[str, Any]:
        """
        Apply trained ISNE model to create production database.
        
        Args:
            model_path: Path to trained ISNE model
            source_db: Source database
            target_db: Target database
            similarity_threshold: Similarity threshold for new edges
            max_edges_per_node: Maximum edges per node
            create_new_db: Create new production database
            
        Returns:
            Operation details including operation ID for status tracking
        """
        payload = {
            "model_path": model_path,
            "source_db": source_db,
            "target_db": target_db,
            "similarity_threshold": similarity_threshold,
            "max_edges_per_node": max_edges_per_node,
            "create_new_db": create_new_db
        }
        
        response = requests.post(f"{self.api_base}/apply-model", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def build_semantic_collections(
        self,
        database_name: str = "isne_production_database"
    ) -> Dict[str, Any]:
        """
        Build semantic collections for production use.
        
        Args:
            database_name: Database name
            
        Returns:
            Operation details including operation ID for status tracking
        """
        payload = {
            "database_name": database_name
        }
        
        response = requests.post(f"{self.api_base}/build-collections", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def run_complete_pipeline(
        self,
        input_dir: str,
        database_prefix: str = "hades_production"
    ) -> Dict[str, Any]:
        """
        Run the complete ISNE production pipeline end-to-end.
        
        This runs all pipeline steps in sequence:
        1. Bootstrap data
        2. Train ISNE model
        3. Apply model
        4. Build semantic collections
        
        Args:
            input_dir: Path to ISNE test dataset directory
            database_prefix: Prefix for database names
            
        Returns:
            Operation details including operation ID for status tracking
        """
        params = {
            "input_dir": input_dir,
            "database_prefix": database_prefix
        }
        
        response = requests.post(f"{self.api_base}/run-complete-pipeline", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_operation_status(self, operation_id: str) -> Dict[str, Any]:
        """
        Get status of a pipeline operation.
        
        Args:
            operation_id: Operation ID to check
            
        Returns:
            Current status of the operation
        """
        response = requests.get(f"{self.api_base}/status/{operation_id}")
        response.raise_for_status()
        
        return response.json()
    
    def list_operations(self) -> List[Dict[str, Any]]:
        """
        List all pipeline operations and their status.
        
        Returns:
            List of all operations with their current status
        """
        response = requests.get(f"{self.api_base}/status")
        response.raise_for_status()
        
        return response.json()
    
    def wait_for_completion(
        self,
        operation_id: str,
        timeout_seconds: int = 3600,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Wait for a pipeline operation to complete.
        
        Args:
            operation_id: Operation ID to wait for
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks
            
        Returns:
            Final status of the operation
            
        Raises:
            TimeoutError: If operation doesn't complete within timeout
            RuntimeError: If operation fails
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_operation_status(operation_id)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise RuntimeError(f"Operation failed: {status.get('progress', 'Unknown error')}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Operation {operation_id} did not complete within {timeout_seconds} seconds")

# Global instance for MCP tool registration
production_tools = ProductionPipelineTools()

def get_production_tools() -> ProductionPipelineTools:
    """Get the global production tools instance."""
    return production_tools

# MCP Tool function implementations
def mcp_bootstrap_isne_dataset(input_dir: str, **kwargs) -> str:
    """MCP tool: Bootstrap ISNE test dataset."""
    result = production_tools.bootstrap_isne_dataset(input_dir, **kwargs)
    return json.dumps(result, indent=2)

def mcp_train_isne_model(**kwargs) -> str:
    """MCP tool: Train ISNE model."""
    result = production_tools.train_isne_model(**kwargs)
    return json.dumps(result, indent=2)

def mcp_apply_isne_model(model_path: str, **kwargs) -> str:
    """MCP tool: Apply trained ISNE model."""
    result = production_tools.apply_isne_model(model_path, **kwargs)
    return json.dumps(result, indent=2)

def mcp_build_semantic_collections(**kwargs) -> str:
    """MCP tool: Build semantic collections."""
    result = production_tools.build_semantic_collections(**kwargs)
    return json.dumps(result, indent=2)

def mcp_run_complete_pipeline(input_dir: str, **kwargs) -> str:
    """MCP tool: Run complete production pipeline."""
    result = production_tools.run_complete_pipeline(input_dir, **kwargs)
    return json.dumps(result, indent=2)

def mcp_get_operation_status(operation_id: str) -> str:
    """MCP tool: Get operation status."""
    result = production_tools.get_operation_status(operation_id)
    return json.dumps(result, indent=2)

def mcp_list_operations() -> str:
    """MCP tool: List all operations."""
    result = production_tools.list_operations()
    return json.dumps(result, indent=2)

def mcp_wait_for_completion(operation_id: str, timeout_seconds: int = 3600) -> str:
    """MCP tool: Wait for operation completion."""
    try:
        result = production_tools.wait_for_completion(operation_id, timeout_seconds)
        return json.dumps(result, indent=2)
    except (TimeoutError, RuntimeError) as e:
        return json.dumps({"error": str(e)}, indent=2)