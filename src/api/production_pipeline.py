"""
Production Pipeline API endpoints for HADES.

This module provides endpoints for running the complete ISNE production pipeline:
- Bootstrap data ingestion
- ISNE model training
- Model application with edge discovery
- Semantic collection building
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Import our production pipeline modules
import sys
# Add the production pipelines directory to path
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "production"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prod", tags=["production"])

# Request/Response Models
class BootstrapRequest(BaseModel):
    """Request for bootstrapping data from ISNE test dataset."""
    input_dir: str = Field(..., description="Path to ISNE test dataset directory")
    database_name: str = Field(default=None, description="Target database name (uses config default if not provided)")
    similarity_threshold: float = Field(default=None, description="Similarity threshold for edge creation (uses config default if not provided)")
    debug_logging: bool = Field(default=False, description="Enable debug logging")
    config_overrides: Dict[str, Any] = Field(default={}, description="Configuration overrides")

class TrainingRequest(BaseModel):
    """Request for training ISNE model."""
    database_name: str = Field(default="isne_training_database", description="Source database name")
    output_dir: str = Field(default="./output/isne_training_efficient", description="Output directory for model")
    epochs: int = Field(default=50, description="Number of training epochs")
    batch_size: int = Field(default=512, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    hidden_dim: int = Field(default=256, description="Hidden dimension size")
    num_layers: int = Field(default=4, description="Number of model layers")

class ModelApplicationRequest(BaseModel):
    """Request for applying trained ISNE model."""
    model_path: str = Field(..., description="Path to trained ISNE model")
    source_db: str = Field(default="isne_training_database", description="Source database")
    target_db: str = Field(default="isne_production_database", description="Target database")
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for new edges")
    max_edges_per_node: int = Field(default=10, description="Maximum edges per node")
    create_new_db: bool = Field(default=True, description="Create new production database")

class SemanticCollectionRequest(BaseModel):
    """Request for building semantic collections."""
    database_name: str = Field(default="isne_production_database", description="Database name")

class PipelineResponse(BaseModel):
    """Response for pipeline operations."""
    success: bool
    message: str
    execution_time_seconds: float
    details: Dict[str, Any] = Field(default_factory=dict)

class PipelineStatus(BaseModel):
    """Status of running pipeline operations."""
    operation: str
    status: str  # "running", "completed", "failed"
    progress: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    details: Dict[str, Any] = Field(default_factory=dict)

# Global status tracking
_pipeline_status: Dict[str, PipelineStatus] = {}

def update_pipeline_status(operation_id: str, status: str, progress: Optional[str] = None, 
                          details: Optional[Dict[str, Any]] = None) -> None:
    """Update pipeline operation status."""
    if operation_id in _pipeline_status:
        _pipeline_status[operation_id].status = status
        if progress:
            _pipeline_status[operation_id].progress = progress
        if details:
            _pipeline_status[operation_id].details.update(details)
        if status in ["completed", "failed"]:
            _pipeline_status[operation_id].end_time = datetime.now(timezone.utc)

async def run_bootstrap_pipeline(request: BootstrapRequest, operation_id: str):
    """Run bootstrap pipeline in background."""
    try:
        update_pipeline_status(operation_id, "running", "Initializing bootstrap pipeline")
        
        # Import and run bootstrap script
        from bootstrap_full_isne_testdata import FullISNETestDataBootstrap
        
        update_pipeline_status(operation_id, "running", "Processing ISNE test dataset")
        
        bootstrap = FullISNETestDataBootstrap(
            input_dir=request.input_dir,
            database_name=request.database_name
        )
        
        # Run the bootstrap process
        result = bootstrap.run_full_bootstrap(
            similarity_threshold=request.similarity_threshold,
            debug_logging=request.debug_logging
        )
        
        update_pipeline_status(operation_id, "completed", "Bootstrap completed successfully", result)
        
    except Exception as e:
        logger.error(f"Bootstrap pipeline failed: {e}")
        update_pipeline_status(operation_id, "failed", f"Error: {str(e)}")

async def run_training_pipeline(request: TrainingRequest, operation_id: str):
    """Run ISNE training pipeline in background."""
    try:
        update_pipeline_status(operation_id, "running", "Initializing ISNE training")
        
        # Import and run training script
        from train_isne_memory_efficient import MemoryEfficientISNETrainer
        
        trainer = MemoryEfficientISNETrainer(request.database_name)
        
        update_pipeline_status(operation_id, "running", "Connecting to database")
        db = trainer.connect_to_database()
        
        update_pipeline_status(operation_id, "running", "Extracting graph data")
        graph_data = trainer.extract_graph_data(db)
        
        update_pipeline_status(operation_id, "running", "Training ISNE model")
        results = trainer.run_training(
            graph_data=graph_data,
            output_dir=Path(request.output_dir),
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            hidden_dim=request.hidden_dim,
            num_layers=request.num_layers
        )
        
        update_pipeline_status(operation_id, "completed", "Training completed successfully", results)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        update_pipeline_status(operation_id, "failed", f"Error: {str(e)}")

async def run_model_application_pipeline(request: ModelApplicationRequest, operation_id: str):
    """Run model application pipeline in background."""
    try:
        update_pipeline_status(operation_id, "running", "Initializing model application")
        
        # Import and run application script
        from apply_efficient_isne_model import EfficientISNEApplicationPipeline
        
        pipeline = EfficientISNEApplicationPipeline(
            source_db=request.source_db,
            target_db=request.target_db
        )
        
        update_pipeline_status(operation_id, "running", "Applying ISNE model")
        results = pipeline.run_pipeline(
            model_path=request.model_path,
            similarity_threshold=request.similarity_threshold,
            max_edges_per_node=request.max_edges_per_node,
            create_new_db=request.create_new_db
        )
        
        update_pipeline_status(operation_id, "completed", "Model application completed", results)
        
    except Exception as e:
        logger.error(f"Model application pipeline failed: {e}")
        update_pipeline_status(operation_id, "failed", f"Error: {str(e)}")

async def run_collection_pipeline(request: SemanticCollectionRequest, operation_id: str):
    """Run semantic collection building in background."""
    try:
        update_pipeline_status(operation_id, "running", "Building semantic collections")
        
        # Import and run semantic collection script
        from build_semantic_collections import SemanticCollectionBuilder
        
        builder = SemanticCollectionBuilder(request.database_name)
        db = builder.connect_to_database()
        
        update_pipeline_status(operation_id, "running", "Extracting and organizing entities")
        stats = builder.build_semantic_collections(db)
        
        update_pipeline_status(operation_id, "completed", "Semantic collections built", stats)
        
    except Exception as e:
        logger.error(f"Semantic collection pipeline failed: {e}")
        update_pipeline_status(operation_id, "failed", f"Error: {str(e)}")

# API Endpoints

@router.post("/bootstrap", response_model=PipelineResponse)
async def bootstrap_data(request: BootstrapRequest, background_tasks: BackgroundTasks):
    """
    Bootstrap ISNE test dataset into ArangoDB.
    
    This endpoint processes the ISNE test dataset and creates a graph structure
    ready for ISNE training.
    """
    operation_id = f"bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize status
    _pipeline_status[operation_id] = PipelineStatus(
        operation="bootstrap",
        status="starting",
        start_time=datetime.now(timezone.utc)
    )
    
    # Start background task
    background_tasks.add_task(run_bootstrap_pipeline, request, operation_id)
    
    return PipelineResponse(
        success=True,
        message=f"Bootstrap pipeline started with ID: {operation_id}",
        execution_time_seconds=0.0,
        details={"operation_id": operation_id, "status_endpoint": f"/prod/status/{operation_id}"}
    )

@router.post("/train", response_model=PipelineResponse)
async def train_isne_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ISNE model on graph data.
    
    This endpoint trains a memory-efficient ISNE model on the bootstrapped graph data.
    """
    operation_id = f"training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize status
    _pipeline_status[operation_id] = PipelineStatus(
        operation="training",
        status="starting",
        start_time=datetime.now(timezone.utc)
    )
    
    # Start background task
    background_tasks.add_task(run_training_pipeline, request, operation_id)
    
    return PipelineResponse(
        success=True,
        message=f"Training pipeline started with ID: {operation_id}",
        execution_time_seconds=0.0,
        details={"operation_id": operation_id, "status_endpoint": f"/prod/status/{operation_id}"}
    )

@router.post("/apply-model", response_model=PipelineResponse)
async def apply_isne_model(request: ModelApplicationRequest, background_tasks: BackgroundTasks):
    """
    Apply trained ISNE model to create production database.
    
    This endpoint applies the trained model to enhance embeddings and discover new edges.
    """
    operation_id = f"application_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize status
    _pipeline_status[operation_id] = PipelineStatus(
        operation="model_application",
        status="starting", 
        start_time=datetime.now(timezone.utc)
    )
    
    # Start background task
    background_tasks.add_task(run_model_application_pipeline, request, operation_id)
    
    return PipelineResponse(
        success=True,
        message=f"Model application pipeline started with ID: {operation_id}",
        execution_time_seconds=0.0,
        details={"operation_id": operation_id, "status_endpoint": f"/prod/status/{operation_id}"}
    )

@router.post("/build-collections", response_model=PipelineResponse)
async def build_collections(request: SemanticCollectionRequest, background_tasks: BackgroundTasks):
    """
    Build semantic collections for production use.
    
    This endpoint creates organized collections for code, documentation, and semantic analysis.
    """
    operation_id = f"collections_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize status
    _pipeline_status[operation_id] = PipelineStatus(
        operation="semantic_collections",
        status="starting",
        start_time=datetime.now(timezone.utc)
    )
    
    # Start background task
    background_tasks.add_task(run_collection_pipeline, request, operation_id)
    
    return PipelineResponse(
        success=True,
        message=f"Semantic collection pipeline started with ID: {operation_id}",
        execution_time_seconds=0.0,
        details={"operation_id": operation_id, "status_endpoint": f"/prod/status/{operation_id}"}
    )

@router.get("/status/{operation_id}", response_model=PipelineStatus)
async def get_pipeline_status(operation_id: str):
    """Get status of a running pipeline operation."""
    if operation_id not in _pipeline_status:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    
    return _pipeline_status[operation_id]

@router.get("/status", response_model=List[PipelineStatus])
async def list_operations():
    """List all pipeline operations and their status."""
    return list(_pipeline_status.values())

@router.post("/run-pipeline", response_model=PipelineResponse)
async def run_pipeline(
    input_dir: str,
    database_prefix: str = "hades_production",
    background_tasks: BackgroundTasks = None
):
    """
    Run the complete ISNE production pipeline end-to-end.
    
    This convenience endpoint runs all pipeline steps in sequence:
    1. Bootstrap data
    2. Train ISNE model  
    3. Apply model
    4. Build semantic collections
    """
    operation_id = f"complete_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize status
    _pipeline_status[operation_id] = PipelineStatus(
        operation="complete_pipeline",
        status="starting",
        start_time=datetime.now(timezone.utc)
    )
    
    # Define pipeline steps
    async def run_complete():
        try:
            training_db = f"{database_prefix}_training"
            production_db = f"{database_prefix}_production"
            
            # Step 1: Bootstrap
            update_pipeline_status(operation_id, "running", "Step 1/4: Bootstrapping data")
            bootstrap_req = BootstrapRequest(
                input_dir=input_dir,
                database_name=training_db
            )
            await run_bootstrap_pipeline(bootstrap_req, f"{operation_id}_bootstrap")
            
            # Step 2: Training
            update_pipeline_status(operation_id, "running", "Step 2/4: Training ISNE model")
            training_req = TrainingRequest(
                database_name=training_db,
                output_dir=f"./output/complete_pipeline_{operation_id}"
            )
            await run_training_pipeline(training_req, f"{operation_id}_training")
            
            # Step 3: Apply Model
            update_pipeline_status(operation_id, "running", "Step 3/4: Applying ISNE model")
            # Find the trained model
            output_dir = Path(f"./output/complete_pipeline_{operation_id}")
            model_files = list(output_dir.glob("isne_model_*.pth"))
            if not model_files:
                raise Exception("No trained model found")
            
            model_path = str(model_files[0])
            application_req = ModelApplicationRequest(
                model_path=model_path,
                source_db=training_db,
                target_db=production_db
            )
            await run_model_application_pipeline(application_req, f"{operation_id}_application")
            
            # Step 4: Build Collections
            update_pipeline_status(operation_id, "running", "Step 4/4: Building semantic collections")
            collections_req = SemanticCollectionRequest(database_name=production_db)
            await run_semantic_collection_pipeline(collections_req, f"{operation_id}_collections")
            
            update_pipeline_status(operation_id, "completed", "Complete pipeline finished successfully", {
                "training_database": training_db,
                "production_database": production_db,
                "model_path": model_path
            })
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {e}")
            update_pipeline_status(operation_id, "failed", f"Error: {str(e)}")
    
    # Start background task
    background_tasks.add_task(run_complete)
    
    return PipelineResponse(
        success=True,
        message=f"Complete pipeline started with ID: {operation_id}",
        execution_time_seconds=0.0,
        details={
            "operation_id": operation_id, 
            "status_endpoint": f"/prod/status/{operation_id}",
            "expected_duration": "15-30 minutes"
        }
    )