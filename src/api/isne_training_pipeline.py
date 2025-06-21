"""
ISNE Training Pipeline API

API endpoints for ISNE training operations with MCP support.
This provides safe, configurable training operations for regular model improvements.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.isne.training.pipeline import ISNETrainingPipeline, ISNETrainingConfig, TrainingResult

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/isne", tags=["ISNE Training"])

# Global training job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}


class TrainingRequest(BaseModel):
    """Request model for ISNE training."""
    
    # Optional config overrides
    data_dir: Optional[str] = Field(None, description="Data directory override")
    data_percentage: Optional[float] = Field(None, ge=0.01, le=1.0, description="Data percentage (0.01-1.0)")
    epochs: Optional[int] = Field(None, ge=1, le=200, description="Number of training epochs")
    learning_rate: Optional[float] = Field(None, gt=0, le=0.1, description="Learning rate")
    batch_size: Optional[int] = Field(None, ge=1, le=512, description="Batch size")
    model_name: Optional[str] = Field(None, description="Custom model name")
    enable_evaluation: Optional[bool] = Field(None, description="Enable post-training evaluation")
    
    # Job configuration
    job_id: Optional[str] = Field(None, description="Custom job ID (auto-generated if not provided)")


class TrainingResponse(BaseModel):
    """Response model for ISNE training."""
    
    job_id: str
    status: str  # "started", "running", "completed", "failed"
    message: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[int] = None
    progress: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""
    
    job_id: str
    status: str
    progress_percent: float = 0.0
    current_stage: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


@router.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start ISNE training pipeline.
    
    This endpoint starts a background training job with the specified configuration.
    The training will continue an existing model with new data.
    """
    try:
        # Generate job ID if not provided
        job_id = request.job_id or f"training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Check if job already exists
        if job_id in active_jobs:
            raise HTTPException(
                status_code=422, 
                detail=f"Job ID '{job_id}' already exists. Please use a different job_id or omit for auto-generation."
            )
        
        # Prepare config overrides
        overrides = {}
        if request.data_dir is not None:
            overrides["data.data_dir"] = request.data_dir
        if request.data_percentage is not None:
            overrides["data.data_percentage"] = request.data_percentage
        if request.epochs is not None:
            overrides["training.epochs"] = request.epochs
        if request.learning_rate is not None:
            overrides["training.learning_rate"] = request.learning_rate
        if request.batch_size is not None:
            overrides["training.batch_size"] = request.batch_size
        if request.model_name is not None:
            overrides["model.target_model.name"] = request.model_name
        if request.enable_evaluation is not None:
            overrides["evaluation.enabled"] = request.enable_evaluation
        
        # Initialize job tracking
        active_jobs[job_id] = {
            "status": "started",
            "progress_percent": 0.0,
            "current_stage": "initialization",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "overrides": overrides,
            "results": None,
            "error_message": None
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job, job_id, overrides)
        
        logger.info(f"Started ISNE training job: {job_id}")
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            message=f"Training job {job_id} started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a training job.
    
    Returns current status, progress, and results of the specified training job.
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = active_jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress_percent=job["progress_percent"],
        current_stage=job.get("current_stage"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error_message=job.get("error_message"),
        results=job.get("results")
    )


@router.get("/jobs", response_model=Dict[str, JobStatusResponse])
async def list_jobs():
    """
    List all training jobs.
    
    Returns status of all active and completed training jobs.
    """
    jobs = {}
    for job_id, job_data in active_jobs.items():
        jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status=job_data["status"],
            progress_percent=job_data["progress_percent"],
            current_stage=job_data.get("current_stage"),
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            error_message=job_data.get("error_message"),
            results=job_data.get("results")
        )
    
    return jobs


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a training job.
    
    Note: This currently only removes completed jobs from tracking.
    Active training cannot be safely cancelled once started.
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = active_jobs[job_id]
    
    if job["status"] in ["running"]:
        raise HTTPException(
            status_code=400, 
            detail="Cannot cancel running job - training operations cannot be safely interrupted"
        )
    
    del active_jobs[job_id]
    logger.info(f"Removed job: {job_id}")
    
    return {"message": f"Job {job_id} removed"}


async def run_training_job(job_id: str, overrides: Dict[str, Any]):
    """
    Run training job in background.
    
    This function executes the actual training pipeline and updates job status.
    """
    try:
        # Update job status
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["current_stage"] = "loading_config"
        active_jobs[job_id]["progress_percent"] = 5.0
        
        # Initialize training pipeline
        config = ISNETrainingConfig(overrides=overrides)
        pipeline = ISNETrainingPipeline(config)
        
        active_jobs[job_id]["current_stage"] = "data_sampling"
        active_jobs[job_id]["progress_percent"] = 10.0
        
        # Run training (this is synchronous)
        result = pipeline.train()
        
        if result.success:
            # Training completed successfully
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["progress_percent"] = 100.0
            active_jobs[job_id]["current_stage"] = "completed"
            active_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            active_jobs[job_id]["results"] = {
                "model_path": result.model_path,
                "model_name": result.model_name,
                "version": result.version,
                "training_stats": result.training_stats,
                "evaluation_results": result.evaluation_results
            }
            
            logger.info(f"Training job {job_id} completed successfully")
            
        else:
            # Training failed
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["current_stage"] = "failed"
            active_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            active_jobs[job_id]["error_message"] = result.error_message
            
            logger.error(f"Training job {job_id} failed: {result.error_message}")
            
    except Exception as e:
        # Unexpected error
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["current_stage"] = "failed"
        active_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        active_jobs[job_id]["error_message"] = str(e)
        
        logger.error(f"Training job {job_id} failed with exception: {e}")
        import traceback
        traceback.print_exc()


# MCP Tool Definitions
# These tools expose the API endpoints as MCP tools for external systems

def get_mcp_tools():
    """Get MCP tool definitions for ISNE training."""
    return [
        {
            "name": "isne_start_training",
            "description": "Start ISNE training pipeline with configurable parameters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_dir": {
                        "type": "string",
                        "description": "Data directory override (default: ../test-data3)"
                    },
                    "data_percentage": {
                        "type": "number",
                        "minimum": 0.01,
                        "maximum": 1.0,
                        "description": "Percentage of data to use (0.01-1.0, default: 0.10)"
                    },
                    "epochs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Number of training epochs (default: 20)"
                    },
                    "learning_rate": {
                        "type": "number",
                        "minimum": 0.0001,
                        "maximum": 0.1,
                        "description": "Learning rate (default: 0.001)"
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 512,
                        "description": "Batch size (default: 64)"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Custom model name (default: auto-generated with version)"
                    },
                    "enable_evaluation": {
                        "type": "boolean",
                        "description": "Enable post-training evaluation (default: true)"
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Custom job ID (default: auto-generated)"
                    }
                }
            }
        },
        {
            "name": "isne_get_job_status",
            "description": "Get status and progress of an ISNE training job",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to check status for"
                    }
                },
                "required": ["job_id"]
            }
        },
        {
            "name": "isne_list_jobs",
            "description": "List all ISNE training jobs and their status",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "isne_cancel_job",
            "description": "Cancel or remove an ISNE training job",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to cancel/remove"
                    }
                },
                "required": ["job_id"]
            }
        }
    ]


# MCP Tool Implementations
async def handle_mcp_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool calls for ISNE training."""
    
    if tool_name == "isne_start_training":
        request = TrainingRequest(**arguments)
        background_tasks = BackgroundTasks()
        result = await start_training(request, background_tasks)
        return result.dict()
    
    elif tool_name == "isne_get_job_status":
        job_id = arguments["job_id"]
        result = await get_job_status(job_id)
        return result.dict()
    
    elif tool_name == "isne_list_jobs":
        result = await list_jobs()
        return {job_id: job_data.dict() for job_id, job_data in result.items()}
    
    elif tool_name == "isne_cancel_job":
        job_id = arguments["job_id"]
        result = await cancel_job(job_id)
        return result
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")