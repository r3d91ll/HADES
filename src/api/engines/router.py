"""
FastAPI router for the multi-engine RAG platform.

This module provides REST API endpoints for:
- Engine-specific retrieval
- A/B testing and comparison
- Configuration management
- Performance monitoring
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .base import (
    EngineType, RetrievalMode, EngineRetrievalRequest, EngineRetrievalResponse,
    EngineConfigRequest, EngineConfigResponse, ExperimentRequest, ExperimentResponse,
    get_engine_registry
)
from .comparison import get_experiment_manager, ComparisonResult, ComparisonMetric
from .pathrag_engine import PathRAGEngine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/engines", tags=["Multi-Engine RAG"])


# Additional request/response models
class EngineStatusResponse(BaseModel):
    """Response with engine status information."""
    engine_type: EngineType
    is_healthy: bool
    supported_modes: List[RetrievalMode]
    metrics: Dict[str, Any]
    last_updated: datetime


class ComparisonRequest(BaseModel):
    """Request for comparing multiple engines."""
    query: str = Field(..., description="Query to test across engines")
    engines: List[EngineType] = Field(..., description="Engines to compare")
    mode: RetrievalMode = Field(default=RetrievalMode.FULL_ENGINE, description="Retrieval mode")
    top_k: int = Field(default=10, description="Number of results to retrieve")
    engine_configs: Optional[Dict[EngineType, Dict[str, Any]]] = Field(default=None, description="Engine-specific configs")
    metrics: List[ComparisonMetric] = Field(default=[ComparisonMetric.RELEVANCE, ComparisonMetric.SPEED], description="Metrics to calculate")


class PathAnalysisRequest(BaseModel):
    """Request for PathRAG path analysis."""
    query: str = Field(..., description="Query to analyze")
    top_k: int = Field(default=5, description="Number of paths to analyze")


class EngineListResponse(BaseModel):
    """Response with list of available engines."""
    engines: Dict[EngineType, Dict[str, Any]]
    default_engine: Optional[EngineType]
    total_engines: int


# Initialize engines on startup
async def initialize_engines() -> None:
    """Initialize all available engines."""
    registry = get_engine_registry()
    
    try:
        # Initialize PathRAG engine
        pathrag_engine = PathRAGEngine()
        registry.register_engine(pathrag_engine, is_default=True)
        logger.info("PathRAG engine registered")
        
        # TODO: Add other engines (SageGraph, etc.) as they become available
        
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")


# Engine Management Endpoints

@router.on_event("startup")
async def startup_event() -> None:
    """Initialize engines on router startup."""
    await initialize_engines()


@router.get("/", response_model=EngineListResponse, operation_id="list_engines")
async def list_engines() -> EngineListResponse:
    """
    List all available RAG engines and their capabilities.
    
    Returns information about each registered engine including:
    - Engine type and status
    - Supported retrieval modes
    - Current configuration
    - Performance metrics
    """
    registry = get_engine_registry()
    engines_info = {}
    
    for engine_type in registry.list_engines():
        engine = registry.get_engine(engine_type)
        if engine:
            try:
                engines_info[engine_type] = {
                    "supported_modes": [mode.value for mode in engine.supported_modes],
                    "is_healthy": await engine.health_check(),
                    "metrics": await engine.get_metrics(),
                    "config_schema": await engine.get_config_schema()
                }
            except Exception as e:
                engines_info[engine_type] = {
                    "error": str(e),
                    "is_healthy": False
                }
    
    default_engine = None
    default_engine_obj = registry.get_default_engine()
    if default_engine_obj:
        default_engine = default_engine_obj.engine_type
    
    return EngineListResponse(
        engines=engines_info,
        default_engine=default_engine,
        total_engines=len(engines_info)
    )


@router.get("/{engine_type}/status", response_model=EngineStatusResponse, operation_id="get_engine_status")
async def get_engine_status(engine_type: EngineType) -> EngineStatusResponse:
    """
    Get detailed status information for a specific engine.
    
    Args:
        engine_type: Type of engine to check
        
    Returns:
        Detailed status including health, metrics, and capabilities
    """
    registry = get_engine_registry()
    engine = registry.get_engine(engine_type)
    
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
    
    try:
        return EngineStatusResponse(
            engine_type=engine_type,
            is_healthy=await engine.health_check(),
            supported_modes=engine.supported_modes,
            metrics=await engine.get_metrics(),
            last_updated=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get status for {engine_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")


# Engine-Specific Retrieval Endpoints

@router.post("/{engine_type}/retrieve", response_model=EngineRetrievalResponse, operation_id="retrieve_with_engine")
async def retrieve_with_engine(
    engine_type: EngineType,
    request: EngineRetrievalRequest
) -> EngineRetrievalResponse:
    """
    Perform retrieval using a specific RAG engine.
    
    This endpoint allows you to query a specific engine with full control
    over retrieval parameters and engine-specific configurations.
    
    Args:
        engine_type: Type of engine to use for retrieval
        request: Retrieval request with query and parameters
        
    Returns:
        Retrieval response with results and metadata
    """
    registry = get_engine_registry()
    engine = registry.get_engine(engine_type)
    
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
    
    try:
        # Validate mode support
        if request.mode not in engine.supported_modes:
            raise HTTPException(
                status_code=400, 
                detail=f"Mode {request.mode} not supported by {engine_type}. Supported modes: {[m.value for m in engine.supported_modes]}"
            )
        
        # Execute retrieval
        response = await engine.retrieve(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed for {engine_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.post("/{engine_type}/configure", response_model=Dict[str, Any], operation_id="configure_engine")
async def configure_engine(
    engine_type: EngineType,
    request: EngineConfigRequest
) -> Dict[str, Any]:
    """
    Update configuration for a specific engine.
    
    Args:
        engine_type: Type of engine to configure
        request: Configuration update request
        
    Returns:
        Success status and updated configuration
    """
    registry = get_engine_registry()
    engine = registry.get_engine(engine_type)
    
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
    
    try:
        success = await engine.configure(request.config_updates)
        
        if success:
            current_config = await engine.get_config()
            return {
                "success": True,
                "message": f"Configuration updated for {engine_type}",
                "current_config": current_config,
                "temporary": request.temporary
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration update failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration failed for {engine_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/{engine_type}/config", response_model=EngineConfigResponse, operation_id="get_engine_config")
async def get_engine_config(engine_type: EngineType) -> EngineConfigResponse:
    """
    Get current configuration for a specific engine.
    
    Args:
        engine_type: Type of engine to get configuration for
        
    Returns:
        Current configuration and schema
    """
    registry = get_engine_registry()
    engine = registry.get_engine(engine_type)
    
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
    
    try:
        return EngineConfigResponse(
            engine_type=engine_type,
            current_config=await engine.get_config(),
            config_schema=await engine.get_config_schema(),
            last_updated=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get config for {engine_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


# A/B Testing and Comparison Endpoints

@router.post("/compare", response_model=ComparisonResult, operation_id="compare_engines")
async def compare_engines(request: ComparisonRequest) -> ComparisonResult:
    """
    Compare multiple engines on a single query.
    
    This endpoint runs the same query across multiple engines and provides
    detailed comparison metrics including speed, relevance, and diversity.
    
    Args:
        request: Comparison request with query and engines to test
        
    Returns:
        Detailed comparison results with metrics and analysis
    """
    experiment_manager = get_experiment_manager()
    
    try:
        # Validate engines exist
        registry = get_engine_registry()
        available_engines = registry.list_engines()
        
        for engine_type in request.engines:
            if engine_type not in available_engines:
                raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
        
        # Run comparison
        result = await experiment_manager.run_comparison(
            query=request.query,
            engines=request.engines,
            mode=request.mode,
            top_k=request.top_k,
            engine_configs=request.engine_configs,
            metrics=request.metrics
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Engine comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/experiments", response_model=Dict[str, Any], operation_id="start_experiment")
async def start_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start a new A/B testing experiment.
    
    This endpoint starts a comprehensive experiment that runs multiple queries
    across multiple engines with detailed metrics collection and analysis.
    
    Args:
        request: Experiment configuration
        background_tasks: FastAPI background tasks for async execution
        
    Returns:
        Experiment ID and status information
    """
    experiment_manager = get_experiment_manager()
    
    try:
        # Validate engines exist
        registry = get_engine_registry()
        available_engines = registry.list_engines()
        
        for engine_type in request.engines:
            if engine_type not in available_engines:
                raise HTTPException(status_code=404, detail=f"Engine {engine_type} not found")
        
        # Start experiment
        experiment_id = await experiment_manager.start_experiment(request)
        
        return {
            "experiment_id": experiment_id,
            "status": "started",
            "message": f"Experiment '{request.name}' started with {len(request.test_queries)} queries across {len(request.engines)} engines",
            "estimated_duration_minutes": len(request.test_queries) * len(request.engines) * request.iterations * 0.1,  # Rough estimate
            "status_endpoint": f"/engines/experiments/{experiment_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start experiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse, operation_id="get_experiment")
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    """
    Get experiment results and status.
    
    Args:
        experiment_id: ID of the experiment to retrieve
        
    Returns:
        Complete experiment results and analysis
    """
    experiment_manager = get_experiment_manager()
    
    experiment = await experiment_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    return experiment


@router.get("/experiments", response_model=List[ExperimentResponse], operation_id="list_experiments")
async def list_experiments() -> List[ExperimentResponse]:
    """
    List all experiments with their current status.
    
    Returns:
        List of all experiments ordered by creation date
    """
    experiment_manager = get_experiment_manager()
    experiments = await experiment_manager.list_experiments()
    return sorted(experiments, key=lambda x: x.created_at, reverse=True)


@router.delete("/experiments/{experiment_id}", operation_id="cancel_experiment")
async def cancel_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Cancel a running experiment.
    
    Args:
        experiment_id: ID of the experiment to cancel
        
    Returns:
        Cancellation status
    """
    experiment_manager = get_experiment_manager()
    
    success = await experiment_manager.cancel_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found or not running")
    
    return {"message": f"Experiment {experiment_id} cancelled successfully"}


# PathRAG-Specific Endpoints

@router.post("/pathrag/analyze-paths", response_model=Dict[str, Any], operation_id="analyze_paths")
async def analyze_pathrag_paths(request: PathAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze PathRAG path construction and scoring.
    
    This endpoint provides detailed insight into how PathRAG constructs
    and scores paths for a given query, useful for debugging and optimization.
    
    Args:
        request: Path analysis request
        
    Returns:
        Detailed path analysis with resource allocation and scoring information
    """
    registry = get_engine_registry()
    engine = registry.get_engine(EngineType.PATHRAG)
    
    if not engine:
        raise HTTPException(status_code=404, detail="PathRAG engine not available")
    
    # Check if it's actually a PathRAG engine
    if not isinstance(engine, PathRAGEngine):
        raise HTTPException(status_code=500, detail="Engine is not a PathRAG instance")
    
    try:
        analysis = await engine.analyze_paths(request.query, request.top_k)
        return analysis
    except Exception as e:
        logger.error(f"PathRAG path analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Path analysis failed: {str(e)}")


@router.get("/pathrag/knowledge-base", response_model=Dict[str, Any], operation_id="get_kb_stats")
async def get_pathrag_knowledge_base_stats() -> Dict[str, Any]:
    """
    Get PathRAG knowledge base statistics.
    
    Returns detailed information about the knowledge graph including
    node/edge counts, connectivity metrics, and data distribution.
    
    Returns:
        Knowledge base statistics and health information
    """
    registry = get_engine_registry()
    engine = registry.get_engine(EngineType.PATHRAG)
    
    if not engine:
        raise HTTPException(status_code=404, detail="PathRAG engine not available")
    
    if not isinstance(engine, PathRAGEngine):
        raise HTTPException(status_code=500, detail="Engine is not a PathRAG instance")
    
    try:
        stats = await engine.get_knowledge_base_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get PathRAG knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base stats: {str(e)}")


# Health Check Endpoint

@router.get("/health", operation_id="health_check")
async def health_check() -> Dict[str, Any]:
    """
    Check health of all engines.
    
    Returns:
        Health status for each registered engine
    """
    registry = get_engine_registry()
    health_results = await registry.health_check_all()
    
    overall_healthy = all(health_results.values())
    
    return {
        "overall_healthy": overall_healthy,
        "engines": {engine_type.value: status for engine_type, status in health_results.items()},
        "total_engines": len(health_results),
        "healthy_engines": sum(health_results.values()),
        "timestamp": datetime.utcnow().isoformat()
    }