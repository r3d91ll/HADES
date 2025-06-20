"""
FastAPI server implementation for HADES.

This module provides the API endpoints for interacting with the HADES system.
"""

import logging
import time
import json
from typing import List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

try:
    from fastapi_mcp import FastApiMCP
    FASTAPI_MCP_AVAILABLE = True
except ImportError:
    FastApiMCP = None
    FASTAPI_MCP_AVAILABLE = False

from src.types.api import WriteRequest, QueryRequest, WriteResponse, QueryResponse, StatusResponse, QueryResult
from .core import PathRAGSystem
from src.components.registry import get_global_registry
from .isne_training_pipeline import router as isne_training_router
from .production_pipeline import router as production_pipeline_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HADES API",
    description="Simple API for the HADES knowledge graph retrieval system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include ISNE training endpoints
app.include_router(isne_training_router)

# Include production pipeline endpoints  
app.include_router(production_pipeline_router)

# Disable OAuth for MCP integration to avoid long tool names
logger.info("OAuth disabled for MCP integration to prevent tool name length issues")

# Configure FastAPI-MCP without OAuth
auth_config = None

# Initialize FastAPI-MCP integration without OAuth
if FASTAPI_MCP_AVAILABLE:
    mcp_server = FastApiMCP(
        app,
        name="HADES",
        description="Heuristic Adaptive Data Extrapolation System - Production RAG Service"
    )

    # FastAPI-MCP automatically converts FastAPI endpoints to MCP tools
    # The production pipeline endpoints are already included via the router
    # The tools are automatically generated from the endpoint definitions

    # Mount the MCP server with registered tools
    try:
        mcp_server.mount()
        logger.info("✓ FastAPI-MCP integration enabled with registered HADES tools")
    except Exception as e:
        logger.error(f"Failed to mount FastAPI-MCP: {e}")
        logger.info("Continuing without FastAPI-MCP integration")
else:
    logger.warning("fastapi_mcp not available - MCP integration disabled")

# PathRAG system instance
_pathrag_system = None


def get_pathrag_system() -> PathRAGSystem:
    """
    Get or initialize the PathRAG system.
    
    This dependency ensures the system is lazily initialized.
    
    Returns:
        PathRAGSystem instance
    """
    global _pathrag_system
    if _pathrag_system is None:
        logger.info("Initializing PathRAG system")
        _pathrag_system = PathRAGSystem()
    return _pathrag_system


def collect_component_metrics() -> str:
    """
    Collect Prometheus metrics from all registered HADES components.
    
    This function auto-discovers components from the global registry and
    collects metrics from any component that implements export_metrics_prometheus().
    
    Returns:
        Combined Prometheus metrics string
    """
    metrics_output = []
    registry = get_global_registry()
    
    # Get all registered components
    all_components = registry.list_components()
    
    logger.info(f"Collecting metrics from component registry: {all_components}")
    
    for component_type, component_names in all_components.items():
        for component_name in component_names:
            try:
                # Get component info
                component_info = registry.get_component_info(component_type, component_name)
                if not component_info:
                    continue
                
                # Try to create component instance (with minimal config)
                try:
                    component_instance = registry.get_component(component_type, component_name, config={})
                    
                    # Check if component supports metrics export
                    if hasattr(component_instance, 'export_metrics_prometheus'):
                        logger.debug(f"Collecting metrics from {component_type}.{component_name}")
                        component_metrics = component_instance.export_metrics_prometheus()
                        
                        if component_metrics and component_metrics.strip():
                            # Add component identification comment
                            metrics_output.append(f"# Component: {component_type}.{component_name}")
                            metrics_output.append(component_metrics)
                            logger.debug(f"Collected metrics from {component_type}.{component_name}")
                        else:
                            logger.debug(f"Empty metrics from {component_type}.{component_name}")
                    else:
                        logger.debug(f"Component {component_type}.{component_name} does not support metrics export")
                        
                except Exception as e:
                    logger.warning(f"Failed to instantiate component {component_type}.{component_name}: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to collect metrics from {component_type}.{component_name}: {e}")
    
    # Add general HADES API metrics
    api_metrics = generate_api_metrics()
    if api_metrics:
        metrics_output.append("# HADES API Metrics")
        metrics_output.append(api_metrics)
    
    combined_metrics = "\n".join(metrics_output)
    logger.info(f"Collected metrics from {len([m for m in metrics_output if not m.startswith('#')])} components")
    
    return combined_metrics


def generate_api_metrics() -> str:
    """
    Generate basic API server metrics.
    
    Returns:
        Prometheus metrics string for API server
    """
    try:
        import psutil
        import time
        
        # Get current process info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Calculate uptime (approximation since server start)
        uptime_seconds = time.time() - process.create_time()
        
        api_metrics = []
        
        # API server uptime
        api_metrics.append("# HELP hades_api_uptime_seconds API server uptime in seconds")
        api_metrics.append("# TYPE hades_api_uptime_seconds gauge")
        api_metrics.append(f'hades_api_uptime_seconds{{service="hades-api"}} {uptime_seconds:.2f}')
        
        # API server memory usage
        api_metrics.append("# HELP hades_api_memory_rss_mb API server memory RSS usage in MB")
        api_metrics.append("# TYPE hades_api_memory_rss_mb gauge")
        api_metrics.append(f'hades_api_memory_rss_mb{{service="hades-api"}} {memory_info.rss / 1024 / 1024:.2f}')
        
        # API server status
        api_metrics.append("# HELP hades_api_status API server status (1=up, 0=down)")
        api_metrics.append("# TYPE hades_api_status gauge")
        api_metrics.append(f'hades_api_status{{service="hades-api"}} 1')
        
        return "\n".join(api_metrics)
        
    except Exception as e:
        logger.warning(f"Failed to generate API metrics: {e}")
        return ""


@app.post("/write", response_model=WriteResponse)
async def write(
    request: WriteRequest, 
    system: PathRAGSystem = Depends(get_pathrag_system)
) -> WriteResponse:
    """
    Write/update data in the knowledge graph.
    
    Args:
        request: Write request containing content, path, and metadata
        
    Returns:
        Status and ID of the created/updated entity
    """
    try:
        logger.info(f"Processing write request for path: {request.path}")
        entity_id = system.write(
            content=request.content,
            path=request.path,
            metadata=request.metadata
        )
        return WriteResponse(
            status="success",
            id=entity_id,
            message="Successfully processed write request"
        )
    except Exception as e:
        logger.error(f"Error processing write request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, 
    system: PathRAGSystem = Depends(get_pathrag_system)
) -> QueryResponse:
    """
    Query the PathRAG system and get results.
    
    Args:
        request: Query request containing natural language query and max_results
        
    Returns:
        Query results including content snippets and confidence scores
    """
    try:
        logger.info(f"Processing query: {request.query}")
        start_time = time.time()
        
        raw_results = system.query(
            query_text=request.query,
            max_results=request.max_results
        )
        
        # Convert raw results to QueryResult objects
        query_results: List[QueryResult] = []
        for result in raw_results:
            query_results.append(QueryResult(
                content=result.get("content", ""),
                path=result.get("path", "unknown"), # Path is required
                confidence=result.get("confidence", 0.0),
                metadata=result.get("metadata", {})
            ))
        
        processing_time = time.time() - start_time
        return QueryResponse(
            results=query_results,
            execution_time_ms=processing_time * 1000
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# OAuth discovery endpoint removed to prevent long MCP tool names


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.
    
    Returns:
        Basic API information and available endpoints
    """
    return {
        "name": "HADES API",
        "version": "0.1.0",
        "description": "Heuristic Adaptive Data Extrapolation System - Production RAG Service",
        "endpoints": {
            "POST /write": "Write/update data in the knowledge graph",
            "POST /query": "Query the PathRAG system",
            "GET /status": "Check system status",
            "GET /metrics": "Prometheus metrics",
            "GET /health": "Health check endpoint",
            "GET /docs": "OpenAPI documentation",
            "GET /redoc": "ReDoc documentation"
        }
    }


@app.post("/register")
async def register(request: Request) -> Union[Dict[str, Any], JSONResponse]:
    """
    Client registration endpoint for MCP.
    
    For lab environment, accepts any registration and returns a simple response.
    
    ⚠️ WARNING: This endpoint has no authentication or validation.
    DO NOT use in production environments.
    """
    try:
        # Get the request body to see what's being sent
        body = await request.json()
        logger.info(f"Registration request received: {body}")
        
        # For lab environment, always succeed with minimal response
        return {
            "client_id": body.get("client_id", "hades-client"),
            "client_secret": None,  # No secret needed for lab
            "redirect_uris": body.get("redirect_uris", []),
            "grant_types": [],
            "response_types": [],
            "scope": "all",
            "token_endpoint_auth_method": "none"
        }
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "error_description": "Invalid JSON"}
        )
    except Exception as e:
        logger.error(f"Error processing registration: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "server_error", "error_description": "Internal server error"}
        )


@app.get("/status", response_model=StatusResponse)
async def status(system: PathRAGSystem = Depends(get_pathrag_system)) -> StatusResponse:
    """
    Check the system status.
    
    Returns:
        System status including online status and document count
    """
    try:
        logger.info("Processing status request")
        status_info = system.get_status()
        # Convert status_info to match the expected StatusResponse format
        return StatusResponse(
            status=status_info.get("status", "unknown"),
            document_count=status_info.get("document_count", 0),
            version=status_info.get("version", "0.0.0")
        )
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Simple health status
    """
    return {"status": "healthy", "service": "HADES API"}


@app.get("/metrics")
async def metrics() -> Response:
    """
    Unified Prometheus metrics endpoint for all HADES components.
    
    This endpoint automatically discovers all registered components and collects
    metrics from any component that implements the export_metrics_prometheus() method.
    
    Returns:
        Response containing Prometheus-formatted metrics for all components
    """
    try:
        logger.info("Processing metrics request")
        
        # Collect metrics from all registered components
        combined_metrics = collect_component_metrics()
        
        if not combined_metrics.strip():
            # Return minimal metrics if no components found
            minimal_metrics = generate_api_metrics()
            combined_metrics = minimal_metrics or "# No metrics available\n"
        
        return Response(
            content=combined_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}")
        # Return error metrics instead of HTTP error to avoid breaking Prometheus scraping
        error_metrics = f"""# Error collecting metrics: {str(e)}
# HELP hades_metrics_collection_errors_total Number of metrics collection errors
# TYPE hades_metrics_collection_errors_total counter
hades_metrics_collection_errors_total{{error="collection_failed"}} 1
"""
        return Response(
            content=error_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting HADES API server")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8595, reload=True)
