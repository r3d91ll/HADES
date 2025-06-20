"""
HADES Unified Service - FastAPI application with MCP integration.

This module implements the unified HADES service architecture as specified in
HADES_SERVICE_ARCHITECTURE.md. It provides a single FastAPI service with
automatic MCP tool exposure for all RAG operations.
"""

import logging
import time
from typing import Dict, Any, List, AsyncGenerator
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi_mcp import FastApiMCP  # Disabled due to OAuth requirements
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import uvicorn
import asyncio
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service state tracking
service_state = {
    "start_time": time.time(),
    "status": "initializing",
    "components": {
        "docproc": "not_loaded",
        "chunking": "not_loaded", 
        "embedding": "not_loaded",
        "graph_enhancement": "not_loaded",
        "storage": "not_loaded"
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage service lifecycle - startup and shutdown events."""
    # Startup
    logger.info("Starting HADES unified service...")
    
    # TODO: Initialize components when they are implemented
    # This will be done incrementally as we build each component
    
    service_state["status"] = "online"
    logger.info("HADES service startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HADES service...")
    service_state["status"] = "offline"

# Initialize FastAPI app with unified HADES service configuration
app = FastAPI(
    title="HADES - Heuristic Adaptive Data Extrapolation System",
    description="Unified RAG service providing document processing, embedding, graph enhancement, and retrieval capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP integration will be initialized after app setup

# Health and status endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Service health check endpoint.
    
    Returns:
        Health status and basic service information
    """
    uptime = time.time() - service_state["start_time"]
    return {
        "status": service_state["status"],
        "service": "HADES",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "timestamp": time.time()
    }

@app.get("/status", tags=["Health"])
async def service_status() -> Dict[str, Any]:
    """
    Detailed service status including component health.
    
    Returns:
        Comprehensive service and component status information
    """
    uptime = time.time() - service_state["start_time"]
    return {
        "service": {
            "name": "HADES",
            "version": "1.0.0",
            "status": service_state["status"],
            "uptime_seconds": uptime,
            "start_time": service_state["start_time"]
        },
        "components": service_state["components"],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "docs": "/docs",
            "api_v1": "/api/v1/"
        }
    }

# API v1 route group - placeholder structure for RAG endpoints
api_v1 = FastAPI()

@api_v1.get("/", tags=["API Info"])
async def api_info() -> Dict[str, Any]:
    """API v1 information and available endpoints."""
    return {
        "api_version": "v1",
        "service": "HADES",
        "available_endpoints": [
            "/process - Document processing (coming soon)",
            "/chunk - Text chunking (coming soon)", 
            "/embed - Embedding generation (coming soon)",
            "/enhance - Graph enhancement (coming soon)",
            "/store - Data storage (coming soon)",
            "/retrieve - Data retrieval (coming soon)",
            "/pipeline - Full pipeline execution (coming soon)"
        ],
        "mcp_integration": "enabled",
        "note": "Endpoints will be added incrementally as components are implemented"
    }

# Placeholder endpoints - will be implemented as components are built
@api_v1.post("/process", tags=["Document Processing"])
async def process_documents() -> Dict[str, str]:
    """Process documents - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Document processing endpoint coming soon"}

@api_v1.post("/chunk", tags=["Chunking"])
async def chunk_text() -> Dict[str, str]:
    """Chunk text - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Text chunking endpoint coming soon"}

@api_v1.post("/embed", tags=["Embedding"])
async def generate_embeddings() -> Dict[str, str]:
    """Generate embeddings - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Embedding generation endpoint coming soon"}

@api_v1.post("/enhance", tags=["Graph Enhancement"])
async def enhance_embeddings() -> Dict[str, str]:
    """Enhance embeddings with graph information - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Graph enhancement endpoint coming soon"}

@api_v1.post("/store", tags=["Storage"])
async def store_data() -> Dict[str, str]:
    """Store data to ArangoDB - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Data storage endpoint coming soon"}

@api_v1.post("/retrieve", tags=["Retrieval"])
async def retrieve_data() -> Dict[str, str]:
    """Retrieve data and perform RAG queries - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Data retrieval endpoint coming soon"}

@api_v1.post("/pipeline", tags=["Pipeline"])
async def run_pipeline() -> Dict[str, str]:
    """Execute full RAG pipeline - placeholder endpoint."""
    return {"status": "not_implemented", "message": "Pipeline execution endpoint coming soon"}

# Import and include pipeline routes
from .pipelines import router as pipelines_router
from .production_pipeline import router as production_pipeline_router

api_v1.include_router(pipelines_router)
api_v1.include_router(production_pipeline_router)

# Mount API v1 routes
app.mount("/api/v1", api_v1)

# Note: FastAPI-MCP integration is implemented via SSE transport endpoints
# MCP functionality is provided through the /_mcp/sse and /_mcp/messages endpoints
# This enables MCP clients to interact with HADES via HTTP-based MCP protocol

logger.info("FastAPI-MCP integration enabled via SSE transport endpoints")

# Create MCP Server for SSE transport
mcp_sse_server = Server("HADES")

# Add SSE transport endpoint
@app.get("/_mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for MCP communication."""
    async def event_stream():
        # Create a queue for SSE messages
        message_queue = asyncio.Queue()
        transport = SseServerTransport("/sse", "/messages")
        
        # Add message queue to transport for communication
        transport._message_queue = message_queue
        
        async def handle_session():
            async with mcp_sse_server.create_session(transport) as session:
                await session.handle()
        
        # Start handling the session in the background
        task = asyncio.create_task(handle_session())
        
        try:
            # Stream SSE events with actual messages
            while not task.done():
                try:
                    # Wait for actual messages with timeout
                    message = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/_mcp/messages")
async def mcp_messages_endpoint(request: Request):
    """Messages endpoint for MCP communication."""
    try:
        data = await request.json()
        
        # Validate required fields
        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid request format"}
            )
        
        # Process MCP message here
        return {"status": "ok", "response": data}
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid JSON"}
        )
    except Exception as e:
        logger.error(f"Error processing MCP message: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"}
        )

def run_server(
    host: str = "0.0.0.0",
    port: int = 8595,
    reload: bool = False,
    log_level: str = "info"
) -> None:
    """
    Run the HADES unified service.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    logger.info(f"Starting HADES service on {host}:{port}")
    logger.info("Service will be available at:")
    logger.info(f"  REST API: http://{host}:{port}/api/v1/")
    logger.info(f"  MCP Integration: Enabled via FastAPI-MCP")
    logger.info(f"  API Docs: http://{host}:{port}/docs")
    logger.info(f"  Health: http://{host}:{port}/health")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

if __name__ == "__main__":
    # Development server entry point
    run_server(reload=True)