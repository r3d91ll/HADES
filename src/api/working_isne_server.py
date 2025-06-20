#!/usr/bin/env python3
"""
Working ISNE Production Pipeline Server

Fixed version based on successful debugging.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi_mcp import FastApiMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import uvicorn
import asyncio
import json
import logging

# Import our production pipeline router
from src.api.production_pipeline import router as production_router

# Create FastAPI app
app = FastAPI(
    title="HADES ISNE Production Pipelines",
    description="Production-ready ISNE bootstrap, training, and application pipelines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HADES ISNE Pipelines",
        "production_routes": len(production_router.routes),
        "endpoints": {
            "bootstrap": "/production/bootstrap",
            "train": "/production/train", 
            "apply_model": "/production/apply-model",
            "build_collections": "/production/build-collections",
            "status": "/production/status",
        }
    }

# Include production pipeline router
app.include_router(production_router)

# Initialize MCP integration
mcp = FastApiMCP(app)

# Mount the MCP server directly to your FastAPI app
mcp.mount()

# Configure logging
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    print("🚀 Starting HADES ISNE Pipeline Server...")
    print("📡 Server will be available at http://0.0.0.0:8595")
    print("📚 API documentation at http://0.0.0.0:8595/docs")
    print("🔧 MCP integration enabled")
    print()
    print("Available endpoints:")
    print("  /production/bootstrap - Bootstrap ISNE dataset")
    print("  /production/train - Train ISNE model")
    print("  /production/apply-model - Apply trained model")
    print("  /production/build-collections - Build semantic collections")
    print("  /production/status - Check operation status")
    print()
    
    # Use the direct app reference instead of module string
    uvicorn.run(
        app,  # Direct app reference, not string
        host="0.0.0.0",
        port=8595,
        log_level="info"
    )