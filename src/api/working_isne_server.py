#!/usr/bin/env python3
"""
Working ISNE Production Pipeline Server

Fixed version based on successful debugging.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
import uvicorn

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