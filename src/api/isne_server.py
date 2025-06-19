#!/usr/bin/env python3
"""
ISNE Production Pipeline Server

Minimal FastAPI server that exposes ONLY the ISNE production pipelines
we've built and tested over the last 2 days.
"""

import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app - focused only on ISNE pipelines
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
        "endpoints": {
            "bootstrap": "/production/bootstrap",
            "train": "/production/train",
            "apply_model": "/production/apply-model",
            "build_collections": "/production/build-collections",
            "status": "/production/status",
        }
    }

# Include ONLY our production pipeline router
app.include_router(production_router)

# Initialize MCP integration
mcp_server = FastApiMCP(
    app,
    name="HADES-ISNE",
    description="HADES ISNE Production Pipeline MCP Server"
)

def run_server(
    host: str = "0.0.0.0",
    port: int = 8595,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the ISNE pipeline server."""
    logger.info(f"Starting HADES ISNE Pipeline Server on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  /production/bootstrap - Bootstrap ISNE dataset")
    logger.info("  /production/train - Train ISNE model")
    logger.info("  /production/apply-model - Apply trained model")
    logger.info("  /production/build-collections - Build semantic collections")
    logger.info("  /production/status - Check operation status")
    logger.info("  /docs - Interactive API documentation")
    
    uvicorn.run(
        "src.api.isne_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

if __name__ == "__main__":
    run_server(reload=True)