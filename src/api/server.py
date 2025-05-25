"""
FastAPI server implementation for HADES-PathRAG.

This module provides the API endpoints for interacting with the HADES-PathRAG system.
"""

import logging
import time
from typing import Optional, Dict, Any, List, cast
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from .models import WriteRequest, QueryRequest, WriteResponse, QueryResponse, StatusResponse, QueryResult
from .core import PathRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HADES-PathRAG API",
    description="Simple API for the HADES-PathRAG knowledge graph retrieval system",
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


@app.post("/write", response_model=WriteResponse)
async def write(request: WriteRequest, system: PathRAGSystem = Depends(get_pathrag_system)) -> WriteResponse:
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
async def query(request: QueryRequest, system: PathRAGSystem = Depends(get_pathrag_system)) -> QueryResponse:
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


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting HADES-PathRAG API server")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
