#!/usr/bin/env python3
"""Test server with MCP integration."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
import uvicorn

# Create app
app = FastAPI(title="Test MCP Integration")

@app.get("/health")
async def health():
    return {"status": "ok", "mcp": "enabled"}

print("Initializing MCP...")
try:
    # Initialize MCP integration
    mcp_server = FastApiMCP(
        app,
        name="Test-MCP",
        description="Test MCP Server"
    )
    print("✓ MCP initialized successfully")
except Exception as e:
    print(f"✗ MCP initialization failed: {e}")

if __name__ == "__main__":
    print("Starting server with MCP...")
    uvicorn.run(
        "test_with_mcp:app",
        host="0.0.0.0",
        port=8598,
        log_level="info"
    )