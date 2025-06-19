#!/usr/bin/env python3
"""Minimal FastAPI server to isolate the issue."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
import uvicorn

# Create minimal app
app = FastAPI(title="Minimal Test Server")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Minimal server working"}

@app.get("/test")
async def test():
    return {"test": "endpoint working"}

if __name__ == "__main__":
    print("Starting minimal server on 0.0.0.0:8596...")
    uvicorn.run(
        "test_minimal_server:app",
        host="0.0.0.0",
        port=8596,
        log_level="info"
    )