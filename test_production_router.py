#!/usr/bin/env python3
"""Test server with just production router, no MCP."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
import uvicorn

print("Testing production router import...")
try:
    from src.api.production_pipeline import router as production_router
    print("✓ Production router imported successfully")
except Exception as e:
    print(f"✗ Production router import failed: {e}")
    sys.exit(1)

# Create app with production router
app = FastAPI(title="Test Production Router")

@app.get("/health")
async def health():
    return {"status": "ok", "production_routes": len(production_router.routes)}

# Include production router
app.include_router(production_router)

if __name__ == "__main__":
    print(f"Starting server with {len(production_router.routes)} production routes...")
    uvicorn.run(
        "test_production_router:app",
        host="0.0.0.0",
        port=8597,
        log_level="info"
    )