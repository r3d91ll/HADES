#!/usr/bin/env python3
"""
Startup script for HADES unified service.

This script provides a convenient way to start the HADES service
with different configurations for development and production.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.main import run_server

def main():
    parser = argparse.ArgumentParser(description="Start HADES unified service")
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting HADES - Heuristic Adaptive Data Extrapolation System")
    print(f"📡 Service will be available at http://{args.host}:{args.port}")
    print("📚 API documentation at /docs")
    print("🔧 MCP integration enabled")
    print()
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()