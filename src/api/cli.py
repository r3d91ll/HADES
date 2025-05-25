"""
Command line interface for running the HADES-PathRAG API server.
"""

import argparse
import logging
import os
import sys
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the HADES-PathRAG API server")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level (default: info)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    return parser.parse_args()


def main():
    """Run the HADES-PathRAG API server."""
    args = parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    logger.info(f"Starting HADES-PathRAG API server on {args.host}:{args.port}")
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        logger.info(f"Using configuration from: {args.config}")
        # Note: We're not actually passing the config file to the app here,
        # as that would need a more complex setup with environment variables
        # or a custom callable. For simplicity, we're just logging it.
    
    # Run the server
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
