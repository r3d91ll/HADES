#!/usr/bin/env python3
"""
Launch script for HADES MCP Server.

This script sets up the environment and starts the MCP server.
"""

import os
import sys
import asyncio
import argparse
import yaml
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.server import ArxivMCPServer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    
    return expand_env_vars(config)


def setup_logging(log_level: str):
    """Configure logging for the server."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hades_mcp_server.log')
        ]
    )


async def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description='HADES MCP Server')
    parser.add_argument(
        '--config',
        default='mcp_server/config/server_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--name',
        help='Override server name from config'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Override server port from config'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.name:
        config['server']['name'] = args.name
    if args.port:
        config['server']['port'] = args.port
    if args.log_level:
        config['server']['log_level'] = args.log_level
    
    # Setup logging
    setup_logging(config['server']['log_level'])
    logger = logging.getLogger(__name__)
    
    # Set environment variables from config
    os.environ['ARANGO_HOST'] = config['database']['host']
    os.environ['ARANGO_PORT'] = str(config['database']['port'])
    os.environ['ARANGO_USERNAME'] = config['database']['username']
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("HADES MCP Server Starting")
    logger.info("=" * 60)
    logger.info(f"Server Name: {config['server']['name']}")
    logger.info(f"Port: {config['server']['port']}")
    logger.info(f"Database: {config['database']['default_db']}")
    logger.info(f"Collection: {config['database']['default_collection']}")
    logger.info(f"GPU Enabled: {config['gpu']['enabled']}")
    if config['gpu']['enabled']:
        logger.info(f"GPU Devices: {config['gpu']['devices']}")
    logger.info("=" * 60)
    
    # Create and run server
    try:
        server = ArxivMCPServer(name=config['server']['name'])
        
        # Store config for server to use
        server.config = config
        
        logger.info("Starting MCP server...")
        await server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())