"""
ArangoDB Database Interface

Provides optimized ArangoDB connections including Unix socket support
for improved performance and atomic transaction handling.
"""

# Note: legacy Arango clients have been archived to Acheron. The optimized
# HTTP/2 client and memory wrapper provide the current integration surface.

from .memory_client import (
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
    CollectionDefinition,
    MemoryServiceError,
    resolve_memory_config,
)
from .optimized_client import ArangoHttp2Client, ArangoHttp2Config, ArangoHttpError

__all__ = [
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
    "ArangoMemoryClient",
    "ArangoMemoryClientConfig",
    "CollectionDefinition",
    "MemoryServiceError",
    "resolve_memory_config",
]
