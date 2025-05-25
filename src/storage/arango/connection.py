"""Typed ArangoDB connection wrapper for HADES-PathRAG.

This replaces the legacy `src.db.arango_connection` module.
Only a subset of operations required by the ingestion pipeline are
implemented for now. Missing calls should be added as the migration
progresses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, ClassVar, Optional

from arango import ArangoClient
from arango.collection import StandardCollection
from arango.database import StandardDatabase
from arango.graph import Graph

from src.storage.arango.utils import safe_key, safe_name
from src.types.common import StorageConfig

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level convenience functions (legacy compatibility)
# ---------------------------------------------------------------------------

def get_client(hosts: str = "http://localhost:8529") -> ArangoClient:  # noqa: D401
    """Return raw ``ArangoClient`` (legacy helper)."""

    return ArangoClient(hosts=hosts)


@dataclass
class ArangoConnection:
    """Type-safe thin wrapper around ``python-arango`` client."""

    # Default collection and graph names
    DEFAULT_NODE_COLLECTION: ClassVar[str] = "nodes"
    DEFAULT_EDGE_COLLECTION: ClassVar[str] = "edges"
    DEFAULT_GRAPH_NAME: ClassVar[str] = "pathrag"
    
    __slots__ = ("config", "_db", "_client", "_username", "_password")

    config: StorageConfig

    # ------------------------------------------------------------------
    # Custom flexible constructor ------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        *,
        db_name: Optional[str] = None,
        host: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
    ) -> None:  # noqa: D401
        """Create a connection to ArangoDB.
        
        This constructor is *backwards-compatible* with the legacy signature
        used throughout the test-suite (`ArangoConnection(db_name=..., host=..., ... )`).
        
        If a ``config`` mapping is supplied we use that directly. Otherwise we
        build one from the explicit keyword arguments.
        """
        
        if config is None:
            # Build config from individual parameters for compatibility
            cfg: StorageConfig = {
                "database": db_name or "hades",
                "host": host,
                "username": username,
                "password": password,
            }
            config = cfg
        
        # dataclass emulation – we need to set the field manually and then call
        # __post_init__ which performs the real connection logic
        object.__setattr__(self, "config", config)
        
        # Establish connection inside __post_init__
        self.__post_init__()

    # ------------------------------------------------------------------
    # Internal initialisation ---------------------------------------------------
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        host = self.config.get("host", "http://localhost:8529")
        username = self.config.get("username", "root")
        password = self.config.get("password", "")
        database = self.config.get("database", "hades")

        logger.info(
            "Connecting to ArangoDB", extra={"host": host, "db": database, "user": username}
        )
        self._client: ArangoClient = ArangoClient(hosts=host)
        self._db: StandardDatabase = self._client.db(database, username=username, password=password)
        # Save credentials for helper calls
        self._username = username
        self._password = password

    # ------------------------------------------------------------------
    # Collections helpers
    # ------------------------------------------------------------------

    def get_collection(self, name: str) -> StandardCollection:
        return cast(StandardCollection, self._db.collection(safe_name(name)))

    def create_or_get_collection(self, name: str, *, edge: bool = False) -> StandardCollection:
        """Create collection if it doesn't exist, otherwise return existing one."""
        coll_name = safe_name(name)
        if not self._db.has_collection(coll_name):
            logger.info(f"Creating collection '{coll_name}' (edge: {edge})")
            return cast(StandardCollection, self._db.create_collection(coll_name, edge=edge))  # type: ignore[redundant-cast]
        return cast(StandardCollection, self._db.collection(coll_name))  # type: ignore[redundant-cast]

    def delete_collection(self, name: str) -> None:
        """Delete a collection if it exists."""
        coll_name = safe_name(name)
        if self._db.has_collection(coll_name):
            self._db.delete_collection(coll_name)
            logger.info(f"Deleted collection '{name}'")

    # legacy helpers ---------------------------------------------------------

    def collection_exists(self, name: str) -> bool:  # noqa: D401
        return bool(self._db.has_collection(safe_name(name)))

    def create_collection(self, name: str) -> StandardCollection:
        return cast(StandardCollection, self._db.create_collection(safe_name(name)))

    def create_edge_collection(self, name: str) -> StandardCollection:
        return cast(StandardCollection, self._db.create_collection(safe_name(name), edge=True))

    # ------------------------------------------------------------------
    # Basic CRUD (minimal subset)
    # ------------------------------------------------------------------

    def insert_document(
        self,
        collection: str,
        document: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        coll = self.get_collection(collection)
        if "_key" in document:
            document["_key"] = safe_key(str(document["_key"]))
        return cast(dict[str, Any], coll.insert(document, overwrite=overwrite))

    # ------------------------------------------------------------------
    # Graph helpers (stubs – extend as we migrate)
    # ------------------------------------------------------------------

    def has_graph(self, name: str) -> bool:
        return bool(self._db.has_graph(safe_name(name)))

    def graph_exists(self, name: str) -> bool:
        return bool(self._db.has_graph(safe_name(name)))
        
    def delete_graph(self, name: str, *, drop_collections: bool = False) -> None:
        """Delete a graph, optionally dropping its collections."""
        if self.graph_exists(name):
            self._db.delete_graph(safe_name(name), drop_collections=drop_collections)
            logger.info(f"Deleted graph '{name}'")

    def create_graph(self, name: str, edge_definitions: list[dict[str, Any]]) -> Graph:
        return cast(Graph, self._db.create_graph(safe_name(name), edge_definitions=edge_definitions))

    # More graph/edge helpers can be added later.

    # ------------------------------------------------------------------
    # Expose underlying db for advanced use (discouraged but handy).
    # ------------------------------------------------------------------

    @property
    def raw_db(self) -> StandardDatabase:  # noqa: D401
        """Return the underlying python-arango `StandardDatabase`."""
        return self._db
        
    # Provide access to the ArangoClient for legacy tests
    @property
    def client(self) -> ArangoClient:  # noqa: D401
        """Return the underlying `ArangoClient` (legacy helper)."""
        return self._client

    # ------------------------------------------------------------------
    # Misc helpers needed by test-suite
    # ------------------------------------------------------------------

    def database_exists(self, name: str) -> bool:  # noqa: D401
        """Check if a database exists (helper for integration tests)."""
        try:
            # Get a system database connection
            sys_db = self._client.db("_system", username=self._username, password=self._password)
            return bool(sys_db.has_database(name))
        except Exception as e:
            logger.error(f"Error checking if database exists: {e}")
            return False

    # ------------------------------------------------------------------
    # Initialization and bootstrap helpers
    # ------------------------------------------------------------------
    
    @classmethod
    def bootstrap(cls, config: StorageConfig, *, 
                node_collection: Optional[str] = None,
                edge_collection: Optional[str] = None,
                graph_name: Optional[str] = None,
                force: bool = False) -> "ArangoConnection":
        """Bootstrap a database with the basic schema structure.
        
        Creates database, collections, and graph if they don't exist.
        Safe to call multiple times unless force=True, which recreates everything.
        
        Args:
            config: Database connection config
            node_collection: Name for node collection (default: "nodes")
            edge_collection: Name for edge collection (default: "edges")
            graph_name: Name for primary graph (default: "pathrag")
            force: If True, drops existing collections/graph before creating 
        
        Returns:
            Connection to bootstrapped database
        """
        # Use defaults if not provided
        node_collection = node_collection or cls.DEFAULT_NODE_COLLECTION
        edge_collection = edge_collection or cls.DEFAULT_EDGE_COLLECTION
        graph_name = graph_name or cls.DEFAULT_GRAPH_NAME
        
        # Create database if needed
        host = config.get("host", "http://localhost:8529")
        username = config.get("username", "root")
        password = config.get("password", "")
        database = config.get("database", "hades")
        
        client = ArangoClient(hosts=host)
        sys_db = client.db("_system", username=username, password=password)
        
        if not sys_db.has_database(database):
            logger.info(f"Creating database '{database}'")
            sys_db.create_database(database)
        
        # Connect to database
        conn = cls(config=config)
        
        # Handle force recreation
        if force:
            # Drop graph first (to avoid foreign key constraints)
            if conn.graph_exists(graph_name):
                conn.delete_graph(graph_name)
                logger.info(f"Dropped existing graph '{graph_name}'")
            
            # Drop collections
            if conn.collection_exists(node_collection):
                conn.delete_collection(node_collection)
                logger.info(f"Dropped existing collection '{node_collection}'")
                
            if conn.collection_exists(edge_collection):
                conn.delete_collection(edge_collection)
                logger.info(f"Dropped existing collection '{edge_collection}'")
        
        # Create collections
        conn.create_or_get_collection(node_collection)
        conn.create_or_get_collection(edge_collection, edge=True)
        
        # Create graph if needed
        if not conn.graph_exists(graph_name):
            edge_definitions = [{
                'edge_collection': edge_collection,
                'from_vertex_collections': [node_collection],
                'to_vertex_collections': [node_collection]
            }]
            conn.create_graph(graph_name, edge_definitions)
            logger.info(f"Created graph '{graph_name}'")
        
        return conn
