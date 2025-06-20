"""
ArangoDB schema management for incremental storage.

This module defines the database schema, creates collections and indexes,
and provides validation for all data structures used in incremental storage.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    CollectionCreateError, 
    IndexCreateError, 
    DatabaseCreateError,
    ArangoError
)

from .types import IncrementalConfig, DatabaseStatus

logger = logging.getLogger(__name__)


@dataclass
class CollectionDef:
    """Definition of a collection."""
    name: str
    edge: bool = False
    indexes: List[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None


@dataclass 
class IndexDef:
    """Definition of an index."""
    collection: str
    fields: List[str]
    type: str = "persistent"
    unique: bool = False
    sparse: bool = False
    name: Optional[str] = None


class IncrementalSchema:
    """
    Schema definition for incremental storage collections.
    
    Defines 9 specialized collections for documents, chunks, embeddings,
    graph nodes/edges, models, and change tracking.
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = "1.0.0"
    
    # Collection definitions
    COLLECTIONS = [
        # Document storage
        CollectionDef(
            name="documents",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "size": {"type": "integer"},
                    "modified_time": {"type": "string", "format": "date-time"},
                    "ingestion_time": {"type": "string", "format": "date-time"},
                    "metadata": {"type": "object"},
                    "processing_status": {"type": "string", "enum": ["pending", "processing", "completed", "error"]},
                    "chunk_count": {"type": "integer", "minimum": 0}
                },
                "required": ["file_path", "content_hash", "size", "modified_time"]
            }
        ),
        
        # Chunk storage
        CollectionDef(
            name="chunks",
            edge=False,
            schema={
                "type": "object", 
                "properties": {
                    "_key": {"type": "string"},
                    "document_id": {"type": "string"},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "start_pos": {"type": "integer", "minimum": 0},
                    "end_pos": {"type": "integer", "minimum": 0},
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "created_at": {"type": "string", "format": "date-time"},
                    "metadata": {"type": "object"},
                    "embedding_id": {"type": "string"}
                },
                "required": ["document_id", "content", "content_hash", "start_pos", "end_pos"]
            }
        ),
        
        # Embedding storage
        CollectionDef(
            name="embeddings",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "vector": {"type": "array", "items": {"type": "number"}},
                    "model_name": {"type": "string"},
                    "model_version": {"type": "string"},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "created_at": {"type": "string", "format": "date-time"},
                    "metadata": {"type": "object"}
                },
                "required": ["chunk_id", "vector", "model_name", "embedding_dim"]
            }
        ),
        
        # Graph nodes
        CollectionDef(
            name="nodes",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "embedding_id": {"type": "string"},
                    "node_index": {"type": "integer", "minimum": 0},
                    "created_at": {"type": "string", "format": "date-time"},
                    "isne_enhanced": {"type": "boolean"},
                    "metadata": {"type": "object"}
                },
                "required": ["chunk_id", "embedding_id", "node_index"]
            }
        ),
        
        # Graph edges
        CollectionDef(
            name="edges",
            edge=True,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "_from": {"type": "string"},
                    "_to": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "edge_type": {"type": "string", "enum": ["similarity", "structural", "temporal", "semantic"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "temporal_bounds": {"type": "object"},
                    "metadata": {"type": "object"}
                },
                "required": ["_from", "_to", "weight", "edge_type"]
            }
        ),
        
        # Model versions
        CollectionDef(
            name="models",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "version_id": {"type": "string"},
                    "version_number": {"type": "string"},
                    "model_type": {"type": "string", "enum": ["isne", "pathrag", "baseline"]},
                    "node_count": {"type": "integer", "minimum": 0},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "hidden_dim": {"type": "integer", "minimum": 1},
                    "num_layers": {"type": "integer", "minimum": 1},
                    "model_parameters": {"type": "integer", "minimum": 0},
                    "model_path": {"type": "string"},
                    "training_config": {"type": "object"},
                    "performance_metrics": {"type": "object"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "created_by": {"type": "string"},
                    "description": {"type": "string"},
                    "parent_version": {"type": "string"},
                    "is_current": {"type": "boolean"},
                    "metadata": {"type": "object"}
                },
                "required": ["version_id", "model_type", "node_count", "embedding_dim", "model_path"]
            }
        ),
        
        # Ingestion logs
        CollectionDef(
            name="ingestion_logs",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "batch_id": {"type": "string"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "input_path": {"type": "string"},
                    "total_documents": {"type": "integer", "minimum": 0},
                    "processed_documents": {"type": "integer", "minimum": 0},
                    "new_documents": {"type": "integer", "minimum": 0},
                    "updated_documents": {"type": "integer", "minimum": 0},
                    "error_documents": {"type": "integer", "minimum": 0},
                    "status": {"type": "string", "enum": ["running", "completed", "failed", "cancelled"]},
                    "config": {"type": "object"},
                    "results": {"type": "object"},
                    "errors": {"type": "array"},
                    "metadata": {"type": "object"}
                },
                "required": ["batch_id", "start_time", "input_path", "status"]
            }
        ),
        
        # Model training history
        CollectionDef(
            name="model_versions",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "model_id": {"type": "string"},
                    "training_job_id": {"type": "string"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "training_data_hash": {"type": "string"},
                    "data_version": {"type": "string"},
                    "training_config": {"type": "object"},
                    "final_loss": {"type": "number"},
                    "best_loss": {"type": "number"},
                    "epochs_trained": {"type": "integer", "minimum": 0},
                    "convergence_achieved": {"type": "boolean"},
                    "validation_metrics": {"type": "object"},
                    "model_size_mb": {"type": "number", "minimum": 0},
                    "status": {"type": "string", "enum": ["training", "completed", "failed", "cancelled"]},
                    "metadata": {"type": "object"}
                },
                "required": ["model_id", "training_job_id", "start_time", "status"]
            }
        ),
        
        # Conflict resolution records
        CollectionDef(
            name="conflicts",
            edge=False,
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "document_id": {"type": "string"},
                    "file_path": {"type": "string"},
                    "conflict_type": {"type": "string", "enum": ["content_change", "duplicate", "version_mismatch", "metadata_conflict"]},
                    "detection_time": {"type": "string", "format": "date-time"},
                    "resolution_time": {"type": "string", "format": "date-time"},
                    "strategy_used": {"type": "string", "enum": ["skip", "update", "merge", "keep_both"]},
                    "original_hash": {"type": "string"},
                    "new_hash": {"type": "string"},
                    "resolution_details": {"type": "object"},
                    "auto_resolved": {"type": "boolean"},
                    "reviewer": {"type": "string"},
                    "metadata": {"type": "object"}
                },
                "required": ["document_id", "file_path", "conflict_type", "detection_time", "strategy_used"]
            }
        )
    ]
    
    # Index definitions for performance optimization
    INDEXES = [
        # Documents indexes
        IndexDef("documents", ["content_hash"], unique=True, name="documents_content_hash"),
        IndexDef("documents", ["file_path"], unique=True, name="documents_file_path"),
        IndexDef("documents", ["modified_time"], name="documents_modified_time"),
        IndexDef("documents", ["processing_status"], name="documents_status"),
        
        # Chunks indexes  
        IndexDef("chunks", ["document_id"], name="chunks_document_id"),
        IndexDef("chunks", ["content_hash"], name="chunks_content_hash"),
        IndexDef("chunks", ["embedding_id"], name="chunks_embedding_id"),
        IndexDef("chunks", ["document_id", "chunk_index"], unique=True, name="chunks_doc_index"),
        
        # Embeddings indexes
        IndexDef("embeddings", ["chunk_id"], unique=True, name="embeddings_chunk_id"),
        IndexDef("embeddings", ["model_name", "model_version"], name="embeddings_model"),
        
        # Nodes indexes
        IndexDef("nodes", ["chunk_id"], unique=True, name="nodes_chunk_id"),
        IndexDef("nodes", ["embedding_id"], name="nodes_embedding_id"),
        IndexDef("nodes", ["node_index"], unique=True, name="nodes_index"),
        
        # Edges indexes
        IndexDef("edges", ["_from"], name="edges_from"),
        IndexDef("edges", ["_to"], name="edges_to"),
        IndexDef("edges", ["weight"], name="edges_weight"),
        IndexDef("edges", ["edge_type"], name="edges_type"),
        IndexDef("edges", ["_from", "_to"], unique=True, name="edges_from_to"),
        
        # Models indexes
        IndexDef("models", ["version_id"], unique=True, name="models_version_id"),
        IndexDef("models", ["is_current"], name="models_current"),
        IndexDef("models", ["model_type"], name="models_type"),
        IndexDef("models", ["created_at"], name="models_created"),
        
        # Ingestion logs indexes
        IndexDef("ingestion_logs", ["batch_id"], unique=True, name="logs_batch_id"),
        IndexDef("ingestion_logs", ["start_time"], name="logs_start_time"),
        IndexDef("ingestion_logs", ["status"], name="logs_status"),
        
        # Model versions indexes
        IndexDef("model_versions", ["model_id"], name="versions_model_id"),
        IndexDef("model_versions", ["training_job_id"], unique=True, name="versions_job_id"),
        IndexDef("model_versions", ["start_time"], name="versions_start_time"),
        IndexDef("model_versions", ["status"], name="versions_status"),
        
        # Conflicts indexes
        IndexDef("conflicts", ["document_id"], name="conflicts_document_id"),
        IndexDef("conflicts", ["file_path"], name="conflicts_file_path"),
        IndexDef("conflicts", ["conflict_type"], name="conflicts_type"),
        IndexDef("conflicts", ["detection_time"], name="conflicts_detection_time"),
        IndexDef("conflicts", ["auto_resolved"], name="conflicts_auto_resolved")
    ]


class SchemaManager:
    """
    Manages ArangoDB schema for incremental storage.
    
    Handles database initialization, collection creation, index management,
    schema validation, and migrations.
    """
    
    def __init__(self, client: ArangoClient, config: IncrementalConfig):
        """
        Initialize schema manager.
        
        Args:
            client: ArangoDB client instance
            config: Incremental storage configuration
        """
        self.client = client
        self.config = config
        self.db_name = config.db_name
        self.db: Optional[StandardDatabase] = None
        self.schema = IncrementalSchema()
        
    def initialize_database(self) -> bool:
        """
        Initialize database with collections and indexes.
        
        Returns:
            True if initialization successful
            
        Raises:
            DatabaseCreateError: If database creation fails
            CollectionCreateError: If collection creation fails
            IndexCreateError: If index creation fails
        """
        try:
            # Create database if it doesn't exist
            sys_db = self.client.db("_system")
            if not sys_db.has_database(self.db_name):
                logger.info(f"Creating database: {self.db_name}")
                sys_db.create_database(self.db_name)
            
            # Connect to database
            self.db = self.client.db(self.db_name)
            
            # Create collections
            self._create_collections()
            
            # Create indexes
            self._create_indexes()
            
            # Store schema version
            self._store_schema_version()
            
            logger.info(f"Database '{self.db_name}' initialized successfully")
            return True
            
        except ArangoError as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
            
    def _create_collections(self) -> None:
        """Create all required collections."""
        for collection_def in self.schema.COLLECTIONS:
            if not self.db.has_collection(collection_def.name):
                logger.info(f"Creating collection: {collection_def.name}")
                
                if collection_def.edge:
                    self.db.create_collection(collection_def.name, edge=True)
                else:
                    self.db.create_collection(collection_def.name)
            else:
                logger.debug(f"Collection '{collection_def.name}' already exists")
                
    def _create_indexes(self) -> None:
        """Create all required indexes."""
        for index_def in self.schema.INDEXES:
            collection = self.db.collection(index_def.collection)
            
            # Check if index already exists
            existing_indexes = collection.indexes()
            index_exists = any(
                set(idx.get("fields", [])) == set(index_def.fields)
                for idx in existing_indexes
            )
            
            if not index_exists:
                logger.info(f"Creating index on {index_def.collection}: {index_def.fields}")
                
                try:
                    collection.add_index({
                        "type": index_def.type,
                        "fields": index_def.fields,
                        "unique": index_def.unique,
                        "sparse": index_def.sparse,
                        "name": index_def.name
                    })
                except IndexCreateError as e:
                    logger.warning(f"Failed to create index {index_def.name}: {e}")
            else:
                logger.debug(f"Index on {index_def.collection}:{index_def.fields} already exists")
                
    def _store_schema_version(self) -> None:
        """Store current schema version in database."""
        collection = self.db.collection("models")
        
        version_doc = {
            "_key": "schema_version",
            "version": self.schema.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collections": [col.name for col in self.schema.COLLECTIONS],
            "indexes": len(self.schema.INDEXES)
        }
        
        collection.insert(version_doc, overwrite=True)
        logger.info(f"Stored schema version: {self.schema.SCHEMA_VERSION}")
        
    def validate_schema(self) -> bool:
        """
        Validate current database schema against expected schema.
        
        Returns:
            True if schema is valid
        """
        try:
            if not self.db:
                self.db = self.client.db(self.db_name)
            
            # Check collections exist
            for collection_def in self.schema.COLLECTIONS:
                if not self.db.has_collection(collection_def.name):
                    logger.error(f"Missing collection: {collection_def.name}")
                    return False
                    
            # Check critical indexes exist
            critical_indexes = [
                ("documents", ["content_hash"]),
                ("chunks", ["document_id"]),
                ("embeddings", ["chunk_id"]),
                ("nodes", ["node_index"]),
                ("edges", ["_from", "_to"])
            ]
            
            for collection_name, fields in critical_indexes:
                collection = self.db.collection(collection_name)
                indexes = collection.indexes()
                
                index_exists = any(
                    set(idx.get("fields", [])) == set(fields)
                    for idx in indexes
                )
                
                if not index_exists:
                    logger.error(f"Missing critical index on {collection_name}: {fields}")
                    return False
                    
            logger.info("Schema validation passed")
            return True
            
        except ArangoError as e:
            logger.error(f"Schema validation failed: {e}")
            return False
            
    def get_database_status(self) -> DatabaseStatus:
        """
        Get current database status and statistics.
        
        Returns:
            Database status information
        """
        try:
            if not self.db:
                self.db = self.client.db(self.db_name)
            
            # Get collection counts
            collections = {}
            total_docs = 0
            
            for collection_def in self.schema.COLLECTIONS:
                if self.db.has_collection(collection_def.name):
                    collection = self.db.collection(collection_def.name)
                    count = collection.count()
                    collections[collection_def.name] = count
                    total_docs += count
                else:
                    collections[collection_def.name] = 0
            
            # Get index information
            indexes = {}
            for collection_name in collections.keys():
                if self.db.has_collection(collection_name):
                    collection = self.db.collection(collection_name)
                    collection_indexes = collection.indexes()
                    indexes[collection_name] = [
                        idx.get("name", str(idx.get("fields", [])))
                        for idx in collection_indexes
                    ]
                else:
                    indexes[collection_name] = []
            
            # Get schema version
            try:
                models_collection = self.db.collection("models")
                version_doc = models_collection.get("schema_version")
                schema_version = version_doc.get("version", "unknown") if version_doc else "unknown"
            except:
                schema_version = "unknown"
            
            # Get database size (approximation)
            db_size_mb = sum(collections.values()) * 0.001  # Rough estimate
            
            return DatabaseStatus(
                connected=True,
                db_name=self.db_name,
                collections=collections,
                indexes=indexes,
                schema_version=schema_version,
                total_documents=total_docs,
                database_size_mb=db_size_mb,
                last_updated=datetime.now(timezone.utc),
                uptime_seconds=0.0,  # Would need server stats for this
                errors=[],
                warnings=[]
            )
            
        except ArangoError as e:
            logger.error(f"Failed to get database status: {e}")
            return DatabaseStatus(
                connected=False,
                db_name=self.db_name,
                collections={},
                indexes={},
                schema_version="unknown",
                total_documents=0,
                database_size_mb=0.0,
                last_updated=datetime.now(timezone.utc),
                uptime_seconds=0.0,
                errors=[str(e)],
                warnings=[]
            )
            
    def migrate_schema(self, target_version: str) -> bool:
        """
        Migrate schema to target version.
        
        Args:
            target_version: Target schema version
            
        Returns:
            True if migration successful
        """
        # For now, just validate current schema
        # In the future, implement actual migration logic
        logger.info(f"Schema migration to {target_version} not yet implemented")
        return self.validate_schema()
        
    def drop_database(self) -> bool:
        """
        Drop the entire database (use with caution).
        
        Returns:
            True if successful
        """
        try:
            sys_db = self.client.db("_system")
            if sys_db.has_database(self.db_name):
                logger.warning(f"Dropping database: {self.db_name}")
                sys_db.delete_database(self.db_name)
                return True
            return False
        except ArangoError as e:
            logger.error(f"Failed to drop database: {e}")
            return False