"""
Sequential-ISNE Modality-Specific Schema for ArangoDB

This module defines a modality-specific schema architecture optimized for 
Sequential-ISNE's cross-modal discovery capabilities. The schema supports
theory-practice bridge discovery between code, documentation, and configuration files.

Key Features:
- Modality-specific collections (code, documentation, config)
- Cross-modal relationship tracking
- Directory-informed graph bootstrap
- ISNE training optimization
- File-level and chunk-level granularity
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    CollectionCreateError, 
    IndexCreateError, 
    DatabaseCreateError,
    ArangoError
)

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    """File type enumeration for modality classification."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    DATA = "data"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Edge type enumeration for relationship classification."""
    # Intra-modal edges (within same modality)
    CODE_IMPORTS = "code_imports"
    CODE_CALLS = "code_calls"
    CODE_INHERITANCE = "code_inheritance"
    DOC_REFERENCES = "doc_references"
    DOC_HIERARCHY = "doc_hierarchy"
    CONFIG_INCLUDES = "config_includes"
    
    # Cross-modal edges (between different modalities)
    CODE_TO_DOC = "code_to_doc"           # Implementation → Documentation
    DOC_TO_CODE = "doc_to_code"           # Documentation → Implementation
    CODE_TO_CONFIG = "code_to_config"     # Code → Configuration
    CONFIG_TO_CODE = "config_to_code"     # Configuration → Code
    DOC_TO_CONFIG = "doc_to_config"       # Documentation → Configuration
    
    # Directory structure edges
    DIRECTORY_COLOCATION = "directory_colocation"
    DIRECTORY_HIERARCHY = "directory_hierarchy"
    
    # ISNE discovered edges
    ISNE_SIMILARITY = "isne_similarity"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CollectionDef:
    """Definition of a collection."""
    name: str
    edge: bool = False
    indexes: List[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass 
class IndexDef:
    """Definition of an index."""
    collection: str
    fields: List[str]
    type: str = "persistent"
    unique: bool = False
    sparse: bool = False
    name: Optional[str] = None


class SequentialISNESchema:
    """
    Modality-specific schema for Sequential-ISNE with cross-modal discovery.
    
    This schema supports:
    1. File-level nodes organized by modality (code, documentation, config)
    2. Chunk-level processing for text analysis
    3. Cross-modal edge discovery for theory-practice bridges
    4. Directory-informed graph bootstrap
    5. ISNE training optimization
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = "2.0.0-sequential-isne"
    
    # Collection definitions organized by purpose
    COLLECTIONS = [
        # ===== MODALITY-SPECIFIC FILE COLLECTIONS =====
        
        # Code files collection
        CollectionDef(
            name="code_files",
            edge=False,
            description="Source code files (Python, JavaScript, Java, C++, etc.)",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "file_path": {"type": "string"},
                    "file_name": {"type": "string"},
                    "directory": {"type": "string"},
                    "extension": {"type": "string"},
                    "file_type": {"type": "string", "enum": ["python", "javascript", "typescript", "java", "cpp", "c", "rust", "go", "other"]},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "size": {"type": "integer", "minimum": 0},
                    "lines_of_code": {"type": "integer", "minimum": 0},
                    "modified_time": {"type": "string", "format": "date-time"},
                    "ingestion_time": {"type": "string", "format": "date-time"},
                    "processing_status": {"type": "string", "enum": ["pending", "processing", "completed", "error"]},
                    
                    # Code-specific metadata
                    "ast_metadata": {"type": "object"},
                    "imports": {"type": "array", "items": {"type": "string"}},
                    "functions": {"type": "array", "items": {"type": "object"}},
                    "classes": {"type": "array", "items": {"type": "object"}},
                    "complexity_score": {"type": "number", "minimum": 0},
                    
                    # Sequential-ISNE specific
                    "node_id": {"type": "integer"},
                    "embedding_id": {"type": "string"},
                    "chunk_count": {"type": "integer", "minimum": 0},
                    "directory_depth": {"type": "integer", "minimum": 0},
                    
                    "metadata": {"type": "object"}
                },
                "required": ["file_path", "file_name", "directory", "extension", "file_type", "content_hash", "size", "modified_time"]
            }
        ),
        
        # Documentation files collection
        CollectionDef(
            name="documentation_files",
            edge=False,
            description="Documentation files (Markdown, text, PDF, etc.)",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "file_path": {"type": "string"},
                    "file_name": {"type": "string"},
                    "directory": {"type": "string"},
                    "extension": {"type": "string"},
                    "file_type": {"type": "string", "enum": ["markdown", "text", "rst", "pdf", "html", "other"]},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "size": {"type": "integer", "minimum": 0},
                    "word_count": {"type": "integer", "minimum": 0},
                    "modified_time": {"type": "string", "format": "date-time"},
                    "ingestion_time": {"type": "string", "format": "date-time"},
                    "processing_status": {"type": "string", "enum": ["pending", "processing", "completed", "error"]},
                    
                    # Documentation-specific metadata
                    "document_structure": {"type": "object"},
                    "headings": {"type": "array", "items": {"type": "object"}},
                    "links": {"type": "array", "items": {"type": "string"}},
                    "code_references": {"type": "array", "items": {"type": "string"}},
                    "readability_score": {"type": "number", "minimum": 0},
                    
                    # Sequential-ISNE specific
                    "node_id": {"type": "integer"},
                    "embedding_id": {"type": "string"},
                    "chunk_count": {"type": "integer", "minimum": 0},
                    "directory_depth": {"type": "integer", "minimum": 0},
                    
                    "metadata": {"type": "object"}
                },
                "required": ["file_path", "file_name", "directory", "extension", "file_type", "content_hash", "size", "modified_time"]
            }
        ),
        
        # Configuration files collection
        CollectionDef(
            name="config_files",
            edge=False,
            description="Configuration files (JSON, YAML, TOML, etc.)",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "file_path": {"type": "string"},
                    "file_name": {"type": "string"},
                    "directory": {"type": "string"},
                    "extension": {"type": "string"},
                    "file_type": {"type": "string", "enum": ["json", "yaml", "toml", "ini", "xml", "env", "other"]},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "size": {"type": "integer", "minimum": 0},
                    "modified_time": {"type": "string", "format": "date-time"},
                    "ingestion_time": {"type": "string", "format": "date-time"},
                    "processing_status": {"type": "string", "enum": ["pending", "processing", "completed", "error"]},
                    
                    # Config-specific metadata
                    "parsed_config": {"type": "object"},
                    "config_schema": {"type": "object"},
                    "validation_status": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    
                    # Sequential-ISNE specific
                    "node_id": {"type": "integer"},
                    "embedding_id": {"type": "string"},
                    "chunk_count": {"type": "integer", "minimum": 0},
                    "directory_depth": {"type": "integer", "minimum": 0},
                    
                    "metadata": {"type": "object"}
                },
                "required": ["file_path", "file_name", "directory", "extension", "file_type", "content_hash", "size", "modified_time"]
            }
        ),
        
        # ===== CHUNK-LEVEL PROCESSING =====
        
        # Text chunks from all file types
        CollectionDef(
            name="chunks",
            edge=False,
            description="Text chunks extracted from files for processing",
            schema={
                "type": "object", 
                "properties": {
                    "_key": {"type": "string"},
                    "source_file_collection": {"type": "string", "enum": ["code_files", "documentation_files", "config_files"]},
                    "source_file_id": {"type": "string"},
                    "content": {"type": "string"},
                    "content_hash": {"type": "string"},
                    "start_pos": {"type": "integer", "minimum": 0},
                    "end_pos": {"type": "integer", "minimum": 0},
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "chunk_type": {"type": "string", "enum": ["text", "code", "comment", "docstring", "config"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "embedding_id": {"type": "string"},
                    "node_id": {"type": "integer"},
                    "metadata": {"type": "object"}
                },
                "required": ["source_file_collection", "source_file_id", "content", "content_hash", "start_pos", "end_pos", "chunk_index"]
            }
        ),
        
        # ===== EMBEDDINGS =====
        
        # Vector embeddings for all content
        CollectionDef(
            name="embeddings",
            edge=False,
            description="Vector embeddings for files and chunks",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["file", "chunk"]},
                    "source_collection": {"type": "string"},
                    "source_id": {"type": "string"},
                    "embedding_type": {"type": "string", "enum": ["original", "isne_enhanced"]},
                    "vector": {"type": "array", "items": {"type": "number"}},
                    "model_name": {"type": "string"},
                    "model_version": {"type": "string"},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "created_at": {"type": "string", "format": "date-time"},
                    "isne_metadata": {"type": "object"},
                    "metadata": {"type": "object"}
                },
                "required": ["source_type", "source_collection", "source_id", "embedding_type", "vector", "model_name", "embedding_dim"]
            }
        ),
        
        # ===== RELATIONSHIP EDGES =====
        
        # Intra-modal edges (within same modality)
        CollectionDef(
            name="intra_modal_edges",
            edge=True,
            description="Relationships within the same modality (code-to-code, doc-to-doc, etc.)",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "_from": {"type": "string"},
                    "_to": {"type": "string"},
                    "edge_type": {"type": "string", "enum": [
                        "code_imports", "code_calls", "code_inheritance",
                        "doc_references", "doc_hierarchy", 
                        "config_includes", "directory_colocation", "directory_hierarchy"
                    ]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": "string", "enum": ["directory_bootstrap", "ast_analysis", "text_analysis", "manual"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "metadata": {"type": "object"}
                },
                "required": ["_from", "_to", "edge_type", "weight", "source"]
            }
        ),
        
        # Cross-modal edges (between different modalities) - The key Sequential-ISNE innovation!
        CollectionDef(
            name="cross_modal_edges",
            edge=True,
            description="Theory-practice bridges between different modalities",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "_from": {"type": "string"},
                    "_to": {"type": "string"},
                    "_from_modality": {"type": "string", "enum": ["code", "documentation", "config"]},
                    "_to_modality": {"type": "string", "enum": ["code", "documentation", "config"]},
                    "edge_type": {"type": "string", "enum": [
                        "code_to_doc", "doc_to_code", 
                        "code_to_config", "config_to_code",
                        "doc_to_config", "isne_similarity", "semantic_similarity"
                    ]},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "similarity_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": "string", "enum": ["isne_training", "semantic_analysis", "reference_analysis", "manual"]},
                    "discovery_method": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "isne_metadata": {"type": "object"},
                    "metadata": {"type": "object"}
                },
                "required": ["_from", "_to", "_from_modality", "_to_modality", "edge_type", "weight", "source"]
            }
        ),
        
        # ===== ISNE TRAINING AND MODELS =====
        
        # ISNE model versions and training runs
        CollectionDef(
            name="isne_models",
            edge=False,
            description="ISNE model versions and training metadata",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "model_id": {"type": "string"},
                    "version": {"type": "string"},
                    "model_type": {"type": "string", "enum": ["sequential_isne", "baseline_isne"]},
                    "training_config": {"type": "object"},
                    "architecture": {"type": "object"},
                    "performance_metrics": {"type": "object"},
                    "model_path": {"type": "string"},
                    "node_count": {"type": "integer", "minimum": 0},
                    "edge_count": {"type": "integer", "minimum": 0},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "hidden_dim": {"type": "integer", "minimum": 1},
                    "num_layers": {"type": "integer", "minimum": 1},
                    "epochs_trained": {"type": "integer", "minimum": 0},
                    "final_loss": {"type": "number"},
                    "convergence_achieved": {"type": "boolean"},
                    "training_time_seconds": {"type": "number", "minimum": 0},
                    "created_at": {"type": "string", "format": "date-time"},
                    "created_by": {"type": "string"},
                    "is_current": {"type": "boolean"},
                    "metadata": {"type": "object"}
                },
                "required": ["model_id", "version", "model_type", "node_count", "embedding_dim", "model_path"]
            }
        ),
        
        # Directory structure mapping for graph bootstrap
        CollectionDef(
            name="directory_structure",
            edge=False,
            description="Directory hierarchy for graph bootstrap",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "directory_path": {"type": "string"},
                    "parent_directory": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 0},
                    "file_count": {"type": "integer", "minimum": 0},
                    "subdirectory_count": {"type": "integer", "minimum": 0},
                    "total_size": {"type": "integer", "minimum": 0},
                    "created_at": {"type": "string", "format": "date-time"},
                    "metadata": {"type": "object"}
                },
                "required": ["directory_path", "depth"]
            }
        ),
        
        # ===== OPERATIONAL COLLECTIONS =====
        
        # Processing logs and batch operations
        CollectionDef(
            name="processing_logs",
            edge=False,
            description="Logs for batch processing operations",
            schema={
                "type": "object",
                "properties": {
                    "_key": {"type": "string"},
                    "batch_id": {"type": "string"},
                    "operation_type": {"type": "string", "enum": ["bootstrap", "training", "ingestion", "update"]},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "status": {"type": "string", "enum": ["running", "completed", "failed", "cancelled"]},
                    "input_path": {"type": "string"},
                    "files_processed": {"type": "integer", "minimum": 0},
                    "files_added": {"type": "integer", "minimum": 0},
                    "files_updated": {"type": "integer", "minimum": 0},
                    "files_skipped": {"type": "integer", "minimum": 0},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "config": {"type": "object"},
                    "results": {"type": "object"},
                    "metadata": {"type": "object"}
                },
                "required": ["batch_id", "operation_type", "start_time", "status"]
            }
        )
    ]
    
    # Index definitions optimized for Sequential-ISNE operations
    INDEXES = [
        # ===== CODE FILES INDEXES =====
        IndexDef("code_files", ["file_path"], unique=True, name="code_files_path"),
        IndexDef("code_files", ["content_hash"], name="code_files_hash"),
        IndexDef("code_files", ["directory"], name="code_files_directory"),
        IndexDef("code_files", ["file_type"], name="code_files_type"),
        IndexDef("code_files", ["node_id"], unique=True, name="code_files_node_id"),
        IndexDef("code_files", ["processing_status"], name="code_files_status"),
        IndexDef("code_files", ["modified_time"], name="code_files_modified"),
        IndexDef("code_files", ["directory", "file_type"], name="code_files_dir_type"),
        
        # ===== DOCUMENTATION FILES INDEXES =====
        IndexDef("documentation_files", ["file_path"], unique=True, name="doc_files_path"),
        IndexDef("documentation_files", ["content_hash"], name="doc_files_hash"),
        IndexDef("documentation_files", ["directory"], name="doc_files_directory"),
        IndexDef("documentation_files", ["file_type"], name="doc_files_type"),
        IndexDef("documentation_files", ["node_id"], unique=True, name="doc_files_node_id"),
        IndexDef("documentation_files", ["processing_status"], name="doc_files_status"),
        IndexDef("documentation_files", ["modified_time"], name="doc_files_modified"),
        IndexDef("documentation_files", ["directory", "file_type"], name="doc_files_dir_type"),
        
        # ===== CONFIG FILES INDEXES =====
        IndexDef("config_files", ["file_path"], unique=True, name="config_files_path"),
        IndexDef("config_files", ["content_hash"], name="config_files_hash"),
        IndexDef("config_files", ["directory"], name="config_files_directory"),
        IndexDef("config_files", ["file_type"], name="config_files_type"),
        IndexDef("config_files", ["node_id"], unique=True, name="config_files_node_id"),
        IndexDef("config_files", ["processing_status"], name="config_files_status"),
        IndexDef("config_files", ["modified_time"], name="config_files_modified"),
        
        # ===== CHUNKS INDEXES =====
        IndexDef("chunks", ["source_file_collection", "source_file_id"], name="chunks_source"),
        IndexDef("chunks", ["content_hash"], name="chunks_hash"),
        IndexDef("chunks", ["node_id"], unique=True, name="chunks_node_id"),
        IndexDef("chunks", ["embedding_id"], name="chunks_embedding"),
        IndexDef("chunks", ["chunk_type"], name="chunks_type"),
        
        # ===== EMBEDDINGS INDEXES =====
        IndexDef("embeddings", ["source_collection", "source_id"], name="embeddings_source"),
        IndexDef("embeddings", ["embedding_type"], name="embeddings_type"),
        IndexDef("embeddings", ["model_name", "model_version"], name="embeddings_model"),
        IndexDef("embeddings", ["embedding_dim"], name="embeddings_dim"),
        
        # ===== EDGE INDEXES =====
        # Intra-modal edges
        IndexDef("intra_modal_edges", ["_from"], name="intra_edges_from"),
        IndexDef("intra_modal_edges", ["_to"], name="intra_edges_to"),
        IndexDef("intra_modal_edges", ["edge_type"], name="intra_edges_type"),
        IndexDef("intra_modal_edges", ["source"], name="intra_edges_source"),
        IndexDef("intra_modal_edges", ["_from", "_to"], unique=True, name="intra_edges_from_to"),
        IndexDef("intra_modal_edges", ["weight"], name="intra_edges_weight"),
        
        # Cross-modal edges (theory-practice bridges)
        IndexDef("cross_modal_edges", ["_from"], name="cross_edges_from"),
        IndexDef("cross_modal_edges", ["_to"], name="cross_edges_to"),
        IndexDef("cross_modal_edges", ["edge_type"], name="cross_edges_type"),
        IndexDef("cross_modal_edges", ["_from_modality", "_to_modality"], name="cross_edges_modalities"),
        IndexDef("cross_modal_edges", ["source"], name="cross_edges_source"),
        IndexDef("cross_modal_edges", ["similarity_score"], name="cross_edges_similarity"),
        IndexDef("cross_modal_edges", ["_from", "_to"], unique=True, name="cross_edges_from_to"),
        
        # ===== ISNE MODEL INDEXES =====
        IndexDef("isne_models", ["model_id"], unique=True, name="isne_models_id"),
        IndexDef("isne_models", ["version"], name="isne_models_version"),
        IndexDef("isne_models", ["is_current"], name="isne_models_current"),
        IndexDef("isne_models", ["model_type"], name="isne_models_type"),
        IndexDef("isne_models", ["created_at"], name="isne_models_created"),
        
        # ===== DIRECTORY STRUCTURE INDEXES =====
        IndexDef("directory_structure", ["directory_path"], unique=True, name="dir_structure_path"),
        IndexDef("directory_structure", ["parent_directory"], name="dir_structure_parent"),
        IndexDef("directory_structure", ["depth"], name="dir_structure_depth"),
        
        # ===== PROCESSING LOGS INDEXES =====
        IndexDef("processing_logs", ["batch_id"], unique=True, name="logs_batch_id"),
        IndexDef("processing_logs", ["operation_type"], name="logs_operation"),
        IndexDef("processing_logs", ["status"], name="logs_status"),
        IndexDef("processing_logs", ["start_time"], name="logs_start_time")
    ]


class SequentialISNESchemaManager:
    """
    Schema manager for Sequential-ISNE modality-specific collections.
    
    Handles database initialization, collection creation, index management,
    and schema validation for the Sequential-ISNE architecture.
    """
    
    def __init__(self, client: ArangoClient, db_name: str = "sequential_isne"):
        """
        Initialize Sequential-ISNE schema manager.
        
        Args:
            client: ArangoDB client instance
            db_name: Database name for Sequential-ISNE
        """
        self.client = client
        self.db_name = db_name
        self.db: Optional[StandardDatabase] = None
        self.schema = SequentialISNESchema()
        
        logger.info(f"Initialized Sequential-ISNE schema manager for database: {db_name}")
    
    def initialize_database(self) -> bool:
        """
        Initialize database with Sequential-ISNE modality-specific collections.
        
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
                logger.info(f"Creating Sequential-ISNE database: {self.db_name}")
                sys_db.create_database(self.db_name)
            
            # Connect to database
            self.db = self.client.db(self.db_name)
            
            # Create modality-specific collections
            self._create_collections()
            
            # Create optimized indexes
            self._create_indexes()
            
            # Store schema version
            self._store_schema_version()
            
            logger.info(f"Sequential-ISNE database '{self.db_name}' initialized successfully")
            return True
            
        except ArangoError as e:
            logger.error(f"Failed to initialize Sequential-ISNE database: {e}")
            raise
    
    def _create_collections(self) -> None:
        """Create all Sequential-ISNE collections."""
        for collection_def in self.schema.COLLECTIONS:
            if not self.db.has_collection(collection_def.name):
                logger.info(f"Creating collection: {collection_def.name} - {collection_def.description}")
                
                if collection_def.edge:
                    self.db.create_collection(collection_def.name, edge=True)
                else:
                    self.db.create_collection(collection_def.name)
            else:
                logger.debug(f"Collection '{collection_def.name}' already exists")
    
    def _create_indexes(self) -> None:
        """Create all Sequential-ISNE optimized indexes."""
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
                    if index_def.type == "persistent":
                        collection.add_persistent_index(
                            fields=index_def.fields,
                            unique=index_def.unique,
                            sparse=index_def.sparse,
                            name=index_def.name
                        )
                    elif index_def.type == "hash":
                        collection.add_hash_index(
                            fields=index_def.fields,
                            unique=index_def.unique,
                            sparse=index_def.sparse,
                            name=index_def.name
                        )
                    else:
                        # Default to persistent
                        collection.add_persistent_index(
                            fields=index_def.fields,
                            unique=index_def.unique,
                            sparse=index_def.sparse,
                            name=index_def.name
                        )
                except IndexCreateError as e:
                    logger.warning(f"Failed to create index {index_def.name}: {e}")
            else:
                logger.debug(f"Index on {index_def.collection}:{index_def.fields} already exists")
    
    def _store_schema_version(self) -> None:
        """Store Sequential-ISNE schema version in database."""
        collection = self.db.collection("isne_models")
        
        version_doc = {
            "_key": "schema_version",
            "model_id": "schema_info",
            "version": self.schema.SCHEMA_VERSION,
            "model_type": "schema_metadata",
            "created_at": datetime.now().isoformat(),
            "collections": [col.name for col in self.schema.COLLECTIONS],
            "indexes": len(self.schema.INDEXES),
            "modality_support": ["code", "documentation", "config"],
            "features": [
                "cross_modal_discovery",
                "theory_practice_bridges", 
                "directory_bootstrap",
                "isne_training_optimization"
            ],
            "node_count": 0,
            "embedding_dim": 768,  # Default
            "model_path": "schema_metadata",
            "metadata": {
                "schema_type": "modality_specific",
                "architecture": "sequential_isne"
            }
        }
        
        collection.insert(version_doc, overwrite=True)
        logger.info(f"Stored Sequential-ISNE schema version: {self.schema.SCHEMA_VERSION}")
    
    def get_modality_collections(self) -> Dict[str, str]:
        """Get mapping of modalities to their collection names."""
        return {
            "code": "code_files",
            "documentation": "documentation_files", 
            "config": "config_files"
        }
    
    def get_cross_modal_edge_types(self) -> List[str]:
        """Get list of cross-modal edge types for theory-practice bridge discovery."""
        return [
            "code_to_doc", "doc_to_code",
            "code_to_config", "config_to_code", 
            "doc_to_config", "isne_similarity", "semantic_similarity"
        ]
    
    def validate_schema(self) -> bool:
        """
        Validate Sequential-ISNE schema against expected structure.
        
        Returns:
            True if schema is valid
        """
        try:
            if not self.db:
                self.db = self.client.db(self.db_name)
            
            # Check all collections exist
            for collection_def in self.schema.COLLECTIONS:
                if not self.db.has_collection(collection_def.name):
                    logger.error(f"Missing Sequential-ISNE collection: {collection_def.name}")
                    return False
            
            # Check critical indexes for performance
            critical_indexes = [
                ("code_files", ["file_path"]),
                ("documentation_files", ["file_path"]),
                ("config_files", ["file_path"]),
                ("cross_modal_edges", ["_from_modality", "_to_modality"]),
                ("chunks", ["source_file_collection", "source_file_id"]),
                ("embeddings", ["source_collection", "source_id"])
            ]
            
            for collection_name, fields in critical_indexes:
                collection = self.db.collection(collection_name)
                indexes = collection.indexes()
                
                index_exists = any(
                    set(idx.get("fields", [])) == set(fields)
                    for idx in indexes
                )
                
                if not index_exists:
                    logger.error(f"Missing critical Sequential-ISNE index on {collection_name}: {fields}")
                    return False
            
            logger.info("Sequential-ISNE schema validation passed")
            return True
            
        except ArangoError as e:
            logger.error(f"Sequential-ISNE schema validation failed: {e}")
            return False
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get Sequential-ISNE database statistics.
        
        Returns:
            Statistics about the modality-specific collections
        """
        try:
            if not self.db:
                self.db = self.client.db(self.db_name)
            
            stats = {
                "schema_version": self.schema.SCHEMA_VERSION,
                "collections": {},
                "modality_distribution": {},
                "cross_modal_edges": 0,
                "total_files": 0,
                "total_chunks": 0,
                "total_embeddings": 0
            }
            
            # Get collection counts
            modality_collections = self.get_modality_collections()
            for modality, collection_name in modality_collections.items():
                if self.db.has_collection(collection_name):
                    collection = self.db.collection(collection_name)
                    count = collection.count()
                    stats["collections"][collection_name] = count
                    stats["modality_distribution"][modality] = count
                    stats["total_files"] += count
            
            # Get chunks and embeddings counts
            if self.db.has_collection("chunks"):
                stats["collections"]["chunks"] = self.db.collection("chunks").count()
                stats["total_chunks"] = stats["collections"]["chunks"]
            
            if self.db.has_collection("embeddings"):
                stats["collections"]["embeddings"] = self.db.collection("embeddings").count()
                stats["total_embeddings"] = stats["collections"]["embeddings"]
            
            # Get cross-modal edge count
            if self.db.has_collection("cross_modal_edges"):
                stats["collections"]["cross_modal_edges"] = self.db.collection("cross_modal_edges").count()
                stats["cross_modal_edges"] = stats["collections"]["cross_modal_edges"]
            
            return stats
            
        except ArangoError as e:
            logger.error(f"Failed to get Sequential-ISNE database statistics: {e}")
            return {"error": str(e)}