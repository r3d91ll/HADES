"""
Storage stage for the HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for storing processed documents and their relationships
in a database (ArangoDB).
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Set
import time
from datetime import datetime
import os
import json

from src.orchestration.pipelines.stages.base import PipelineStage, PipelineStageError
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema, ValidationResult, ValidationIssue, ValidationSeverity, StorageResult
# Import ArangoClient directly from the module file
from src.database.arango_client import ArangoClient as ArangoClient
from src.config.database_config import get_database_config

logger = logging.getLogger(__name__)


class StorageMode:
    """Storage operation modes."""
    CREATE = "create"  # Create new collections/database
    APPEND = "append"  # Append to existing collections/database
    UPSERT = "upsert"  # Update existing or insert new documents


class StorageStage(PipelineStage):
    """Pipeline stage for database storage.
    
    This stage handles storing processed documents, chunks, and relationships
    in the database (ArangoDB). It supports different modes of operation:
    - CREATE: Create a new database and collections
    - APPEND: Add new documents to an existing database
    - UPSERT: Update existing documents or insert new ones
    """
    
    def __init__(
        self,
        name: str = "storage",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize storage stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with storage options
        """
        super().__init__(name, config or {})
        
        # Configure storage parameters
        self.database_name = self.config.get("database_name", "hades")
        self.mode = self.config.get("mode", StorageMode.CREATE)
        self.documents_collection = self.config.get("documents_collection", "documents")
        self.chunks_collection = self.config.get("chunks_collection", "chunks")
        self.relationships_collection = self.config.get("relationships_collection", "relationships") 
        self.overwrite = self.config.get("overwrite", False)
        self.batch_size = self.config.get("batch_size", 100)
        
        # Initialize database client
        # If db_config is provided in the stage config, use it; otherwise load from config system
        db_config = self.config.get("db_config")
        if db_config is None:
            # Load database configuration from the config system
            db_config = get_database_config()
            self.logger.info("Using database configuration from config system")
        
        try:
            # Initialize client with appropriate config
            self.db_client = ArangoClient(**db_config)
            self.logger.info(f"Initialized database client for {self.database_name}")
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to initialize database client: {str(e)}",
                original_error=e
            )
    
    def run(self, input_data: List[DocumentSchema]) -> StorageResult:
        """Store documents in the database.
        
        This method stores processed documents, chunks, and relationships
        in the database according to the configured storage mode.
        
        Args:
            input_data: List of DocumentSchema objects to store
            
        Returns:
            StorageResult with storage statistics
            
        Raises:
            PipelineStageError: If storage operation fails
        """
        if not input_data:
            raise PipelineStageError(
                self.name,
                "No documents provided for storage"
            )
        
        start_time = time.time()
        
        try:
            # Initialize database and collections based on mode
            self._initialize_storage()
            
            # Process documents based on mode
            if self.mode == StorageMode.CREATE:
                result = self._create_new_datastore(input_data)
            elif self.mode == StorageMode.APPEND:
                result = self._append_to_datastore(input_data)
            elif self.mode == StorageMode.UPSERT:
                result = self._upsert_datastore(input_data)
            else:
                raise PipelineStageError(
                    self.name,
                    f"Unknown storage mode: {self.mode}"
                )
            
            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create storage result
            storage_result = StorageResult(
                stored_documents=result["documents"],
                stored_chunks=result["chunks"],
                stored_relationships=result["relationships"],
                operation_mode=self.mode,
                database_name=self.database_name,
                collections=result["collections"],
                execution_time=execution_time,
                errors=result.get("errors", [])
            )
            
            self.logger.info(
                f"Storage completed in {execution_time:.2f}s: "
                f"{storage_result.stored_documents} documents, "
                f"{storage_result.stored_chunks} chunks, "
                f"{storage_result.stored_relationships} relationships"
            )
            
            return storage_result
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Storage operation failed: {str(e)}",
                original_error=e
            )
    
    def validate(self, data: Union[List[DocumentSchema], StorageResult, Any]) -> Tuple[bool, List[str]]:
        """Validate input or output data for this stage.
        
        Args:
            data: Data to validate (either input DocumentSchema objects or output StorageResult)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # For input validation (DocumentSchema objects)
        if isinstance(data, list):
            if not all(isinstance(doc, DocumentSchema) for doc in data):
                return False, ["All items in the list must be DocumentSchema objects"]
            
            # Check that documents have chunks and embeddings
            if not all(hasattr(doc, 'chunks') and doc.chunks for doc in data):
                return False, ["One or more documents have no chunks"]
            
            if not all(
                all(chunk.embedding is not None for chunk in doc.chunks)
                for doc in data if doc.chunks
            ):
                return False, ["One or more chunks have no embeddings"]
            
            return True, []
        
        # For output validation (StorageResult)
        elif isinstance(data, StorageResult):
            # Check for errors
            if data.errors:
                error_messages = [f"Storage error: {error.get('message', 'Unknown error')}" 
                                 for error in data.errors]
                return False, error_messages
            
            # Check that documents were stored
            if data.stored_documents == 0:
                return False, ["No documents were stored in the database"]
            
            return True, []
        
        # Invalid data type
        return False, [f"Invalid data type: {type(data)}. Expected List[DocumentSchema] or StorageResult"]
    
    def _initialize_storage(self) -> None:
        """Initialize database and collections based on storage mode.
        
        Raises:
            PipelineStageError: If initialization fails
        """
        try:
            # Check if database exists
            database_exists = self.db_client.database_exists(self.database_name)
            
            # Handle different modes
            if self.mode == StorageMode.CREATE:
                if database_exists and not self.overwrite:
                    raise PipelineStageError(
                        self.name,
                        f"Database '{self.database_name}' already exists and overwrite=False"
                    )
                
                if database_exists and self.overwrite:
                    self.logger.warning(f"Dropping existing database '{self.database_name}'")
                    self.db_client.delete_database(self.database_name)
                
                # Create new database
                self.logger.info(f"Creating new database '{self.database_name}'")
                self.db_client.create_database(self.database_name)
                
                # Create collections
                self._create_collections()
                
            elif self.mode in [StorageMode.APPEND, StorageMode.UPSERT]:
                if not database_exists:
                    raise PipelineStageError(
                        self.name,
                        f"Database '{self.database_name}' does not exist for {self.mode} mode"
                    )
                
                # Verify collections exist
                self._verify_collections()
            
            self.logger.info(f"Storage initialized in {self.mode} mode")
            
        except Exception as e:
            if not isinstance(e, PipelineStageError):
                raise PipelineStageError(
                    self.name,
                    f"Failed to initialize storage: {str(e)}",
                    original_error=e
                )
            raise
    
    def _create_collections(self) -> None:
        """Create database collections.
        
        Raises:
            PipelineStageError: If collection creation fails
        """
        try:
            # Create document collection
            self.db_client.create_collection(
                self.database_name,
                self.documents_collection,
                edge=False
            )
            
            # Create chunk collection
            self.db_client.create_collection(
                self.database_name,
                self.chunks_collection,
                edge=False
            )
            
            # Create relationship collection (edge collection)
            self.db_client.create_collection(
                self.database_name,
                self.relationships_collection,
                edge=True
            )
            
            # Create indexes for faster lookup
            self.db_client.create_index(
                self.database_name,
                self.documents_collection,
                index_type="hash",
                fields=["file_id"],
                unique=True
            )
            
            self.db_client.create_index(
                self.database_name,
                self.chunks_collection,
                index_type="hash",
                fields=["id"],
                unique=True
            )
            
            self.logger.info("Created collections and indexes")
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to create collections: {str(e)}",
                original_error=e
            )
    
    def _verify_collections(self) -> None:
        """Verify that required collections exist.
        
        Raises:
            PipelineStageError: If collections verification fails
        """
        try:
            collections = self.db_client.get_collections(self.database_name)
            collection_names = set(col["name"] for col in collections if isinstance(col, dict))
            
            required_collections = {
                self.documents_collection,
                self.chunks_collection,
                self.relationships_collection
            }
            
            missing_collections = required_collections - collection_names
            
            if missing_collections:
                raise PipelineStageError(
                    self.name,
                    f"Missing required collections: {', '.join(missing_collections)}"
                )
            
            self.logger.info("Verified collections exist")
            
        except Exception as e:
            if not isinstance(e, PipelineStageError):
                raise PipelineStageError(
                    self.name,
                    f"Failed to verify collections: {str(e)}",
                    original_error=e
                )
            raise
    
    def _create_new_datastore(self, documents: List[DocumentSchema]) -> Dict[str, Any]:
        """Create a new datastore with the provided documents.
        
        Args:
            documents: List of documents to store
            
        Returns:
            Dictionary with storage statistics
            
        Raises:
            PipelineStageError: If storage operation fails
        """
        result: Dict[str, Any] = {
            "documents": 0,
            "chunks": 0,
            "relationships": 0,
            "collections": {},
            "errors": []
        }
        
        # Track relationship edges to avoid duplicates
        processed_relationships = set()
        
        try:
            # Process each document
            for document in documents:
                try:
                    # Store document
                    doc_data = document.dict(exclude={"chunks", "validation"})
                    doc_data["_key"] = document.file_id
                    
                    self.db_client.insert_document(
                        self.database_name,
                        self.documents_collection,
                        doc_data
                    )
                    result["documents"] += 1
                    
                    # Store chunks
                    for chunk in document.chunks:
                        # Prepare chunk data
                        chunk_data = chunk.dict(exclude={"relationships", "validation"})
                        chunk_data["_key"] = chunk.id
                        chunk_data["document_id"] = document.file_id
                        
                        self.db_client.insert_document(
                            self.database_name,
                            self.chunks_collection,
                            chunk_data
                        )
                        result["chunks"] += 1
                        
                        # Store relationships
                        for relationship in chunk.relationships:
                            # Create a unique key for the relationship to avoid duplicates
                            rel_key = f"{relationship.source}_{relationship.target}_{relationship.type}"
                            
                            if rel_key in processed_relationships:
                                continue
                            
                            processed_relationships.add(rel_key)
                            
                            # Prepare relationship data
                            rel_data = relationship.dict()
                            rel_data["_from"] = f"{self.chunks_collection}/{relationship.source}"
                            rel_data["_to"] = f"{self.chunks_collection}/{relationship.target}"
                            
                            self.db_client.insert_edge(
                                self.database_name,
                                self.relationships_collection,
                                from_vertex=f"{self.chunks_collection}/{relationship.source}",
                                to_vertex=f"{self.chunks_collection}/{relationship.target}",
                                data=rel_data
                            )
                            result["relationships"] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Error storing document {document.file_name}: {str(e)}",
                        exc_info=True
                    )
                    result["errors"].append({
                        "document_id": document.file_id,
                        "message": str(e),
                        "type": "document_storage_error"
                    })
            
            # Get collection counts
            collections = self.db_client.get_collections(self.database_name)
            for collection in collections:
                count = self.db_client.get_collection_count(
                    self.database_name,
                    collection["name"]
                )
                result["collections"][collection["name"]] = count
            
            return result
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to create datastore: {str(e)}",
                original_error=e
            )
    
    def _append_to_datastore(self, documents: List[DocumentSchema]) -> Dict[str, Any]:
        """Append documents to an existing datastore.
        
        Args:
            documents: List of documents to append
            
        Returns:
            Dictionary with storage statistics
            
        Raises:
            PipelineStageError: If storage operation fails
        """
        result: Dict[str, Any] = {
            "documents": 0,
            "chunks": 0,
            "relationships": 0,
            "collections": {},
            "errors": []
        }
        
        # Track relationship edges to avoid duplicates
        processed_relationships = set()
        
        # Get existing document IDs to avoid conflicts
        existing_document_ids = self._get_existing_document_ids()
        
        try:
            # Process each document
            for document in documents:
                try:
                    # Skip if document already exists
                    if document.file_id in existing_document_ids:
                        self.logger.warning(
                            f"Document with ID {document.file_id} already exists, skipping"
                        )
                        result["errors"].append({
                            "document_id": document.file_id,
                            "message": "Document already exists",
                            "type": "document_exists"
                        })
                        continue
                    
                    # Store document
                    doc_data = document.dict(exclude={"chunks", "validation"})
                    doc_data["_key"] = document.file_id
                    
                    self.db_client.insert_document(
                        self.database_name,
                        self.documents_collection,
                        doc_data
                    )
                    result["documents"] += 1
                    
                    # Store chunks
                    for chunk in document.chunks:
                        # Prepare chunk data
                        chunk_data = chunk.dict(exclude={"relationships", "validation"})
                        chunk_data["_key"] = chunk.id
                        chunk_data["document_id"] = document.file_id
                        
                        self.db_client.insert_document(
                            self.database_name,
                            self.chunks_collection,
                            chunk_data
                        )
                        result["chunks"] += 1
                        
                        # Store relationships
                        for relationship in chunk.relationships:
                            # Create a unique key for the relationship to avoid duplicates
                            rel_key = f"{relationship.source}_{relationship.target}_{relationship.type}"
                            
                            if rel_key in processed_relationships:
                                continue
                            
                            processed_relationships.add(rel_key)
                            
                            # Prepare relationship data
                            rel_data = relationship.dict()
                            rel_data["_from"] = f"{self.chunks_collection}/{relationship.source}"
                            rel_data["_to"] = f"{self.chunks_collection}/{relationship.target}"
                            
                            # Check if target chunk exists (might be from an existing document)
                            target_exists = self.db_client.document_exists(
                                self.database_name,
                                self.chunks_collection,
                                relationship.target
                            )
                            
                            if target_exists:
                                self.db_client.insert_edge(
                                    self.database_name,
                                    self.relationships_collection,
                                    from_vertex=f"{self.chunks_collection}/{relationship.source}",
                                    to_vertex=f"{self.chunks_collection}/{relationship.target}",
                                    data=rel_data
                                )
                                result["relationships"] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Error storing document {document.file_name}: {str(e)}",
                        exc_info=True
                    )
                    result["errors"].append({
                        "document_id": document.file_id,
                        "message": str(e),
                        "type": "document_storage_error"
                    })
            
            # Get collection counts
            collections = self.db_client.get_collections(self.database_name)
            for collection in collections:
                count = self.db_client.get_collection_count(
                    self.database_name,
                    collection["name"]
                )
                result["collections"][collection["name"]] = count
            
            return result
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to append to datastore: {str(e)}",
                original_error=e
            )
    
    def _upsert_datastore(self, documents: List[DocumentSchema]) -> Dict[str, Any]:
        """Update existing documents or insert new ones.
        
        Args:
            documents: List of documents to upsert
            
        Returns:
            Dictionary with storage statistics
            
        Raises:
            PipelineStageError: If storage operation fails
        """
        result: Dict[str, Any] = {
            "documents": 0,
            "chunks": 0,
            "relationships": 0,
            "updated_documents": 0,
            "updated_chunks": 0,
            "collections": {},
            "errors": []
        }
        
        # Track relationship edges to avoid duplicates
        processed_relationships = set()
        
        # Get existing document IDs
        existing_document_ids = self._get_existing_document_ids()
        
        try:
            # Process each document
            for document in documents:
                try:
                    # Check if document exists
                    is_update = document.file_id in existing_document_ids
                    
                    # Update or insert document
                    doc_data = document.dict(exclude={"chunks", "validation"})
                    doc_data["_key"] = document.file_id
                    
                    if is_update:
                        # Update existing document
                        self.db_client.update_document(
                            self.database_name,
                            self.documents_collection,
                            document.file_id,
                            doc_data
                        )
                        result["updated_documents"] += 1
                        
                        # Remove existing chunks for this document
                        self._remove_document_chunks(document.file_id)
                    else:
                        # Insert new document
                        self.db_client.insert_document(
                            self.database_name,
                            self.documents_collection,
                            doc_data
                        )
                        result["documents"] += 1
                    
                    # Store chunks
                    for chunk in document.chunks:
                        # Prepare chunk data
                        chunk_data = chunk.dict(exclude={"relationships", "validation"})
                        chunk_data["_key"] = chunk.id
                        chunk_data["document_id"] = document.file_id
                        
                        # Try update first, then insert if not exists
                        if self.db_client.document_exists(
                            self.database_name,
                            self.chunks_collection,
                            chunk.id
                        ):
                            self.db_client.update_document(
                                self.database_name,
                                self.chunks_collection,
                                chunk.id,
                                chunk_data
                            )
                            result["updated_chunks"] += 1
                        else:
                            self.db_client.insert_document(
                                self.database_name,
                                self.chunks_collection,
                                chunk_data
                            )
                            result["chunks"] += 1
                        
                        # Store relationships
                        for relationship in chunk.relationships:
                            # Create a unique key for the relationship to avoid duplicates
                            rel_key = f"{relationship.source}_{relationship.target}_{relationship.type}"
                            
                            if rel_key in processed_relationships:
                                continue
                            
                            processed_relationships.add(rel_key)
                            
                            # Prepare relationship data
                            rel_data = relationship.dict()
                            rel_data["_from"] = f"{self.chunks_collection}/{relationship.source}"
                            rel_data["_to"] = f"{self.chunks_collection}/{relationship.target}"
                            
                            # Check if target chunk exists
                            target_exists = self.db_client.document_exists(
                                self.database_name,
                                self.chunks_collection,
                                relationship.target
                            )
                            
                            if target_exists:
                                # Try to find existing edge
                                existing_edge = self.db_client.get_edge(
                                    self.database_name,
                                    self.relationships_collection,
                                    rel_data["_from"],
                                    rel_data["_to"]
                                )
                                
                                if existing_edge:
                                    # Update existing edge
                                    self.db_client.update_edge(
                                        self.database_name,
                                        self.relationships_collection,
                                        existing_edge["_id"],
                                        rel_data
                                    )
                                else:
                                    # Insert new edge
                                    self.db_client.insert_edge(
                                        self.database_name,
                                        self.relationships_collection,
                                        from_vertex=rel_data.get("_from", ""),
                                        to_vertex=rel_data.get("_to", ""),
                                        data=rel_data
                                    )
                                
                                result["relationships"] += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Error upserting document {document.file_name}: {str(e)}",
                        exc_info=True
                    )
                    result["errors"].append({
                        "document_id": document.file_id,
                        "message": str(e),
                        "type": "document_upsert_error"
                    })
            
            # Get collection counts
            collections = self.db_client.get_collections(self.database_name)
            for collection in collections:
                count = self.db_client.get_collection_count(
                    self.database_name,
                    collection["name"]
                )
                result["collections"][collection["name"]] = count
            
            return result
            
        except Exception as e:
            raise PipelineStageError(
                self.name,
                f"Failed to upsert datastore: {str(e)}",
                original_error=e
            )
    
    def _get_existing_document_ids(self) -> Set[str]:
        """Get set of existing document IDs.
        
        Returns:
            Set of document IDs
        """
        try:
            query = f"FOR doc IN {self.documents_collection} RETURN doc.file_id"
            result = self.db_client.execute_query(
                self.database_name,
                query
            )
            # Extract file_ids from results and convert to set of strings
            file_ids = [doc.get('file_id', '') for doc in result if isinstance(doc, dict)]
            return set(file_ids)
        except Exception as e:
            self.logger.warning(f"Failed to get existing document IDs: {str(e)}")
            return set()
    
    def _remove_document_chunks(self, document_id: str) -> None:
        """Remove chunks for a document.
        
        Args:
            document_id: Document ID to remove chunks for
        """
        try:
            # Find chunks for document
            query = f"""
            FOR chunk IN {self.chunks_collection}
                FILTER chunk.document_id == @document_id
                RETURN chunk._key
            """
            chunk_results = self.db_client.execute_query(
                self.database_name,
                query,
                bind_vars={"document_id": document_id}
            )
            
            # Extract chunk keys from results
            chunk_keys = []
            for result in chunk_results:
                if isinstance(result, dict) and "_key" in result:
                    chunk_keys.append(str(result["_key"]))
                # Note: We only handle dict results as ArangoDB always returns objects
            
            # Remove relationships for these chunks
            for chunk_key in chunk_keys:
                # Remove outgoing relationships
                self.db_client.delete_edges(
                    self.database_name,
                    self.relationships_collection,
                    f"{self.chunks_collection}/{chunk_key}",
                    direction="outbound"
                )
                
                # Remove incoming relationships
                self.db_client.delete_edges(
                    self.database_name,
                    self.relationships_collection,
                    f"{self.chunks_collection}/{chunk_key}",
                    direction="inbound"
                )
                
                # Delete the chunk document
                self.db_client.delete_document(
                    self.database_name,
                    self.chunks_collection,
                    chunk_key
                )
            
            self.logger.info(f"Removed {len(chunk_keys)} chunks for document {document_id}")
            
        except Exception as e:
            self.logger.warning(f"Error removing chunks for document {document_id}: {str(e)}")
            # Continue with upsert operation
    
