"""
Incremental Manager for HADES ArangoDB Storage

This module orchestrates incremental document ingestion and ISNE model updates,
providing the main interface for incremental operations without full reprocessing.
"""

import asyncio
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import json

from arango import ArangoClient
from arango.database import StandardDatabase

from .types import (
    IncrementalConfig, 
    IngestionResult, 
    DocumentInfo, 
    DocumentState,
    ConflictStrategy,
    ModelUpdateResult,
    VersionInfo
)
from .schema import SchemaManager
from .graph_builder import GraphBuilder
from .model_updater import ModelUpdater  
from .conflict_resolver import ConflictResolver
from src.alerts import AlertManager, AlertLevel
from src.isne.models.isne_model import ISNEModel

logger = logging.getLogger(__name__)


class IncrementalManager:
    """
    Main orchestrator for incremental storage operations.
    
    Coordinates document processing, graph building, and model updates
    to enable efficient incremental ingestion without full reprocessing.
    """
    
    def __init__(
        self,
        config: Optional[IncrementalConfig] = None,
        client: Optional[ArangoClient] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Initialize incremental manager.
        
        Args:
            config: Incremental storage configuration
            client: ArangoDB client (created if not provided)
            alert_manager: Alert manager for notifications
        """
        self.config = config or IncrementalConfig()
        self.alert_manager = alert_manager or AlertManager()
        
        # Initialize ArangoDB client
        if client:
            self.client = client
        else:
            self.client = ArangoClient(hosts='http://localhost:8529')
        
        # Initialize components
        self.schema_manager = SchemaManager(self.client, self.config)
        self.db: Optional[StandardDatabase] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.model_updater: Optional[ModelUpdater] = None
        self.conflict_resolver: Optional[ConflictResolver] = None
        
        # State tracking
        self.initialized = False
        self.current_model_path: Optional[str] = None
        self.current_model_version: Optional[str] = None
        
    async def initialize(self) -> bool:
        """
        Initialize the incremental storage system.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing incremental storage system...")
            
            # Initialize database schema
            if not self.schema_manager.initialize_database():
                logger.error("Failed to initialize database schema")
                return False
            
            # Connect to database
            self.db = self.client.db(self.config.db_name)
            
            # Initialize components
            self.graph_builder = GraphBuilder(self.db, self.config)
            self.model_updater = ModelUpdater(self.db, self.config, self.alert_manager)
            self.conflict_resolver = ConflictResolver(self.db, self.config)
            
            # Load current model information
            await self._load_current_model_info()
            
            self.initialized = True
            logger.info("Incremental storage system initialized successfully")
            
            # Send initialization alert
            self.alert_manager.alert(
                message="Incremental storage system initialized",
                level=AlertLevel.LOW,
                source="incremental_manager",
                context={"db_name": self.config.db_name}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize incremental storage: {e}")
            self.alert_manager.alert(
                message=f"Incremental storage initialization failed: {e}",
                level=AlertLevel.HIGH,
                source="incremental_manager",
                context={"error": str(e)}
            )
            return False
    
    async def ingest_documents(
        self,
        input_dir: str,
        conflict_strategy: Optional[ConflictStrategy] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> IngestionResult:
        """
        Ingest documents incrementally.
        
        Args:
            input_dir: Directory containing documents to ingest
            conflict_strategy: Strategy for handling conflicts
            batch_size: Batch size for processing
            **kwargs: Additional processing options
            
        Returns:
            Ingestion results with statistics and details
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use configuration defaults if not specified
        conflict_strategy = conflict_strategy or self.config.conflict_strategy
        batch_size = batch_size or self.config.batch_size
        
        logger.info(f"Starting incremental ingestion: {input_dir}")
        logger.info(f"Batch ID: {batch_id}")
        logger.info(f"Conflict strategy: {conflict_strategy}")
        
        try:
            # Create ingestion log entry
            await self._create_ingestion_log(batch_id, input_dir, conflict_strategy)
            
            # Discover and analyze documents
            document_files = self._discover_documents(input_dir)
            logger.info(f"Found {len(document_files)} documents to process")
            
            # Analyze existing documents
            existing_docs = await self._get_existing_documents()
            
            # Classify documents (new, updated, unchanged)
            document_analysis = await self._analyze_documents(document_files, existing_docs)
            
            # Process documents in batches
            processing_results = await self._process_documents_batch(
                document_analysis, 
                conflict_strategy,
                batch_size,
                batch_id
            )
            
            # Build graph for new/updated chunks
            if processing_results['new_chunks'] or processing_results['updated_chunks']:
                graph_results = await self._build_graph_incremental(
                    processing_results['new_chunks'],
                    processing_results['updated_chunks']
                )
                processing_results.update(graph_results)
            
            # Update ISNE model if needed
            model_results = None
            if processing_results.get('new_nodes', 0) > 0:
                model_results = await self._update_model_incremental(
                    processing_results['new_nodes'],
                    batch_id
                )
            
            # Compile final results
            processing_time = time.time() - start_time
            result = self._compile_ingestion_result(
                processing_results, 
                model_results,
                processing_time,
                document_analysis
            )
            
            # Update ingestion log
            await self._update_ingestion_log(batch_id, result)
            
            # Send completion alert
            self.alert_manager.alert(
                message=f"Incremental ingestion completed: {result.new_documents} new, {result.updated_documents} updated",
                level=AlertLevel.LOW,
                source="incremental_manager",
                context={
                    "batch_id": batch_id,
                    "processing_time": processing_time,
                    "total_documents": result.total_documents
                }
            )
            
            logger.info(f"Incremental ingestion completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Incremental ingestion failed: {e}")
            
            # Update ingestion log with error
            await self._update_ingestion_log(batch_id, None, error=str(e))
            
            # Send error alert
            self.alert_manager.alert(
                message=f"Incremental ingestion failed: {e}",
                level=AlertLevel.HIGH,
                source="incremental_manager",
                context={
                    "batch_id": batch_id,
                    "input_dir": input_dir,
                    "error": str(e)
                }
            )
            
            raise
    
    def _discover_documents(self, input_dir: str) -> List[Path]:
        """
        Discover all documents in input directory.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of document file paths
        """
        input_path = Path(input_dir)
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.txt', '.md', '.py', '.json', '.yaml', '.yml', '.html'}
        
        documents = []
        
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)
        
        logger.info(f"Discovered {len(documents)} documents with supported extensions")
        return documents
    
    async def _get_existing_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Get existing documents from database.
        
        Returns:
            Dictionary mapping file paths to document info
        """
        collection = self.db.collection('documents')
        
        query = """
        FOR doc IN documents
        RETURN {
            file_path: doc.file_path,
            content_hash: doc.content_hash,
            modified_time: doc.modified_time,
            _key: doc._key
        }
        """
        
        results = self.db.query(query)
        
        existing_docs = {}
        for doc in results:
            existing_docs[doc['file_path']] = doc
        
        logger.info(f"Found {len(existing_docs)} existing documents in database")
        return existing_docs
    
    async def _analyze_documents(
        self, 
        document_files: List[Path], 
        existing_docs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[DocumentInfo]]:
        """
        Analyze documents to classify them as new, updated, or unchanged.
        
        Args:
            document_files: List of document file paths
            existing_docs: Existing documents from database
            
        Returns:
            Dictionary with categorized document info
        """
        new_docs = []
        updated_docs = []
        unchanged_docs = []
        error_docs = []
        
        for file_path in document_files:
            try:
                # Calculate current file hash and metadata
                file_stats = file_path.stat()
                content_hash = self._calculate_file_hash(file_path)
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                
                doc_info = DocumentInfo(
                    file_path=str(file_path),
                    content_hash=content_hash,
                    size=file_stats.st_size,
                    modified_time=modified_time,
                    state=DocumentState.NEW
                )
                
                # Check if document exists
                file_path_str = str(file_path)
                if file_path_str in existing_docs:
                    existing_doc = existing_docs[file_path_str]
                    
                    if existing_doc['content_hash'] == content_hash:
                        doc_info.state = DocumentState.UNCHANGED
                        unchanged_docs.append(doc_info)
                    else:
                        doc_info.state = DocumentState.UPDATED
                        updated_docs.append(doc_info)
                else:
                    new_docs.append(doc_info)
                    
            except Exception as e:
                error_doc = DocumentInfo(
                    file_path=str(file_path),
                    content_hash="",
                    size=0,
                    modified_time=datetime.now(),
                    state=DocumentState.ERROR,
                    error_message=str(e)
                )
                error_docs.append(error_doc)
                logger.warning(f"Failed to analyze document {file_path}: {e}")
        
        logger.info(f"Document analysis: {len(new_docs)} new, {len(updated_docs)} updated, "
                   f"{len(unchanged_docs)} unchanged, {len(error_docs)} errors")
        
        return {
            'new': new_docs,
            'updated': updated_docs,
            'unchanged': unchanged_docs,
            'error': error_docs
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _process_documents_batch(
        self,
        document_analysis: Dict[str, List[DocumentInfo]],
        conflict_strategy: ConflictStrategy,
        batch_size: int,
        batch_id: str
    ) -> Dict[str, Any]:
        """
        Process documents in batches.
        
        Args:
            document_analysis: Categorized document analysis
            conflict_strategy: Conflict resolution strategy
            batch_size: Processing batch size
            batch_id: Batch identifier
            
        Returns:
            Processing results with statistics
        """
        # For now, implement basic processing logic
        # This would integrate with existing document processing components
        
        results = {
            'processed_documents': 0,
            'new_chunks': 0,
            'updated_chunks': 0,
            'new_embeddings': 0,
            'new_nodes': 0,
            'processing_errors': []
        }
        
        # Process new documents
        for doc_info in document_analysis['new']:
            try:
                # This would call document processing pipeline
                # For now, just increment counters
                results['processed_documents'] += 1
                results['new_chunks'] += 10  # Placeholder
                results['new_embeddings'] += 10
                results['new_nodes'] += 10
                
            except Exception as e:
                results['processing_errors'].append({
                    'document': doc_info.file_path,
                    'error': str(e)
                })
        
        # Process updated documents
        for doc_info in document_analysis['updated']:
            try:
                # Handle conflicts
                resolution = await self.conflict_resolver.resolve_conflict(
                    doc_info, conflict_strategy
                )
                
                if resolution.strategy_used != ConflictStrategy.SKIP:
                    results['processed_documents'] += 1
                    results['updated_chunks'] += 5  # Placeholder
                    results['new_embeddings'] += 5
                    
            except Exception as e:
                results['processing_errors'].append({
                    'document': doc_info.file_path,
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed: {results['processed_documents']} documents processed")
        return results
    
    async def _build_graph_incremental(
        self, 
        new_chunks: int, 
        updated_chunks: int
    ) -> Dict[str, Any]:
        """
        Build graph incrementally for new/updated chunks.
        
        Args:
            new_chunks: Number of new chunks
            updated_chunks: Number of updated chunks
            
        Returns:
            Graph building results
        """
        logger.info(f"Building graph incrementally: {new_chunks} new chunks, {updated_chunks} updated chunks")
        
        # This would call GraphBuilder to construct edges
        graph_results = await self.graph_builder.build_incremental(new_chunks, updated_chunks)
        
        return {
            'new_edges': graph_results.get('new_edges', 0),
            'updated_edges': graph_results.get('updated_edges', 0),
            'total_edges': graph_results.get('total_edges', 0)
        }
    
    async def _update_model_incremental(
        self, 
        new_nodes: int, 
        batch_id: str
    ) -> Optional[ModelUpdateResult]:
        """
        Update ISNE model incrementally.
        
        Args:
            new_nodes: Number of new nodes added
            batch_id: Batch identifier
            
        Returns:
            Model update results if update performed
        """
        if new_nodes == 0:
            return None
        
        logger.info(f"Updating ISNE model with {new_nodes} new nodes")
        
        # This would call ModelUpdater to expand and retrain model
        update_result = await self.model_updater.update_incremental(
            new_nodes, batch_id
        )
        
        return update_result
    
    def _compile_ingestion_result(
        self,
        processing_results: Dict[str, Any],
        model_results: Optional[ModelUpdateResult],
        processing_time: float,
        document_analysis: Dict[str, List[DocumentInfo]]
    ) -> IngestionResult:
        """
        Compile final ingestion results.
        
        Args:
            processing_results: Document processing results
            model_results: Model update results (if any)
            processing_time: Total processing time
            document_analysis: Document analysis results
            
        Returns:
            Complete ingestion result
        """
        total_docs = sum(len(docs) for docs in document_analysis.values())
        
        return IngestionResult(
            total_documents=total_docs,
            new_documents=len(document_analysis['new']),
            updated_documents=len(document_analysis['updated']),
            unchanged_documents=len(document_analysis['unchanged']),
            error_documents=len(document_analysis['error']),
            
            total_chunks=processing_results.get('new_chunks', 0) + processing_results.get('updated_chunks', 0),
            new_chunks=processing_results.get('new_chunks', 0),
            total_embeddings=processing_results.get('new_embeddings', 0),
            
            total_nodes=processing_results.get('new_nodes', 0),
            new_nodes=processing_results.get('new_nodes', 0),
            total_edges=processing_results.get('total_edges', 0),
            new_edges=processing_results.get('new_edges', 0),
            
            previous_model_size=model_results.previous_node_count if model_results else 0,
            new_model_size=model_results.new_node_count if model_results else 0,
            model_expanded=model_results is not None and model_results.success,
            
            processing_time_seconds=processing_time,
            documents_per_second=total_docs / processing_time if processing_time > 0 else 0,
            
            document_results=document_analysis['new'] + document_analysis['updated'] + 
                           document_analysis['unchanged'] + document_analysis['error'],
            conflicts=[],
            errors=processing_results.get('processing_errors', [])
        )
    
    async def _create_ingestion_log(
        self,
        batch_id: str,
        input_path: str,
        conflict_strategy: ConflictStrategy
    ) -> None:
        """Create ingestion log entry."""
        collection = self.db.collection('ingestion_logs')
        
        log_entry = {
            '_key': batch_id,
            'batch_id': batch_id,
            'start_time': datetime.now().isoformat(),
            'input_path': input_path,
            'status': 'running',
            'config': {
                'conflict_strategy': conflict_strategy.value,
                'batch_size': self.config.batch_size
            },
            'total_documents': 0,
            'processed_documents': 0,
            'metadata': {}
        }
        
        collection.insert(log_entry)
        logger.debug(f"Created ingestion log: {batch_id}")
    
    async def _update_ingestion_log(
        self,
        batch_id: str,
        result: Optional[IngestionResult],
        error: Optional[str] = None
    ) -> None:
        """Update ingestion log with results."""
        collection = self.db.collection('ingestion_logs')
        
        update_data = {
            'end_time': datetime.now().isoformat()
        }
        
        if error:
            update_data.update({
                'status': 'failed',
                'errors': [error]
            })
        elif result:
            update_data.update({
                'status': 'completed',
                'total_documents': result.total_documents,
                'processed_documents': result.new_documents + result.updated_documents,
                'new_documents': result.new_documents,
                'updated_documents': result.updated_documents,
                'error_documents': result.error_documents,
                'results': result.model_dump()
            })
        
        collection.update({'_key': batch_id}, update_data)
        logger.debug(f"Updated ingestion log: {batch_id}")
    
    async def _load_current_model_info(self) -> None:
        """Load information about current model."""
        try:
            collection = self.db.collection('models')
            
            query = """
            FOR model IN models
            FILTER model.is_current == true
            LIMIT 1
            RETURN model
            """
            
            results = list(self.db.query(query))
            
            if results:
                current_model = results[0]
                self.current_model_path = current_model.get('model_path')
                self.current_model_version = current_model.get('version_id')
                logger.info(f"Current model: {self.current_model_version} at {self.current_model_path}")
            else:
                logger.info("No current model found in database")
                
        except Exception as e:
            logger.warning(f"Failed to load current model info: {e}")
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            collection = self.db.collection('ingestion_logs')
            
            query = """
            FOR log IN ingestion_logs
            COLLECT status = log.status WITH COUNT INTO count
            RETURN { status: status, count: count }
            """
            
            status_counts = {}
            for result in self.db.query(query):
                status_counts[result['status']] = result['count']
            
            # Get recent ingestion activity
            query = """
            FOR log IN ingestion_logs
            SORT log.start_time DESC
            LIMIT 10
            RETURN {
                batch_id: log.batch_id,
                start_time: log.start_time,
                status: log.status,
                total_documents: log.total_documents,
                processed_documents: log.processed_documents
            }
            """
            
            recent_activity = list(self.db.query(query))
            
            return {
                'status_counts': status_counts,
                'recent_activity': recent_activity,
                'current_model_version': self.current_model_version,
                'database_status': self.schema_manager.get_database_status().model_dump()
            }
            
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {
                'error': str(e),
                'status_counts': {},
                'recent_activity': [],
                'current_model_version': None
            }