"""
Modular Data Ingestion Pipeline with Component Selection

This module implements a modular version of the data ingestion pipeline
that supports dynamic component selection and configuration.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, TypedDict, cast, Union
from pathlib import Path
from datetime import datetime

from src.orchestration.core.pipeline_config import (
    PipelineConfigLoader, 
    PipelineComponentConfig,
    load_pipeline_config
)
from src.orchestration.core.component_factory import ComponentFactory, get_global_registry
from src.orchestration.pipelines.data_ingestion.stages.base import PipelineStage
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema
from src.alerts.alert_manager import AlertManager, AlertLevel
from src.config.config_loader import load_config

logger = logging.getLogger(__name__)


class ModularDataIngestionPipeline:
    """
    Modular data ingestion pipeline with dynamic component selection.
    
    This pipeline supports:
    - Configuration-driven component selection
    - Dynamic component loading
    - Component fallback handling
    - A/B testing support
    """
    
    def __init__(self,
                 config: Optional[Union[Dict[str, Any], str, Path]] = None,
                 enable_debug: bool = False,
                 debug_output_dir: Optional[Path] = None):
        """
        Initialize the modular data ingestion pipeline.
        
        Args:
            config: Pipeline configuration (dict, config name, or path)
            enable_debug: Whether to save intermediate outputs
            debug_output_dir: Directory for debug outputs
        """
        self.enable_debug = enable_debug
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        
        # Load configuration
        self.pipeline_config = load_pipeline_config(config)
        
        # Initialize component factory
        self.component_factory = ComponentFactory(get_global_registry())
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        
        # Pipeline statistics
        self.stats: Dict[str, Any] = {
            "total_files": 0,
            "processed_documents": 0,
            "generated_chunks": 0,
            "embedded_chunks": 0,
            "isne_enhanced_chunks": 0,
            "stored_documents": 0,
            "stored_chunks": 0,
            "stored_relationships": 0,
            "stage_times": {},
            "errors": [],
            "component_info": {}
        }
        
        # Initialize pipeline stages
        self._initialize_stages()
        
        # Setup debug directory if enabled
        if self.enable_debug and self.debug_output_dir:
            self._setup_debug_directories()
    
    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages with dynamic component loading."""
        try:
            # Validate configuration
            loader = PipelineConfigLoader()
            validation_errors = loader.validate_config(self.pipeline_config)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # Create document processor
            self.docproc_stage = self.component_factory.create_component(
                'document_processor',
                self.pipeline_config.document_processor,
                'document_processor'
            )
            self._record_component_info('document_processor', 
                                      self.pipeline_config.document_processor.implementation)
            
            # Create chunker
            self.chunking_stage = self.component_factory.create_component(
                'chunker',
                self.pipeline_config.chunker,
                'chunking'
            )
            self._record_component_info('chunker', 
                                      self.pipeline_config.chunker.implementation)
            
            # Create embedder (optional)
            if self.pipeline_config.embedder.enabled:
                self.embedding_stage = self.component_factory.create_component(
                    'embedder',
                    self.pipeline_config.embedder,
                    'embedding'
                )
                self._record_component_info('embedder', 
                                          self.pipeline_config.embedder.implementation)
            else:
                self.embedding_stage = None
                logger.info("Embedding stage disabled in configuration")
            
            # Create graph enhancer (optional)
            if self.pipeline_config.graph_enhancer.enabled:
                self.isne_stage = self.component_factory.create_component(
                    'graph_enhancer',
                    self.pipeline_config.graph_enhancer,
                    'isne'
                )
                self._record_component_info('graph_enhancer', 
                                          self.pipeline_config.graph_enhancer.implementation)
            else:
                self.isne_stage = None
                logger.info("Graph enhancement stage disabled in configuration")
            
            # Create storage
            self.storage_stage = self.component_factory.create_component(
                'storage',
                self.pipeline_config.storage,
                'storage'
            )
            self._record_component_info('storage', 
                                      self.pipeline_config.storage.implementation)
            
            logger.info("Initialized all pipeline stages successfully")
            logger.info(f"Component configuration: {self.stats['component_info']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline stages: {e}")
            self.alert_manager.alert(
                message=f"Pipeline initialization failed: {e}",
                level=AlertLevel.HIGH,
                source="modular_data_ingestion_pipeline"
            )
            raise
    
    def _record_component_info(self, component_type: str, implementation: str) -> None:
        """Record component information for statistics."""
        self.stats['component_info'][component_type] = implementation
    
    def _setup_debug_directories(self):
        """Setup debug output directories."""
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        stages = ['docproc', 'chunking', 'embedding', 'isne', 'storage']
        for stage in stages:
            (self.debug_output_dir / stage).mkdir(exist_ok=True)
    
    def process_files(self, 
                     input_files: List[str],
                     filter_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a list of input files through the modular pipeline.
        
        Args:
            input_files: List of file paths to process
            filter_types: Optional list of file types to filter
            
        Returns:
            Dictionary containing pipeline results and statistics
        """
        start_time = time.time()
        
        # Filter files by type if specified
        if filter_types:
            filtered_files = [
                f for f in input_files 
                if any(f.lower().endswith(ext.lower()) for ext in filter_types)
            ]
            logger.info(f"Filtered {len(input_files)} files to {len(filtered_files)} based on types: {filter_types}")
            input_files = filtered_files
        
        self.stats["total_files"] = len(input_files)
        
        if not input_files:
            logger.warning("No files to process after filtering")
            return self._create_results_dict(start_time)
        
        try:
            logger.info(f"Starting modular data ingestion pipeline for {len(input_files)} files")
            
            # Stage 1: Document Processing
            documents = self._run_document_processing(input_files)
            if not documents:
                logger.error("No documents produced from document processing stage")
                return self._create_results_dict(start_time)
            
            # Stage 2: Chunking
            chunks = self._run_chunking(documents)
            if not chunks:
                logger.error("No chunks produced from chunking stage")
                return self._create_results_dict(start_time)
            
            # Stage 3: Embedding Generation (optional)
            if self.embedding_stage:
                embedded_chunks = self._run_embedding(chunks)
                if not embedded_chunks:
                    logger.error("No embedded chunks produced from embedding stage")
                    return self._create_results_dict(start_time)
            else:
                embedded_chunks = chunks
                logger.info("Skipping embedding stage (disabled)")
            
            # Stage 4: Graph Enhancement (optional)
            if self.isne_stage:
                enhanced_chunks = self._run_isne_enhancement(embedded_chunks)
                if not enhanced_chunks:
                    logger.error("No enhanced chunks produced")
                    return self._create_results_dict(start_time)
            else:
                enhanced_chunks = embedded_chunks
                logger.info("Skipping graph enhancement stage (disabled)")
            
            # Stage 5: Storage
            storage_results = self._run_storage(enhanced_chunks)
            
            # Calculate total time
            total_time = time.time() - start_time
            self.stats["total_pipeline_time"] = total_time
            
            logger.info(f"Modular data ingestion pipeline completed in {total_time:.2f} seconds")
            self._log_final_statistics()
            
            return self._create_results_dict(start_time, storage_results)
            
        except Exception as e:
            logger.error(f"Modular data ingestion pipeline failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "pipeline",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.alert_manager.alert(
                message=f"Modular data ingestion pipeline failed: {e}",
                level=AlertLevel.HIGH,
                source="modular_data_ingestion_pipeline"
            )
            return self._create_results_dict(start_time)
    
    def _run_document_processing(self, input_files: List[str]) -> List[DocumentSchema]:
        """Run document processing stage."""
        stage_start = time.time()
        logger.info(f"Stage 1: Document Processing ({self.stats['component_info']['document_processor']}) - {len(input_files)} files")
        
        try:
            documents = self.docproc_stage.run(input_files)
            self.stats["processed_documents"] = len(documents)
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "docproc" / "documents.json"
                self._save_debug_output(documents, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["document_processing"] = stage_time
            
            logger.info(f"Document processing completed: {len(documents)} documents in {stage_time:.2f}s")
            return documents
            
        except Exception as e:
            logger.error(f"Document processing stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "document_processing",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return []
    
    def _run_chunking(self, documents: List[DocumentSchema]) -> List[Dict[str, Any]]:
        """Run chunking stage."""
        stage_start = time.time()
        logger.info(f"Stage 2: Chunking ({self.stats['component_info']['chunker']}) - {len(documents)} documents")
        
        try:
            chunk_schemas = self.chunking_stage.run(documents)
            
            # Convert ChunkSchema objects to dictionaries
            chunks = []
            for chunk in chunk_schemas:
                chunk_dict = {
                    'id': chunk.id,
                    'text': chunk.text,
                    'source_document': getattr(chunk, 'source_document', ''),
                    'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
                chunks.append(chunk_dict)
            
            self.stats["generated_chunks"] = len(chunks)
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "chunking" / "chunks.json"
                self._save_debug_output(chunks, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["chunking"] = stage_time
            
            logger.info(f"Chunking completed: {len(chunks)} chunks in {stage_time:.2f}s")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "chunking",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return []
    
    def _run_embedding(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run embedding generation stage."""
        stage_start = time.time()
        logger.info(f"Stage 3: Embedding Generation ({self.stats['component_info']['embedder']}) - {len(chunks)} chunks")
        
        try:
            # Convert chunks to document format for embedding stage
            doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks:
                    doc_chunks[source_doc] = []
                doc_chunks[source_doc].append(chunk)
            
            # Create DocumentSchema objects
            documents_for_embedding = []
            for source_doc, chunk_list in doc_chunks.items():
                doc_schema = DocumentSchema(
                    file_id=f"embedding_{hash(source_doc) % 10000}",
                    file_name=source_doc,
                    file_path=f"temp/{source_doc}",
                    file_type="pdf",
                    content_type="application/pdf",
                    chunks=[
                        ChunkSchema(
                            id=chunk['id'],
                            text=chunk['text'],
                            metadata=chunk.get('metadata', {})
                        )
                        for chunk in chunk_list
                    ]
                )
                documents_for_embedding.append(doc_schema)
            
            # Run embedding stage
            embedded_documents = self.embedding_stage.run(documents_for_embedding)
            
            # Convert back to flat chunk list
            embedded_chunks: List[Dict[str, Any]] = []
            for doc in embedded_documents:
                for chunk_schema in doc.chunks:
                    chunk_dict = {
                        'id': chunk_schema.id,
                        'text': chunk_schema.text,
                        'source_document': doc.file_name,
                        'embedding': chunk_schema.embedding,
                        'metadata': chunk_schema.metadata if hasattr(chunk_schema, 'metadata') else {},
                        'embedding_metadata': {
                            'model': self.stats['component_info']['embedder'],
                            'dimension': len(chunk_schema.embedding) if chunk_schema.embedding else 0,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    embedded_chunks.append(chunk_dict)
            
            self.stats["embedded_chunks"] = len(embedded_chunks)
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "embedding" / "embedded_chunks.json"
                self._save_debug_output(embedded_chunks, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["embedding"] = stage_time
            
            logger.info(f"Embedding generation completed: {len(embedded_chunks)} chunks in {stage_time:.2f}s")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Embedding stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "embedding",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return chunks  # Return original chunks without embeddings
    
    def _run_isne_enhancement(self, embedded_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run graph enhancement stage."""
        stage_start = time.time()
        logger.info(f"Stage 4: Graph Enhancement ({self.stats['component_info']['graph_enhancer']}) - {len(embedded_chunks)} chunks")
        
        try:
            # Convert chunks to document format
            doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in embedded_chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks:
                    doc_chunks[source_doc] = []
                doc_chunks[source_doc].append(chunk)
            
            # Create DocumentSchema objects
            documents_for_isne = []
            for source_doc, chunk_list in doc_chunks.items():
                doc_schema = DocumentSchema(
                    file_id=f"isne_{hash(source_doc) % 10000}",
                    file_name=source_doc,
                    file_path=f"temp/{source_doc}",
                    file_type="pdf",
                    content_type="application/pdf",
                    chunks=[
                        ChunkSchema(
                            id=chunk['id'],
                            text=chunk['text'],
                            embedding=chunk.get('embedding'),
                            metadata=chunk.get('metadata', {})
                        )
                        for chunk in chunk_list
                    ]
                )
                documents_for_isne.append(doc_schema)
            
            # Run graph enhancement stage
            enhanced_documents = self.isne_stage.run(documents_for_isne)
            
            # Convert back to flat chunk list
            enhanced_chunks: List[Dict[str, Any]] = []
            for doc in enhanced_documents:
                for chunk_schema in doc.chunks:
                    chunk_dict: Dict[str, Any] = {
                        'id': chunk_schema.id,
                        'text': chunk_schema.text,
                        'source_document': doc.file_name,
                        'embedding': chunk_schema.embedding,
                        'isne_embedding': getattr(chunk_schema, 'isne_embedding', None),
                        'metadata': chunk_schema.metadata if hasattr(chunk_schema, 'metadata') else {},
                        'embedding_metadata': {
                            'model': self.stats['component_info']['embedder'],
                            'dimension': len(chunk_schema.embedding) if chunk_schema.embedding else 0,
                            'timestamp': datetime.now().isoformat()
                        },
                        'graph_metadata': {
                            'enhancer': self.stats['component_info']['graph_enhancer'],
                            'enhanced_at': datetime.now().isoformat()
                        },
                        'relationships': getattr(chunk_schema, 'relationships', {})
                    }
                    enhanced_chunks.append(chunk_dict)
            
            self.stats["isne_enhanced_chunks"] = len(enhanced_chunks)
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "isne" / "enhanced_chunks.json"
                self._save_debug_output(enhanced_chunks, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["graph_enhancement"] = stage_time
            
            logger.info(f"Graph enhancement completed: {len(enhanced_chunks)} chunks in {stage_time:.2f}s")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Graph enhancement stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "graph_enhancement",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return embedded_chunks  # Return chunks without enhancement
    
    def _run_storage(self, final_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run storage stage."""
        stage_start = time.time()
        logger.info(f"Stage 5: Storage ({self.stats['component_info']['storage']}) - {len(final_chunks)} chunks")
        
        try:
            # Convert chunks to document format for storage
            doc_chunks_for_storage: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in final_chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks_for_storage:
                    doc_chunks_for_storage[source_doc] = []
                doc_chunks_for_storage[source_doc].append(chunk)
            
            # Create DocumentSchema objects
            documents_for_storage = []
            for source_doc, chunk_list in doc_chunks_for_storage.items():
                chunks = []
                for chunk_data in chunk_list:
                    chunk_schema = ChunkSchema(
                        id=chunk_data['id'],
                        text=chunk_data['text'],
                        embedding=chunk_data.get('embedding'),
                        metadata=chunk_data.get('metadata', {})
                    )
                    if 'isne_embedding' in chunk_data:
                        chunk_schema.isne_embedding = chunk_data['isne_embedding']
                    chunks.append(chunk_schema)
                
                doc_schema = DocumentSchema(
                    file_id=f"storage_{hash(source_doc) % 10000}",
                    file_name=source_doc,
                    file_path=f"processed/{source_doc}",
                    file_type="pdf",
                    content_type="application/pdf",
                    chunks=chunks
                )
                documents_for_storage.append(doc_schema)
            
            # Run storage stage
            storage_results = self.storage_stage.run(documents_for_storage)
            
            # Update statistics
            self.stats["stored_documents"] = storage_results.stored_documents
            self.stats["stored_chunks"] = storage_results.stored_chunks
            self.stats["stored_relationships"] = storage_results.stored_relationships
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "storage" / "storage_results.json"
                self._save_debug_output(storage_results, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["storage"] = stage_time
            
            logger.info(f"Storage completed in {stage_time:.2f}s")
            return {
                'stored_documents': storage_results.stored_documents,
                'stored_chunks': storage_results.stored_chunks,
                'stored_relationships': storage_results.stored_relationships,
                'operation_mode': storage_results.operation_mode,
                'database_name': storage_results.database_name,
                'execution_time': storage_results.execution_time
            }
            
        except Exception as e:
            logger.error(f"Storage stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "storage",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {}
    
    def _save_debug_output(self, data: Any, filepath: Path) -> None:
        """Save debug output to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved debug output: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug output to {filepath}: {e}")
    
    def _log_final_statistics(self) -> None:
        """Log final pipeline statistics."""
        logger.info("=== Modular Data Ingestion Pipeline Statistics ===")
        logger.info(f"Component Configuration: {self.stats['component_info']}")
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Documents processed: {self.stats['processed_documents']}")
        logger.info(f"Chunks generated: {self.stats['generated_chunks']}")
        logger.info(f"Chunks embedded: {self.stats['embedded_chunks']}")
        logger.info(f"Chunks enhanced: {self.stats['isne_enhanced_chunks']}")
        logger.info(f"Documents stored: {self.stats['stored_documents']}")
        logger.info(f"Chunks stored: {self.stats['stored_chunks']}")
        logger.info(f"Relationships stored: {self.stats['stored_relationships']}")
        logger.info(f"Total pipeline time: {self.stats.get('total_pipeline_time', 0):.2f}s")
        
        if self.stats['errors']:
            logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                logger.warning(f"  - {error['stage']}: {error['message']}")
    
    def _create_results_dict(self, 
                           start_time: float, 
                           storage_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create final results dictionary."""
        total_time = time.time() - start_time
        
        return {
            'pipeline_stats': self.stats,
            'component_configuration': self.stats['component_info'],
            'stage_times': self.stats.get('stage_times', {}),
            'total_time': total_time,
            'storage_results': storage_results or {},
            'success': len(self.stats['errors']) == 0,
            'timestamp': datetime.now().isoformat(),
            'debug_enabled': self.enable_debug,
            'debug_output_dir': str(self.debug_output_dir) if self.debug_output_dir else None
        }


def run_modular_ingestion_pipeline(
    input_files: List[str],
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    enable_debug: bool = False,
    debug_output_dir: Optional[str] = None,
    filter_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the modular data ingestion pipeline.
    
    Args:
        input_files: List of file paths to process
        config: Pipeline configuration (dict, name, or path)
        enable_debug: Whether to save intermediate outputs
        debug_output_dir: Directory for debug outputs
        filter_types: Optional file type filter
        
    Returns:
        Dictionary containing pipeline results and statistics
    """
    pipeline = ModularDataIngestionPipeline(
        config=config,
        enable_debug=enable_debug,
        debug_output_dir=Path(debug_output_dir) if debug_output_dir else None
    )
    
    return pipeline.process_files(input_files, filter_types)