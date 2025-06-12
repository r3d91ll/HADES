"""
Complete Data Ingestion Pipeline for HADES

This module implements the full data ingestion pipeline that processes documents
from raw files through to storage in ArangoDB with embeddings and relationships.

Pipeline Stages:
1. Document Processing (DocProc) - Extract content from files
2. Chunking - Segment documents into semantic chunks  
3. Embedding - Generate base embeddings using ModernBERT/vLLM
4. ISNE Enhancement - Apply graph-based embedding enhancement
5. Storage - Persist to ArangoDB with relationships

Based on JSON schema analysis and testing with 2 PDF corpus.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, TypedDict, cast
from pathlib import Path
from datetime import datetime

from src.orchestration.pipelines.data_ingestion.stages.document_processor import DocumentProcessorStage
from src.orchestration.pipelines.data_ingestion.stages.chunking import ChunkingStage
from src.orchestration.pipelines.data_ingestion.stages.embedding import EmbeddingStage
from src.orchestration.pipelines.data_ingestion.stages.isne import ISNEStage
# Note: Using enhanced storage stage that handles dict input
from src.orchestration.pipelines.data_ingestion.stages.storage import StorageStage
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema
from src.alerts.alert_manager import AlertManager, AlertLevel
from src.config.config_loader import load_config

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Complete data ingestion pipeline for HADES.
    
    Processes documents through all stages from raw files to database storage
    with comprehensive monitoring, validation, and error handling.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 enable_debug: bool = False,
                 debug_output_dir: Optional[Path] = None):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            config: Pipeline configuration dictionary
            enable_debug: Whether to save intermediate outputs at each stage
            debug_output_dir: Directory for debug outputs (if enabled)
        """
        self.config = config or {}
        self.enable_debug = enable_debug
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        
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
            "errors": []
        }
        
        # Initialize pipeline stages
        self._initialize_stages()
        
        # Setup debug directory if enabled
        if self.enable_debug and self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_output_dir / "docproc").mkdir(exist_ok=True)
            (self.debug_output_dir / "chunking").mkdir(exist_ok=True)
            (self.debug_output_dir / "embedding").mkdir(exist_ok=True)
            (self.debug_output_dir / "isne").mkdir(exist_ok=True)
            (self.debug_output_dir / "storage").mkdir(exist_ok=True)
    
    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages with appropriate configurations."""
        try:
            # Load stage-specific configurations
            docproc_config = self.config.get("docproc", {})
            chunking_config = self.config.get("chunking", {})
            embedding_config = self.config.get("embedding", {})
            isne_config = self.config.get("isne", {})
            storage_config = self.config.get("storage", {})
            
            # Initialize stages
            self.docproc_stage = DocumentProcessorStage(
                name="document_processor",
                config=docproc_config
            )
            
            self.chunking_stage = ChunkingStage(
                name="chunking", 
                config=chunking_config
            )
            
            self.embedding_stage = EmbeddingStage(
                name="embedding",
                config=embedding_config
            )
            
            self.isne_stage = ISNEStage(
                name="isne",
                config=isne_config
            )
            
            self.storage_stage = StorageStage(
                name="storage",
                config=storage_config
            )
            
            logger.info("Initialized all pipeline stages successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline stages: {e}")
            self.alert_manager.alert(
                message=f"Pipeline initialization failed: {e}",
                level=AlertLevel.HIGH,
                source="data_ingestion_pipeline"
            )
            raise
    
    def process_files(self, 
                     input_files: List[str],
                     filter_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a list of input files through the complete pipeline.
        
        Args:
            input_files: List of file paths to process
            filter_types: Optional list of file types to filter (e.g., ['.pdf', '.py'])
            
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
            logger.info(f"Starting data ingestion pipeline for {len(input_files)} files")
            
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
            
            # Stage 3: Embedding Generation
            embedded_chunks = self._run_embedding(chunks)
            if not embedded_chunks:
                logger.error("No embedded chunks produced from embedding stage")
                return self._create_results_dict(start_time)
            
            # Stage 4: ISNE Enhancement
            isne_enhanced_chunks = self._run_isne_enhancement(embedded_chunks)
            if not isne_enhanced_chunks:
                logger.error("No ISNE enhanced chunks produced")
                return self._create_results_dict(start_time)
            
            # Stage 5: Storage
            storage_results = self._run_storage(isne_enhanced_chunks)
            
            # Calculate total time
            total_time = time.time() - start_time
            self.stats["total_pipeline_time"] = total_time
            
            logger.info(f"Data ingestion pipeline completed in {total_time:.2f} seconds")
            self._log_final_statistics()
            
            return self._create_results_dict(start_time, storage_results)
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "pipeline",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.alert_manager.alert(
                message=f"Data ingestion pipeline failed: {e}",
                level=AlertLevel.HIGH,
                source="data_ingestion_pipeline"
            )
            return self._create_results_dict(start_time)
    
    def _run_document_processing(self, input_files: List[str]) -> List[DocumentSchema]:
        """Run document processing stage."""
        stage_start = time.time()
        logger.info(f"Stage 1: Document Processing - {len(input_files)} files")
        
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
        logger.info(f"Stage 2: Chunking - {len(documents)} documents")
        
        try:
            # The chunking stage returns a flat list of ChunkSchema objects
            chunk_schemas = self.chunking_stage.run(documents)
            
            # Convert ChunkSchema objects to dictionaries for easier processing
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
        logger.info(f"Stage 3: Embedding Generation - {len(chunks)} chunks")
        
        try:
            # Convert chunks back to DocumentSchema format for embedding stage
            # Group chunks by source document
            doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks:
                    doc_chunks[source_doc] = []
                doc_chunks[source_doc].append(chunk)
            
            # Create DocumentSchema objects for embedding stage
            documents_for_embedding = []
            for source_doc, chunk_list in doc_chunks.items():
                # Create a minimal DocumentSchema with chunks
                doc_schema = DocumentSchema(
                    file_id=f"embedding_{hash(source_doc) % 10000}",
                    file_name=source_doc,
                    file_path=f"temp/{source_doc}",
                    file_type="pdf",  # Simplified assumption
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
            
            # Convert back to flat chunk list with embeddings
            embedded_chunks: List[Dict[str, Any]] = []
            for doc in embedded_documents:
                for chunk_schema in doc.chunks:
                    # Convert ChunkSchema object to dict
                    chunk_dict = {
                        'id': chunk_schema.id,
                        'text': chunk_schema.text,
                        'source_document': doc.file_name,
                        'embedding': chunk_schema.embedding,
                        'metadata': chunk_schema.metadata if hasattr(chunk_schema, 'metadata') else {},
                        'embedding_metadata': {
                            'model': 'modernbert',  # From config
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
            return []
    
    def _run_isne_enhancement(self, embedded_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run ISNE enhancement stage."""
        stage_start = time.time()
        logger.info(f"Stage 4: ISNE Enhancement - {len(embedded_chunks)} chunks")
        
        try:
            # Convert chunks back to DocumentSchema format for ISNE stage
            # Similar to embedding stage conversion
            doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in embedded_chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks:
                    doc_chunks[source_doc] = []
                doc_chunks[source_doc].append(chunk)
            
            # Create DocumentSchema objects for ISNE stage
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
            
            # Run ISNE stage
            isne_enhanced_documents = self.isne_stage.run(documents_for_isne)
            
            # Convert back to flat chunk list with ISNE embeddings
            isne_enhanced_chunks: List[Dict[str, Any]] = []
            for doc in isne_enhanced_documents:
                for chunk_schema in doc.chunks:
                    # Convert ChunkSchema object to dict with ISNE enhancements
                    chunk_dict: Dict[str, Any] = {
                        'id': chunk_schema.id,
                        'text': chunk_schema.text,
                        'source_document': doc.file_name,
                        'embedding': chunk_schema.embedding,
                        'isne_embedding': getattr(chunk_schema, 'isne_embedding', None),
                        'metadata': chunk_schema.metadata if hasattr(chunk_schema, 'metadata') else {},
                        'embedding_metadata': {
                            'model': 'modernbert',
                            'dimension': len(chunk_schema.embedding) if chunk_schema.embedding else 0,
                            'timestamp': datetime.now().isoformat()
                        },
                        'isne_metadata': {
                            'model_version': 'v1',
                            'training_timestamp': datetime.now().isoformat(),
                            'enhancement_quality': 0.85  # Default quality score
                        },
                        'graph_relationships': getattr(chunk_schema, 'graph_relationships', {}),
                        'relationships': {
                            'similar_chunks': [],
                            'sequential_chunks': [],
                            'cross_document_links': []
                        }
                    }
                    isne_enhanced_chunks.append(chunk_dict)
            
            self.stats["isne_enhanced_chunks"] = len(isne_enhanced_chunks)
            
            # Save debug output
            if self.enable_debug and self.debug_output_dir:
                debug_file = self.debug_output_dir / "isne" / "isne_enhanced_chunks.json"
                self._save_debug_output(isne_enhanced_chunks, debug_file)
            
            stage_time = time.time() - stage_start
            cast(Dict[str, float], self.stats["stage_times"])["isne"] = stage_time
            
            logger.info(f"ISNE enhancement completed: {len(isne_enhanced_chunks)} chunks in {stage_time:.2f}s")
            return isne_enhanced_chunks
            
        except Exception as e:
            logger.error(f"ISNE stage failed: {e}")
            cast(List[Dict[str, Any]], self.stats["errors"]).append({
                "stage": "isne",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            # Return chunks without ISNE enhancement
            for chunk in embedded_chunks:
                chunk['isne_embedding'] = None
                chunk['isne_metadata'] = {}
                chunk['graph_relationships'] = {}
                chunk['relationships'] = {}
            return embedded_chunks
    
    def _run_storage(self, final_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run storage stage."""
        stage_start = time.time()
        logger.info(f"Stage 5: Storage - {len(final_chunks)} chunks")
        
        try:
            # Convert chunks back to DocumentSchema format for storage stage
            doc_chunks_for_storage: Dict[str, List[Dict[str, Any]]] = {}
            for chunk in final_chunks:
                source_doc = chunk.get('source_document', 'unknown')
                if source_doc not in doc_chunks_for_storage:
                    doc_chunks_for_storage[source_doc] = []
                doc_chunks_for_storage[source_doc].append(chunk)
            
            # Create DocumentSchema objects for storage
            documents_for_storage = []
            for source_doc, chunk_list in doc_chunks_for_storage.items():
                # Create ChunkSchema objects
                chunks = []
                for chunk_data in chunk_list:
                    chunk_schema = ChunkSchema(
                        id=chunk_data['id'],
                        text=chunk_data['text'],
                        embedding=chunk_data.get('embedding'),
                        metadata=chunk_data.get('metadata', {})
                    )
                    # Add ISNE-specific fields if available
                    if 'isne_embedding' in chunk_data:
                        chunk_schema.isne_embedding = chunk_data['isne_embedding']
                    chunks.append(chunk_schema)
                
                # Create DocumentSchema
                doc_schema = DocumentSchema(
                    file_id=f"storage_{hash(source_doc) % 10000}",
                    file_name=source_doc,
                    file_path=f"processed/{source_doc}",
                    file_type="pdf",  # Simplified assumption
                    content_type="application/pdf",
                    chunks=chunks
                )
                documents_for_storage.append(doc_schema)
            
            # Run storage stage with properly formatted documents
            storage_results = self.storage_stage.run(documents_for_storage)
            
            # Update statistics from storage results
            # StorageResult has stored_documents, stored_chunks, stored_relationships attributes
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
            # Convert StorageResult to dict for consistency with other stages
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
        logger.info("=== Data Ingestion Pipeline Statistics ===")
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Documents processed: {self.stats['processed_documents']}")
        logger.info(f"Chunks generated: {self.stats['generated_chunks']}")
        logger.info(f"Chunks embedded: {self.stats['embedded_chunks']}")
        logger.info(f"Chunks ISNE enhanced: {self.stats['isne_enhanced_chunks']}")
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
            'stage_times': self.stats.get('stage_times', {}),
            'total_time': total_time,
            'storage_results': storage_results or {},
            'success': len(self.stats['errors']) == 0,
            'timestamp': datetime.now().isoformat(),
            'debug_enabled': self.enable_debug,
            'debug_output_dir': str(self.debug_output_dir) if self.debug_output_dir else None
        }


def run_data_ingestion_pipeline(
    input_files: List[str],
    config: Optional[Dict[str, Any]] = None,
    enable_debug: bool = False,
    debug_output_dir: Optional[str] = None,
    filter_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the complete data ingestion pipeline.
    
    Args:
        input_files: List of file paths to process
        config: Pipeline configuration dictionary
        enable_debug: Whether to save intermediate outputs
        debug_output_dir: Directory for debug outputs
        filter_types: Optional file type filter (e.g., ['.pdf', '.py'])
        
    Returns:
        Dictionary containing pipeline results and statistics
    """
    # Load default configuration if not provided
    if config is None:
        try:
            # First try to load the complete data ingestion config
            data_ingestion_config = load_config('data_ingestion_config')
            logger.info("Loaded data_ingestion_config.yaml")
            
            # Extract stage-specific configs
            config = {
                'docproc': data_ingestion_config.get('docproc', {}),
                'chunking': data_ingestion_config.get('chunking', {}),
                'embedding': data_ingestion_config.get('embedding', {}),
                'isne': data_ingestion_config.get('isne', {}),
                'storage': data_ingestion_config.get('storage', {})
            }
            
        except Exception as e:
            logger.warning(f"Failed to load data_ingestion_config: {e}, trying individual configs")
            
            # Fallback to individual config files
            try:
                config = {
                    'docproc': load_config('docproc'),
                    'chunking': load_config('chunking'), 
                    'embedding': load_config('embedding'),
                    'isne': load_config('isne'),
                    'storage': load_config('storage')
                }
            except Exception as e2:
                logger.warning(f"Failed to load individual configs: {e2}")
                config = {}
    
    # Initialize and run pipeline
    pipeline = DataIngestionPipeline(
        config=config,
        enable_debug=enable_debug,
        debug_output_dir=Path(debug_output_dir) if debug_output_dir else None
    )
    
    return pipeline.process_files(input_files, filter_types)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HADES data ingestion pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory containing input files")
    parser.add_argument("--output-dir", help="Output directory for debug files")
    parser.add_argument("--file-types", nargs="+", default=[".pdf"], help="File types to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Path to custom config file")
    
    args = parser.parse_args()
    
    # Find input files
    input_dir = Path(args.input_dir)
    input_files: List[str] = []
    for file_type in args.file_types:
        input_files.extend(str(f) for f in input_dir.glob(f"*{file_type}"))
    
    # Load custom config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # Run pipeline
    results = run_data_ingestion_pipeline(
        input_files=input_files,
        config=config,
        enable_debug=args.debug,
        debug_output_dir=args.output_dir,
        filter_types=args.file_types
    )
    
    print(f"Pipeline completed: {results['success']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Files processed: {results['pipeline_stats']['total_files']}")
    print(f"Chunks generated: {results['pipeline_stats']['generated_chunks']}")
    print(f"Chunks stored: {results['pipeline_stats']['stored_chunks']}")