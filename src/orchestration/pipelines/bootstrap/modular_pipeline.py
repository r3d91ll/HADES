"""
Modular Bootstrap Pipeline for HADES

This module implements a modular bootstrap pipeline that reuses data ingestion
components through the component factory system.
"""

import logging
import time
import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from src.orchestration.core.pipeline_config import (
    PipelineConfigLoader,
    PipelineComponentConfig,
    load_pipeline_config
)
from src.orchestration.core.component_factory import ComponentFactory, get_global_registry
from src.orchestration.pipelines.schema import DocumentSchema, ChunkSchema
from src.orchestration.pipelines.data_ingestion.stages.base import PipelineStage
from src.isne.training.trainer import ISNETrainer
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
from src.alerts.alert_manager import AlertManager, AlertLevel
from src.config.config_loader import load_config

logger = logging.getLogger(__name__)


class ModularBootstrapPipeline:
    """
    Modular bootstrap pipeline for initializing HADES components.
    
    This pipeline:
    - Reuses data ingestion components (document processing, chunking, embedding)
    - Handles the chicken-and-egg problem of needing a graph to train ISNE
    - Supports component swapping through configuration
    - Creates initial graph structures for training
    """
    
    def __init__(self,
                 config: Optional[Union[Dict[str, Any], str, Path]] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize modular bootstrap pipeline.
        
        Args:
            config: Pipeline configuration (dict, config name, or path)
            output_dir: Directory for outputs and models
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./bootstrap_output")
        
        # Load configuration
        self.pipeline_config = load_pipeline_config(config)
        
        # Initialize component factory
        self.component_factory = ComponentFactory(get_global_registry())
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        
        # Bootstrap-specific parameters
        self.similarity_threshold = 0.3
        self.max_connections_per_chunk = 10
        self.min_cluster_size = 5
        
        # Initialize shared components
        self._initialize_components()
        
        # Setup output directories
        self._setup_output_directories()
        
        # Training components (initialized during bootstrap)
        self.isne_trainer: Optional[ISNETrainer] = None
        self.training_orchestrator: Optional[ISNETrainingOrchestrator] = None
    
    def _initialize_components(self) -> None:
        """Initialize reusable data ingestion components."""
        try:
            # Validate configuration
            loader = PipelineConfigLoader()
            validation_errors = loader.validate_config(self.pipeline_config)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # Create shared components
            self.doc_processor = self.component_factory.create_component(
                'document_processor',
                self.pipeline_config.document_processor,
                'bootstrap_doc_processor'
            )
            
            self.chunker = self.component_factory.create_component(
                'chunker',
                self.pipeline_config.chunker,
                'bootstrap_chunker'
            )
            
            # Embedding is optional but recommended for bootstrap
            if self.pipeline_config.embedder.enabled:
                self.embedder: Optional[PipelineStage] = self.component_factory.create_component(
                    'embedder',
                    self.pipeline_config.embedder,
                    'bootstrap_embedder'
                )
            else:
                self.embedder: Optional[PipelineStage] = None
                logger.warning("Embedder disabled - will use TF-IDF for initial embeddings")
            
            logger.info("Initialized bootstrap pipeline components")
            
        except Exception as e:
            logger.error(f"Failed to initialize bootstrap components: {e}")
            self.alert_manager.alert(
                message=f"Bootstrap pipeline initialization failed: {e}",
                level=AlertLevel.HIGH,
                source="modular_bootstrap_pipeline"
            )
            raise
    
    def _setup_output_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)
    
    def bootstrap_from_corpus(self, 
                            corpus_dir: Path,
                            file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        Execute complete bootstrap process from document corpus.
        
        Args:
            corpus_dir: Directory containing documents
            file_pattern: File pattern to match (default: *.pdf)
            
        Returns:
            Dictionary containing bootstrap results and metrics
        """
        logger.info(f"Starting modular bootstrap from corpus: {corpus_dir}")
        start_time = datetime.now(timezone.utc)
        
        # Find input files
        input_files = list(corpus_dir.glob(file_pattern))
        if not input_files:
            raise ValueError(f"No files matching {file_pattern} found in {corpus_dir}")
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Phase 1: Process documents using shared components
        logger.info("Phase 1: Processing documents")
        all_chunks = self._process_documents(input_files)
        
        if len(all_chunks) < 10:
            raise ValueError(f"Insufficient chunks ({len(all_chunks)}) for meaningful bootstrap")
        
        # Phase 2: Generate or prepare initial embeddings
        logger.info("Phase 2: Preparing initial embeddings")
        if self.embedder:
            embedded_chunks = self._generate_embeddings(all_chunks)
        else:
            embedded_chunks = self._prepare_tfidf_embeddings(all_chunks)
        
        # Phase 3: Build initial similarity graph
        logger.info("Phase 3: Building initial graph")
        initial_graph = self._build_similarity_graph(embedded_chunks)
        
        # Phase 4: Train initial ISNE model
        logger.info("Phase 4: Training initial ISNE model")
        training_results = self._train_initial_isne(embedded_chunks, initial_graph)
        
        # Phase 5: Generate enhanced embeddings
        logger.info("Phase 5: Generating enhanced embeddings")
        enhanced_embeddings = self._generate_enhanced_embeddings(embedded_chunks)
        
        # Phase 6: Build refined graph
        logger.info("Phase 6: Building refined graph")
        refined_graph = self._build_refined_graph(embedded_chunks, enhanced_embeddings)
        
        # Phase 7: Optional retraining
        logger.info("Phase 7: Retraining with refined graph")
        final_results = self._retrain_with_refined_graph(embedded_chunks, refined_graph)
        
        # Phase 8: Save final model and results
        logger.info("Phase 8: Saving results")
        self._save_bootstrap_results(embedded_chunks, enhanced_embeddings, refined_graph)
        
        total_time = datetime.now(timezone.utc) - start_time
        
        # Compile results
        bootstrap_results = {
            'corpus_stats': {
                'total_files': len(input_files),
                'total_chunks': len(all_chunks),
                'corpus_directory': str(corpus_dir)
            },
            'graph_stats': {
                'initial_graph': {
                    'nodes': initial_graph.number_of_nodes(),
                    'edges': initial_graph.number_of_edges(),
                    'density': nx.density(initial_graph)
                },
                'refined_graph': {
                    'nodes': refined_graph.number_of_nodes(),
                    'edges': refined_graph.number_of_edges(),
                    'density': nx.density(refined_graph)
                }
            },
            'training_results': {
                'initial': training_results,
                'final': final_results
            },
            'component_configuration': {
                'document_processor': self.pipeline_config.document_processor.implementation,
                'chunker': self.pipeline_config.chunker.implementation,
                'embedder': self.pipeline_config.embedder.implementation if self.embedder else 'tfidf',
                'graph_enhancer': 'isne'
            },
            'bootstrap_time': total_time.total_seconds(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Save summary
        results_file = self.output_dir / "bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(bootstrap_results, f, indent=2, default=str)
        
        logger.info(f"Bootstrap completed in {total_time}")
        return bootstrap_results
    
    def _process_documents(self, input_files: List[Path]) -> List[Dict[str, Any]]:
        """Process documents using shared components."""
        # Convert paths to strings for component compatibility
        file_paths = [str(f) for f in input_files]
        
        # Process documents
        documents = self.doc_processor.run(file_paths)
        logger.info(f"Processed {len(documents)} documents")
        
        # Chunk documents
        chunks = self.chunker.run(documents)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Convert to dictionary format for easier processing
        all_chunks = []
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'text': chunk.text,
                'source_document': getattr(chunk, 'source_document', 'unknown'),
                'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
            }
            all_chunks.append(chunk_dict)
        
        return all_chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings using configured embedder."""
        # Group chunks by source document for embedder
        doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            source = chunk.get('source_document', 'unknown')
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(chunk)
        
        # Create document schemas for embedder
        documents = []
        for source, chunk_list in doc_chunks.items():
            doc = DocumentSchema(
                file_id=f"bootstrap_{hash(source) % 10000}",
                file_name=source,
                file_path=f"corpus/{source}",
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
            documents.append(doc)
        
        # Generate embeddings
        embedded_docs = self.embedder.run(documents)
        
        # Extract embeddings back to chunks
        embedded_chunks = []
        for doc in embedded_docs:
            for chunk_schema in doc.chunks:
                chunk_dict = {
                    'id': chunk_schema.id,
                    'text': chunk_schema.text,
                    'source_document': doc.file_name,
                    'embedding': chunk_schema.embedding,
                    'metadata': chunk_schema.metadata if hasattr(chunk_schema, 'metadata') else {}
                }
                embedded_chunks.append(chunk_dict)
        
        return embedded_chunks
    
    def _prepare_tfidf_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare TF-IDF embeddings as fallback."""
        texts = [chunk['text'] for chunk in chunks]
        
        vectorizer = TfidfVectorizer(
            max_features=768,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray()
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def _build_similarity_graph(self, chunks: List[Dict[str, Any]]) -> nx.Graph:
        """Build initial graph based on embedding similarity."""
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        
        # Compute similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Build graph
        graph = nx.Graph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            graph.add_node(i, **chunk)
        
        # Add edges based on similarity
        for i in range(len(chunks)):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1]
            
            connections = 0
            for j in top_indices[1:]:  # Skip self
                if similarities[j] >= self.similarity_threshold:
                    if connections < self.max_connections_per_chunk:
                        graph.add_edge(i, j, weight=similarities[j])
                        connections += 1
                else:
                    break
        
        logger.info(f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Save graph
        graph_file = self.output_dir / "graphs" / "initial_graph.json"
        self._save_graph_json(graph, graph_file)
        
        return graph
    
    def _train_initial_isne(self, 
                           chunks: List[Dict[str, Any]], 
                           graph: nx.Graph) -> Dict[str, Any]:
        """Train initial ISNE model."""
        # Convert chunks to document format for training
        documents = self._chunks_to_documents(chunks)
        
        # Configure training
        config_override = {
            "isne": {
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                "model": {
                    "hidden_dim": 128,
                    "output_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1
                }
            }
        }
        
        # Initialize training orchestrator
        self.training_orchestrator = ISNETrainingOrchestrator(
            documents=documents,
            output_dir=self.output_dir / "initial_training",
            model_output_dir=self.output_dir / "models",
            config_override=config_override
        )
        
        # Train model
        training_results = self.training_orchestrator.train()
        self.isne_trainer = self.training_orchestrator.trainer
        
        return training_results
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chunks to document format for training."""
        doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            source = chunk.get('source_document', 'unknown')
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(chunk)
        
        documents = []
        for source, chunk_list in doc_chunks.items():
            doc = {
                'file_id': f"bootstrap_{hash(source) % 10000}",
                'file_name': source,
                'file_path': f"corpus/{source}",
                'chunks': chunk_list,
                'source': f"corpus/{source}"
            }
            documents.append(doc)
        
        return documents
    
    def _generate_enhanced_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate ISNE-enhanced embeddings."""
        if not self.isne_trainer or not self.training_orchestrator:
            raise ValueError("ISNE trainer not initialized")
        
        graph_data = self.training_orchestrator.graph_data
        embeddings = self.isne_trainer.get_embeddings(graph_data.x, graph_data.edge_index)
        
        return embeddings.cpu().numpy()
    
    def _build_refined_graph(self, 
                           chunks: List[Dict[str, Any]], 
                           embeddings: np.ndarray) -> nx.Graph:
        """Build refined graph using enhanced embeddings."""
        # Compute similarities in enhanced space
        similarity_matrix = cosine_similarity(embeddings)
        
        # Build refined graph
        graph = nx.Graph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            graph.add_node(i, **chunk)
        
        # Add edges
        for i in range(len(chunks)):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1]
            
            connections = 0
            for j in top_indices[1:]:
                if similarities[j] >= self.similarity_threshold:
                    if connections < self.max_connections_per_chunk:
                        graph.add_edge(i, j, weight=similarities[j])
                        connections += 1
                else:
                    break
        
        logger.info(f"Built refined graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Save graph
        graph_file = self.output_dir / "graphs" / "refined_graph.json"
        self._save_graph_json(graph, graph_file)
        
        return graph
    
    def _retrain_with_refined_graph(self,
                                   chunks: List[Dict[str, Any]],
                                   graph: nx.Graph) -> Dict[str, Any]:
        """Retrain ISNE with refined graph structure."""
        documents = self._chunks_to_documents(chunks)
        
        config_override = {
            "isne": {
                "training": {
                    "epochs": 50,
                    "learning_rate": 0.0005,
                    "batch_size": 32,
                    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            }
        }
        
        refined_orchestrator = ISNETrainingOrchestrator(
            documents=documents,
            output_dir=self.output_dir / "refined_training",
            model_output_dir=self.output_dir / "models",
            config_override=config_override
        )
        
        training_results = refined_orchestrator.train()
        self.isne_trainer = refined_orchestrator.trainer
        self.training_orchestrator = refined_orchestrator
        
        return training_results
    
    def _save_bootstrap_results(self,
                              chunks: List[Dict[str, Any]],
                              embeddings: np.ndarray,
                              graph: nx.Graph) -> None:
        """Save final bootstrap results."""
        # Save embeddings
        embeddings_file = self.output_dir / "embeddings" / "final_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save chunk metadata
        metadata_file = self.output_dir / "embeddings" / "chunk_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'chunks': chunks,
                'embedding_dimension': embeddings.shape[1],
                'total_chunks': len(chunks),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Saved bootstrap results to {self.output_dir}")
    
    def _save_graph_json(self, graph: nx.Graph, filepath: Path) -> None:
        """Save graph as JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        graph_data = {
            'nodes': [
                {
                    'id': int(n),
                    'chunk_id': str(graph.nodes[n].get('id', '')),
                    'text_preview': str(graph.nodes[n].get('text', ''))[:100]
                }
                for n in graph.nodes()
            ],
            'edges': [
                {
                    'source': int(u),
                    'target': int(v),
                    'weight': float(graph[u][v].get('weight', 1.0))
                }
                for u, v in graph.edges()
            ],
            'stats': {
                'nodes': int(graph.number_of_nodes()),
                'edges': int(graph.number_of_edges()),
                'density': float(nx.density(graph))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)


def run_modular_bootstrap(
    corpus_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    file_pattern: str = "*.pdf"
) -> Dict[str, Any]:
    """
    Convenience function to run modular bootstrap pipeline.
    
    Args:
        corpus_dir: Directory containing documents
        output_dir: Directory for outputs
        config: Pipeline configuration
        file_pattern: File pattern to match
        
    Returns:
        Bootstrap results dictionary
    """
    pipeline = ModularBootstrapPipeline(
        config=config,
        output_dir=Path(output_dir) if output_dir else None
    )
    
    return pipeline.bootstrap_from_corpus(
        Path(corpus_dir),
        file_pattern
    )