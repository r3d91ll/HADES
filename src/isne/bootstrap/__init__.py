"""
ISNE Bootstrap and Adaptive Training Package

This package implements bootstrap and adaptive training strategies for ISNE:

1. Bootstrap Strategy: Cold start training from document corpus
2. Adaptive Training: Intelligent retraining for dynamic knowledge graphs

Main Components:
- ISNEBootstrapper: Complete bootstrap process for cold start
- AdaptiveISNETrainer: Smart retraining strategy
"""

# Import adaptive training module
from .adaptive_training import AdaptiveISNETrainer

# Re-export for backward compatibility
__all__ = ['ISNEBootstrapper', 'AdaptiveISNETrainer', 'bootstrap_corpus']

import logging
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch

from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
from src.isne.training.trainer import ISNETrainer
from src.orchestration.pipelines.stages.document_processor import DocumentProcessorStage
from src.orchestration.pipelines.stages.chunking import ChunkingStage
from src.validation.embedding_validator import validate_embeddings_after_isne

logger = logging.getLogger(__name__)


class ISNEBootstrapper:
    """
    Bootstrap ISNE training from a cold start with document corpus.
    
    This handles the chicken-and-egg problem of needing a graph to train ISNE
    but needing ISNE to create meaningful graph relationships.
    """
    
    def __init__(self, 
                 corpus_dir: Path,
                 output_dir: Path,
                 similarity_threshold: float = 0.3,
                 max_connections_per_chunk: int = 10,
                 min_cluster_size: int = 5):
        """
        Initialize bootstrap process.
        
        Args:
            corpus_dir: Directory containing PDF files
            output_dir: Directory for outputs and models
            similarity_threshold: Minimum similarity for chunk connections
            max_connections_per_chunk: Maximum edges per node in bootstrap graph
            min_cluster_size: Minimum size for document clusters
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.similarity_threshold = similarity_threshold
        self.max_connections_per_chunk = max_connections_per_chunk
        self.min_cluster_size = min_cluster_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.doc_processor = DocumentProcessorStage()
        self.chunker = ChunkingStage()
        self.isne_trainer: Optional[Any] = None
        self.training_orchestrator: Optional[Any] = None
        
    def bootstrap_full_corpus(self) -> Dict[str, Any]:
        """
        Complete bootstrap process for the entire corpus.
        
        Returns:
            Dictionary containing bootstrap results and metrics
        """
        logger.info(f"Starting ISNE bootstrap for corpus: {self.corpus_dir}")
        start_time = datetime.now()
        
        # Phase 1: Process all documents and create chunks
        logger.info("Phase 1: Processing documents and creating chunks")
        all_chunks = self._process_all_documents()
        
        if len(all_chunks) < 10:
            raise ValueError(f"Insufficient chunks ({len(all_chunks)}) for meaningful bootstrap")
        
        logger.info(f"Created {len(all_chunks)} chunks from corpus")
        
        # Phase 2: Add initial embeddings to chunks  
        logger.info("Phase 2: Adding initial TF-IDF embeddings to chunks")
        self._prepare_embeddings_for_chunks(all_chunks)
        
        # Phase 2b: Build initial graph using text similarity
        logger.info("Phase 2b: Building initial graph using text similarity")
        initial_graph = self._build_similarity_graph(all_chunks)
        
        # Phase 3: Train initial ISNE model on similarity graph
        logger.info("Phase 3: Training initial ISNE model")
        training_results = self._train_initial_isne_model(all_chunks, initial_graph)
        
        # Phase 4: Generate production embeddings
        logger.info("Phase 4: Generating production embeddings")
        production_embeddings = self._generate_production_embeddings(all_chunks)
        
        # Phase 5: Build refined graph with ISNE embeddings
        logger.info("Phase 5: Building refined graph with ISNE embeddings")
        refined_graph = self._build_isne_graph(all_chunks, production_embeddings)
        
        # Phase 6: Optional - Retrain with refined graph
        logger.info("Phase 6: Retraining with refined graph structure")
        final_training_results = self._retrain_with_refined_graph(all_chunks, refined_graph)
        
        # Phase 7: Generate final embeddings for database
        logger.info("Phase 7: Generating final embeddings for database")
        final_embeddings = self._generate_final_embeddings(all_chunks)
        
        total_time = datetime.now() - start_time
        
        # Save bootstrap results
        bootstrap_results = {
            'corpus_stats': {
                'total_documents': len(list(self.corpus_dir.glob("*.pdf"))),
                'total_chunks': len(all_chunks),
                'avg_chunk_length': np.mean([len(chunk['text']) for chunk in all_chunks]),
                'corpus_directory': str(self.corpus_dir)
            },
            'graph_stats': {
                'initial_graph': {
                    'nodes': initial_graph.number_of_nodes(),
                    'edges': initial_graph.number_of_edges(),
                    'avg_degree': np.mean([d for n, d in initial_graph.degree()]),
                    'density': nx.density(initial_graph)
                },
                'refined_graph': {
                    'nodes': refined_graph.number_of_nodes(),
                    'edges': refined_graph.number_of_edges(),
                    'avg_degree': np.mean([d for n, d in refined_graph.degree()]),
                    'density': nx.density(refined_graph)
                }
            },
            'training_results': {
                'initial_training': training_results,
                'final_training': final_training_results
            },
            'embedding_stats': {
                'embedding_dimension': final_embeddings.shape[1] if len(final_embeddings) > 0 else 0,
                'total_embeddings': len(final_embeddings)
            },
            'bootstrap_time': total_time.total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        results_file = self.output_dir / "bootstrap_results.json"
        with open(results_file, 'w') as f:
            json.dump(bootstrap_results, f, indent=2, default=str)
        
        logger.info(f"Bootstrap completed in {total_time}")
        logger.info(f"Results saved to {results_file}")
        
        return bootstrap_results
    
    def _process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all PDF documents in corpus directory."""
        
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.corpus_dir}")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        for i, pdf_file in enumerate(pdf_files):
            try:
                logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
                
                # Process document
                documents = self.doc_processor.run([str(pdf_file)])
                
                if documents:
                    # Chunk document
                    chunks = self.chunker.run(documents)
                    
                    # Convert chunks to dictionaries for easier processing
                    for chunk in chunks:
                        chunk_dict = {
                            'id': chunk.id,
                            'text': chunk.text,
                            'source_document': pdf_file.name,
                            'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
                        }
                        all_chunks.append(chunk_dict)
                        
                    logger.info(f"  Created {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        return all_chunks
    
    def _build_similarity_graph(self, chunks: List[Dict[str, Any]]) -> nx.Graph:
        """Build initial graph based on text similarity."""
        
        logger.info("Computing text similarities using TF-IDF")
        
        # Extract text for vectorization
        texts = [chunk['text'] for chunk in chunks]
        
        # Use TF-IDF for initial similarity computation
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Compute cosine similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Build graph
        graph = nx.Graph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            graph.add_node(i, **chunk)
        
        # Add edges based on similarity
        edges_added = 0
        for i in range(len(chunks)):
            # Get top similar chunks
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1]
            
            connections_added = 0
            for j in top_indices[1:]:  # Skip self (index 0)
                if similarities[j] >= self.similarity_threshold:
                    if connections_added < self.max_connections_per_chunk:
                        graph.add_edge(i, j, weight=similarities[j])
                        edges_added += 1
                        connections_added += 1
                else:
                    break
        
        logger.info(f"Built initial graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Save graph for debugging
        graph_file = self.output_dir / "graphs" / "initial_similarity_graph.json"
        self._save_graph_json(graph, graph_file)
        
        return graph
    
    def _train_initial_isne_model(self, 
                                chunks: List[Dict[str, Any]], 
                                graph: nx.Graph) -> Dict[str, Any]:
        """Train initial ISNE model on similarity-based graph."""
        
        # Convert chunks to document format for training orchestrator
        documents = self._chunks_to_documents(chunks)
        
        # Configure training orchestrator - config must be nested under "isne"
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
        
        # Initialize training orchestrator with documents
        self.training_orchestrator = ISNETrainingOrchestrator(
            documents=documents,
            output_dir=self.output_dir / "initial_training",
            model_output_dir=self.output_dir / "models",
            config_override=config_override
        )
        
        # Train model
        training_results = self.training_orchestrator.train()
        
        # Store trainer for later use
        self.isne_trainer = self.training_orchestrator.trainer
        
        return training_results
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert flat list of chunks back to document format for training."""
        
        # Group chunks by source document
        doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            doc_name = chunk.get('source_document', 'unknown_document')
            if doc_name not in doc_chunks:
                doc_chunks[doc_name] = []
            doc_chunks[doc_name].append(chunk)
        
        # Create document objects
        documents = []
        for doc_name, chunk_list in doc_chunks.items():
            doc = {
                'file_id': f"bootstrap_{hash(doc_name) % 10000}",
                'file_name': doc_name,
                'file_path': f"corpus/{doc_name}",
                'chunks': chunk_list,
                'source': f"corpus/{doc_name}"
            }
            documents.append(doc)
        
        return documents
    
    def _generate_production_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings using trained ISNE model."""
        
        if self.isne_trainer is None:
            raise ValueError("ISNE trainer not initialized. Run training first.")
        
        # Convert chunks to appropriate format
        documents = self._chunks_to_documents(chunks)
        
        # Get graph data from training orchestrator
        if self.training_orchestrator is None or self.training_orchestrator.graph_data is None:
            raise ValueError("Graph data not available. Training may have failed.")
        
        graph_data = self.training_orchestrator.graph_data
        
        # Generate embeddings using trained model
        embeddings = self.isne_trainer.get_embeddings(graph_data.x, graph_data.edge_index)
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        # Save embeddings
        embeddings_file = self.output_dir / "embeddings" / "production_embeddings.npy"
        np.save(embeddings_file, embeddings_np)
        
        return embeddings_np
    
    def _build_isne_graph(self, 
                         chunks: List[Dict[str, Any]], 
                         embeddings: np.ndarray) -> nx.Graph:
        """Build refined graph using ISNE embeddings."""
        
        # Compute similarities in ISNE embedding space
        embedding_similarities = cosine_similarity(embeddings)
        
        # Build refined graph
        graph = nx.Graph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            graph.add_node(i, **chunk)
        
        # Add edges based on ISNE embedding similarity
        for i in range(len(chunks)):
            similarities = embedding_similarities[i]
            top_indices = np.argsort(similarities)[::-1]
            
            connections_added = 0
            for j in top_indices[1:]:  # Skip self
                if similarities[j] >= self.similarity_threshold:
                    if connections_added < self.max_connections_per_chunk:
                        graph.add_edge(i, j, weight=similarities[j])
                        connections_added += 1
                else:
                    break
        
        logger.info(f"Built refined graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Save refined graph
        graph_file = self.output_dir / "graphs" / "refined_isne_graph.json"
        self._save_graph_json(graph, graph_file)
        
        return graph
    
    def _retrain_with_refined_graph(self, 
                                  chunks: List[Dict[str, Any]], 
                                  refined_graph: nx.Graph) -> Dict[str, Any]:
        """Retrain ISNE model with refined graph structure."""
        
        # Convert chunks to document format
        documents = self._chunks_to_documents(chunks)
        
        # Configure refined training - config must be nested under "isne"
        config_override = {
            "isne": {
                "training": {
                    "epochs": 50,  # Fewer epochs for refinement
                    "learning_rate": 0.0005,  # Lower learning rate
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
        
        # Initialize new training orchestrator for refinement
        refined_orchestrator = ISNETrainingOrchestrator(
            documents=documents,
            output_dir=self.output_dir / "refined_training",
            model_output_dir=self.output_dir / "models",
            config_override=config_override
        )
        
        # Train refined model
        training_results = refined_orchestrator.train()
        
        # Update our trainer reference
        self.isne_trainer = refined_orchestrator.trainer
        self.training_orchestrator = refined_orchestrator
        
        return training_results
    
    def _generate_final_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate final embeddings for database storage."""
        
        if self.isne_trainer is None:
            raise ValueError("ISNE trainer not initialized. Run training first.")
        
        # Get graph data from training orchestrator
        if self.training_orchestrator is None or self.training_orchestrator.graph_data is None:
            raise ValueError("Graph data not available. Training may have failed.")
        
        graph_data = self.training_orchestrator.graph_data
        
        # Generate final embeddings using trained model
        final_embeddings = self.isne_trainer.get_embeddings(graph_data.x, graph_data.edge_index)
        
        # Convert to numpy
        final_embeddings_np = final_embeddings.cpu().numpy()
        
        # Validate embedding quality
        quality_metrics = self._validate_embedding_quality(final_embeddings_np)
        
        # Save final embeddings and metadata
        embeddings_file = self.output_dir / "embeddings" / "final_embeddings.npy"
        metadata_file = self.output_dir / "embeddings" / "chunk_metadata.json"
        
        np.save(embeddings_file, final_embeddings_np)
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'chunks': chunks,
                'embedding_dimension': final_embeddings_np.shape[1],
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Generated {len(final_embeddings_np)} final embeddings")
        logger.info(f"Embedding quality score: {quality_metrics.get('overall_score', 'N/A')}")
        
        return final_embeddings_np
    
    def _prepare_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Prepare initial embeddings for chunks using TF-IDF."""
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Use TF-IDF to create initial embeddings
        vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
        tfidf_embeddings = vectorizer.fit_transform(texts).toarray()
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = tfidf_embeddings[i].tolist()
    
    def _validate_embedding_quality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Simple embedding quality validation for bootstrap."""
        if len(embeddings) == 0:
            return {'overall_score': 0.0, 'dimension': 0, 'variance': 0.0}
        
        # Basic quality metrics
        variance = float(np.var(embeddings))
        mean_norm = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        
        # Simple quality score (higher variance and reasonable norms are good)
        quality_score = min(1.0, variance * 10)  # Basic heuristic
        
        return {
            'overall_score': quality_score,
            'dimension': embeddings.shape[1],
            'variance': variance,
            'mean_norm': mean_norm,
            'total_embeddings': len(embeddings)
        }
    
    def _save_graph_json(self, graph: nx.Graph, filepath: Path) -> None:
        """Save NetworkX graph as JSON for debugging."""
        
        graph_data = {
            'nodes': [
                {'id': int(n), 'chunk_id': str(graph.nodes[n].get('id', '')), 
                 'text_preview': str(graph.nodes[n].get('text', ''))[:100]}
                for n in graph.nodes()
            ],
            'edges': [
                {'source': int(u), 'target': int(v), 'weight': float(graph[u][v].get('weight', 1.0))}
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


def bootstrap_corpus(corpus_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Convenience function to bootstrap ISNE from a corpus directory.
    
    Args:
        corpus_dir: Directory containing PDF files
        output_dir: Directory for outputs
        
    Returns:
        Bootstrap results dictionary
    """
    bootstrapper = ISNEBootstrapper(
        corpus_dir=Path(corpus_dir),
        output_dir=Path(output_dir)
    )
    
    return bootstrapper.bootstrap_full_corpus()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap ISNE from document corpus")
    parser.add_argument("--corpus-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--similarity-threshold", type=float, default=0.3, 
                       help="Similarity threshold for connections")
    
    args = parser.parse_args()
    
    results = bootstrap_corpus(args.corpus_dir, args.output_dir)
    print(f"Bootstrap completed. Results saved to {args.output_dir}")