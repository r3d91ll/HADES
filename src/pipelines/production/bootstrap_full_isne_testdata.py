#!/usr/bin/env python3
"""
Bootstrap script for ingesting the full ISNE test dataset.

This script processes the same dataset used in the sequential ISNE project
to provide a baseline comparison within the HADES framework.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np
from datetime import datetime, timezone

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.arango_client import ArangoClient
from src.components.docproc.factory import create_docproc_component
from src.components.chunking.factory import create_chunking_component
from src.components.embedding.factory import create_embedding_component
# Simplified imports - avoid complex pipeline types for now

# Configure comprehensive logging
def setup_logging(db_name: str):
    """Setup comprehensive logging with file and console output."""
    log_dir = Path(__file__).parent.parent / "logs" / "bootstrap"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"full_isne_bootstrap_{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with info level (not debug)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with info level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Bootstrap logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.debug(f"Debug logging enabled for comprehensive error tracking")
    
    return log_file

logger = logging.getLogger(__name__)


class ISNETestDataBootstrapper:
    """Bootstrap the full ISNE test dataset into HADES with proper graph structure."""
    
    def __init__(self, db_name: str = "isne_testdata_full"):
        self.db_name = db_name
        self.arango_client = ArangoClient()
        self.stats = defaultdict(int)
        
        # Initialize components with basic configs
        self.doc_processor = create_docproc_component(
            component_name="core",
            config=None
        )
        self.chunker = create_chunking_component(
            component_name="cpu",  
            config={"chunk_size": 512, "overlap": 50}
        )
        self.embedder = create_embedding_component(
            component_name="cpu",
            config={"model_name": "all-MiniLM-L6-v2", "batch_size": 32}
        )
        
    def setup_database(self):
        """Setup database and collections with proper indexes."""
        logger.info(f"Setting up database: {self.db_name}")
        
        # Try to connect to the database, if it doesn't exist, use existing one
        success = self.arango_client.connect_to_database(self.db_name)
        if not success:
            logger.warning(f"Database {self.db_name} doesn't exist, using sequential_isne_testdata")
            self.db_name = "sequential_isne_testdata"
            success = self.arango_client.connect_to_database(self.db_name)
            if not success:
                logger.error(f"Failed to connect to fallback database: {self.db_name}")
                raise Exception(f"Database connection failed: {self.db_name}")
            
        # Get database handle
        db = self.arango_client._database
        
        # Define collections
        collections = {
            # Node collections
            'code_files': {'type': 'document'},
            'documentation_files': {'type': 'document'},
            'config_files': {'type': 'document'},
            'chunks': {'type': 'document'},
            'embeddings': {'type': 'document'},
            
            # Edge collections
            'intra_modal_edges': {'type': 'edge'},
            'cross_modal_edges': {'type': 'edge'},
            'directory_edges': {'type': 'edge'},
            'similarity_edges': {'type': 'edge'},
            'sequential_edges': {'type': 'edge'}
        }
        
        # Create collections
        for name, config in collections.items():
            if not db.has_collection(name):
                if config['type'] == 'edge':
                    db.create_collection(name, edge=True)
                else:
                    db.create_collection(name)
                logger.info(f"Created collection: {name}")
                
        # Create indexes for better query performance
        chunks_coll = db.collection('chunks')
        chunks_coll.add_hash_index(fields=['source_file_id'])
        chunks_coll.add_hash_index(fields=['chunk_index'])
        
        embeddings_coll = db.collection('embeddings')
        embeddings_coll.add_hash_index(fields=['chunk_id'])
        
        return db
        
    def categorize_file(self, file_path: str) -> str:
        """Categorize file based on extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.py', '.js', '.java', '.cpp', '.c', '.rs', '.go']:
            return 'code_files'
        elif ext in ['.md', '.rst', '.txt', '.doc', '.pdf']:
            return 'documentation_files'
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.xml']:
            return 'config_files'
        else:
            # Default to documentation for unknown types
            return 'documentation_files'
            
    def process_directory(self, directory: Path, db) -> Dict[str, List[str]]:
        """Process all files in a directory and return file IDs by collection."""
        logger.info(f"Processing directory: {directory}")
        
        file_ids_by_collection = defaultdict(list)
        files = list(directory.iterdir())
        
        for file_path in files:
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    # Skip __pycache__ and compiled files
                    if '__pycache__' in str(file_path) or file_path.suffix in ['.pyc', '.pyo']:
                        continue
                        
                    # Check if file already exists to avoid duplicates
                    collection_name = self.categorize_file(str(file_path))
                    collection = db.collection(collection_name)
                    
                    # Query for existing file
                    existing_query = f"""
                    FOR doc IN {collection_name}
                    FILTER doc.file_path == @file_path
                    RETURN doc._key
                    """
                    cursor = db.aql.execute(existing_query, bind_vars={'file_path': str(file_path)})
                    existing_files = list(cursor)
                    
                    if existing_files:
                        # File already exists, skip processing
                        file_id = existing_files[0]
                        file_ids_by_collection[collection_name].append(f"{collection_name}/{file_id}")
                        self.stats['files_skipped'] += 1
                        logger.info(f"Skipped existing file: {file_path}")
                        continue
                        
                    # Process document
                    processed_docs = self.doc_processor.process_documents([str(file_path)])
                    if not processed_docs:
                        logger.warning(f"Document processing returned no results for: {file_path}")
                        continue
                        
                    doc = processed_docs[0]
                    logger.info(f"Processing: {file_path} ({len(doc.content)} chars)")
                    
                    # Store file document
                    file_doc = {
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'directory': str(directory),
                        'content': doc.content[:10000],  # Store first 10k chars
                        'metadata': doc.metadata,
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    result = collection.insert(file_doc)
                    file_id = result['_key']
                    file_ids_by_collection[collection_name].append(f"{collection_name}/{file_id}")
                    logger.debug(f"File stored with ID: {file_id}")
                    
                    # Process chunks
                    logger.debug(f"Starting chunking for file: {file_path}")
                    from src.types.components.contracts import ChunkingInput
                    chunk_input = ChunkingInput(
                        text=doc.content,
                        document_id=str(file_path),
                        content=doc.content,
                        source_id=str(file_path),
                        content_type="text"
                    )
                    chunk_result = self.chunker.chunk(chunk_input)
                    chunks = chunk_result.chunks
                    logger.debug(f"Created {len(chunks)} chunks")
                    chunk_ids = []
                    
                    for idx, chunk in enumerate(chunks):
                        chunk_doc = {
                            'source_file_id': f"{collection_name}/{file_id}",
                            'chunk_index': idx,
                            'content': chunk.text,  # TextChunk uses 'text' not 'content'
                            'metadata': chunk.metadata,
                            'start_idx': chunk.start_index,  # TextChunk uses 'start_index'
                            'end_idx': chunk.end_index      # TextChunk uses 'end_index'
                        }
                        
                        chunk_result = db.collection('chunks').insert(chunk_doc)
                        chunk_ids.append(chunk_result['_key'])
                        
                    # Convert TextChunks to DocumentChunks for embedding
                    from src.types.components.contracts import DocumentChunk, EmbeddingInput
                    doc_chunks = []
                    for idx, chunk in enumerate(chunks):
                        doc_chunk = DocumentChunk(
                            id=f"{file_id}_{idx}",
                            content=chunk.text,
                            document_id=str(file_path),
                            chunk_index=idx,
                            start_position=chunk.start_index,
                            end_position=chunk.end_index,
                            chunk_size=len(chunk.text),
                            metadata=chunk.metadata
                        )
                        doc_chunks.append(doc_chunk)
                    
                    # Generate embeddings
                    logger.debug(f"Starting embedding generation for {len(doc_chunks)} chunks")
                    embedding_input = EmbeddingInput(
                        chunks=doc_chunks,
                        model_name="all-MiniLM-L6-v2"
                    )
                    embedding_result = self.embedder.embed(embedding_input)
                    embeddings = embedding_result.embeddings
                    logger.debug(f"Generated {len(embeddings)} embeddings")
                    
                    for chunk_id, embedding in zip(chunk_ids, embeddings):
                        embed_doc = {
                            'chunk_id': f"chunks/{chunk_id}",
                            'embedding': embedding.embedding,  # Already a list of floats
                            'model': embedding.model_name,
                            'dimensions': embedding.embedding_dimension
                        }
                        db.collection('embeddings').insert(embed_doc)
                        
                    # Create sequential edges between chunks
                    for i in range(len(chunk_ids) - 1):
                        edge_doc = {
                            '_from': f"chunks/{chunk_ids[i]}",
                            '_to': f"chunks/{chunk_ids[i+1]}",
                            'edge_type': 'sequential',
                            'weight': 1.0,
                            'source_file': f"{collection_name}/{file_id}"
                        }
                        db.collection('sequential_edges').insert(edge_doc)
                        
                    self.stats['files_processed'] += 1
                    self.stats[f'{collection_name}_count'] += 1
                    self.stats['chunks_created'] += len(chunks)
                    self.stats['embeddings_created'] += len(embeddings)
                    self.stats['sequential_edges'] += len(chunk_ids) - 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    logger.debug(f"Error details for {file_path}", exc_info=True)
                    self.stats['errors'] += 1
                    
        return file_ids_by_collection
        
    def create_directory_edges(self, db):
        """Create edges between files in the same directory."""
        logger.info("Creating directory co-location edges...")
        
        # Query to find files in same directories
        query = """
        FOR doc1 IN UNION(
            (FOR d IN code_files RETURN {id: CONCAT('code_files/', d._key), dir: d.directory}),
            (FOR d IN documentation_files RETURN {id: CONCAT('documentation_files/', d._key), dir: d.directory}),
            (FOR d IN config_files RETURN {id: CONCAT('config_files/', d._key), dir: d.directory})
        )
        FOR doc2 IN UNION(
            (FOR d IN code_files RETURN {id: CONCAT('code_files/', d._key), dir: d.directory}),
            (FOR d IN documentation_files RETURN {id: CONCAT('documentation_files/', d._key), dir: d.directory}),
            (FOR d IN config_files RETURN {id: CONCAT('config_files/', d._key), dir: d.directory})
        )
        FILTER doc1.dir == doc2.dir AND doc1.id < doc2.id
        RETURN {from: doc1.id, to: doc2.id, directory: doc1.dir}
        """
        
        cursor = db.aql.execute(query)
        edge_count = 0
        
        for result in cursor:
            edge_doc = {
                '_from': result['from'],
                '_to': result['to'],
                'edge_type': 'co_location',
                'directory': result['directory'],
                'weight': 0.5
            }
            db.collection('directory_edges').insert(edge_doc)
            edge_count += 1
            
        logger.info(f"Created {edge_count} directory edges")
        self.stats['directory_edges'] = edge_count
        
    def create_cross_modal_edges(self, db):
        """Create edges between different file types in same directory."""
        logger.info("Creating cross-modal edges...")
        
        query = """
        LET code_docs = (FOR d IN code_files RETURN {id: CONCAT('code_files/', d._key), dir: d.directory, name: d.file_name})
        LET doc_files = (FOR d IN documentation_files RETURN {id: CONCAT('documentation_files/', d._key), dir: d.directory, name: d.file_name})
        LET config_docs = (FOR d IN config_files RETURN {id: CONCAT('config_files/', d._key), dir: d.directory, name: d.file_name})
        
        // Code to Documentation edges
        FOR code IN code_docs
            FOR doc IN doc_files
                FILTER code.dir == doc.dir
                RETURN {from: doc.id, to: code.id, type: 'doc_to_code', confidence: 0.8}
        """
        
        cursor = db.aql.execute(query)
        edge_count = 0
        
        for result in cursor:
            edge_doc = {
                '_from': result['from'],
                '_to': result['to'],
                'edge_type': result['type'],
                'confidence': result['confidence']
            }
            db.collection('cross_modal_edges').insert(edge_doc)
            edge_count += 1
            
        logger.info(f"Created {edge_count} cross-modal edges")
        self.stats['cross_modal_edges'] = edge_count
        
    def create_similarity_edges(self, db, threshold: float = 0.8):
        """Create edges based on embedding similarity."""
        logger.info("Creating similarity-based edges...")
        
        # Get all embeddings with chunk info (limit for performance)
        query = """
        FOR e IN embeddings
            LET chunk = DOCUMENT(e.chunk_id)
            FILTER chunk != null
            LIMIT 1000
            RETURN {
                chunk_id: e.chunk_id,
                embedding: e.embedding,
                source_file: chunk.source_file_id
            }
        """
        
        cursor = db.aql.execute(query)
        embeddings_data = list(cursor)
        
        if len(embeddings_data) < 2:
            logger.warning("Not enough embeddings for similarity calculation")
            return
            
        # Convert to numpy array for efficient computation
        embeddings_array = np.array([e['embedding'] for e in embeddings_data])
        
        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / (norms + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Create edges for high similarity pairs
        edge_count = 0
        for i in range(len(embeddings_data)):
            for j in range(i + 1, len(embeddings_data)):
                sim = similarity_matrix[i, j]
                
                # Only create edge if similarity exceeds threshold and from different files
                if sim > threshold and embeddings_data[i]['source_file'] != embeddings_data[j]['source_file']:
                    edge_doc = {
                        '_from': embeddings_data[i]['chunk_id'],
                        '_to': embeddings_data[j]['chunk_id'],
                        'edge_type': 'semantic_similarity',
                        'similarity': float(sim),
                        'weight': float(sim)
                    }
                    db.collection('similarity_edges').insert(edge_doc)
                    edge_count += 1
                    
        logger.info(f"Created {edge_count} similarity edges")
        self.stats['similarity_edges'] = edge_count
        
    def process_dataset(self, dataset_path: Path):
        """Process the entire ISNE test dataset."""
        start_time = time.time()
        
        # Setup database
        db = self.setup_database()
        
        # Process all directories recursively
        total_dirs = sum([1 for root, dirs, files in os.walk(dataset_path) if files])
        processed_dirs = 0
        
        for root, dirs, files in os.walk(dataset_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            if files:
                root_path = Path(root)
                processed_dirs += 1
                logger.info(f"Processing directory {processed_dirs}/{total_dirs}: {root_path}")
                file_ids = self.process_directory(root_path, db)
                
        # Create various edge types
        self.create_directory_edges(db)
        self.create_cross_modal_edges(db)
        self.create_similarity_edges(db)
        
        # Calculate final statistics
        end_time = time.time()
        self.stats['processing_time'] = end_time - start_time
        
        # Get graph density
        total_nodes = sum([
            db.collection('code_files').count(),
            db.collection('documentation_files').count(),
            db.collection('config_files').count(),
            db.collection('chunks').count()
        ])
        
        total_edges = sum([
            db.collection('intra_modal_edges').count(),
            db.collection('cross_modal_edges').count(),
            db.collection('directory_edges').count(),
            db.collection('similarity_edges').count(),
            db.collection('sequential_edges').count()
        ])
        
        self.stats['total_nodes'] = total_nodes
        self.stats['total_edges'] = total_edges
        self.stats['graph_density'] = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="Bootstrap full ISNE test dataset into HADES")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (overrides defaults)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to ISNE test dataset (overrides config)'
    )
    parser.add_argument(
        '--db-name',
        type=str,
        help='Target database name (overrides config)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        help='Similarity threshold for creating edges (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    from .config_loader import config_loader
    
    try:
        config = config_loader.load_bootstrap_config()
        config_loader.validate_config(config, "bootstrap")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    overrides = {}
    if args.dataset_path:
        overrides['input'] = {'dataset_path': args.dataset_path}
    if args.db_name:
        overrides['database'] = {'name': args.db_name}
    if args.similarity_threshold:
        overrides['processing'] = {'similarity_threshold': args.similarity_threshold}
    
    if overrides:
        config = config_loader.merge_config_with_overrides(config, overrides)
    
    # Extract config values
    dataset_path = Path(config['input']['dataset_path'])
    db_name = config['database']['name']
    similarity_threshold = config['processing']['similarity_threshold']
    
    # Setup comprehensive logging first
    log_file = setup_logging(db_name)
    
    # Validate dataset path
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)
        
    logger.info(f"Starting bootstrap of ISNE test dataset from: {dataset_path}")
    logger.info(f"Target database: {db_name}")
    logger.info(f"Using configuration: {config['stage']['description']}")
    logger.debug(f"Full debug logging active, log file: {log_file}")
    
    # Run bootstrap
    bootstrapper = ISNETestDataBootstrapper(db_name=db_name)
    stats = bootstrapper.process_dataset(dataset_path)
    
    # Print results
    print("\n" + "="*60)
    print("ISNE TEST DATA BOOTSTRAP RESULTS")
    print("="*60)
    print(f"✅ Dataset successfully bootstrapped!")
    print(f"Database: {db_name}")
    print(f"\nFiles processed: {stats['files_processed']}")
    print(f"Files skipped (already exist): {stats.get('files_skipped', 0)}")
    print(f"  - Code files: {stats['code_files_count']}")
    print(f"  - Documentation: {stats['documentation_files_count']}")
    print(f"  - Config files: {stats['config_files_count']}")
    print(f"\nGraph structure created:")
    print(f"  - Total nodes: {stats['total_nodes']}")
    print(f"  - Total edges: {stats['total_edges']}")
    print(f"  - Sequential edges: {stats['sequential_edges']}")
    print(f"  - Directory edges: {stats['directory_edges']}")
    print(f"  - Cross-modal edges: {stats['cross_modal_edges']}")
    print(f"  - Similarity edges: {stats['similarity_edges']}")
    print(f"  - Graph density: {stats['graph_density']:.4f}")
    print(f"\nChunks created: {stats['chunks_created']}")
    print(f"Embeddings generated: {stats['embeddings_created']}")
    print(f"Processing time: {stats['processing_time']:.2f}s")
    print(f"Errors: {stats['errors']}")
    
    # Show some statistics about what was enhanced vs fresh
    if stats.get('files_skipped', 0) > 0:
        total_attempted = stats['files_processed'] + stats.get('files_skipped', 0)
        enhancement_ratio = stats.get('files_skipped', 0) / total_attempted * 100
        print(f"\nDatabase Enhancement:")
        print(f"  - {enhancement_ratio:.1f}% files already existed")
        print(f"  - {100-enhancement_ratio:.1f}% new files added")
    
    # Check if ready for ISNE training
    if stats['total_nodes'] >= 10 and stats['total_edges'] >= 5:
        print(f"\n✅ Database is ready for ISNE training!")
        print(f"   Nodes: {stats['total_nodes']} (minimum: 10)")
        print(f"   Edges: {stats['total_edges']} (minimum: 5)")
        print(f"   Edge/Node ratio: {stats['total_edges']/stats['total_nodes']:.2f} (minimum: 0.5)")
    else:
        print(f"\n⚠️  Warning: Insufficient data for ISNE training")
        print(f"   Need at least 10 nodes and 5 edges")


if __name__ == "__main__":
    main()