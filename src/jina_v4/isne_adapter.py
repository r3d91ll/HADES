"""
Adapter for converting Jina v4 output to ISNE input format.

This module provides utilities to bridge between Jina v4's output format
and ISNE's expected input format.
"""

import logging
from typing import Dict, Any, List
import numpy as np

from src.types.components.contracts import (
    GraphEnhancementInput,
    ChunkEmbedding,
    EmbeddingInput,
    DocumentChunk
)
from src.types.common import NodeID

logger = logging.getLogger(__name__)


def convert_jina_to_isne(jina_output: Dict[str, Any]) -> GraphEnhancementInput:
    """
    Convert Jina v4 output to ISNE GraphEnhancementInput format.
    
    Args:
        jina_output: Output from JinaV4Processor.process()
        
    Returns:
        GraphEnhancementInput ready for ISNE processing
        
    Example:
        >>> jina_result = jina_processor.process({"file_path": "doc.pdf"})
        >>> isne_input = convert_jina_to_isne(jina_result)
        >>> enhanced = isne_processor.enhance(isne_input)
    """
    
    chunks = jina_output.get('chunks', [])
    metadata = jina_output.get('document_metadata', {})
    
    # Convert chunks to ChunkEmbedding format
    chunk_embeddings = []
    
    for chunk in chunks:
        # Handle both single-vector and multi-vector embeddings
        embeddings = chunk['embeddings']
        
        if isinstance(embeddings, np.ndarray):
            if len(embeddings.shape) == 2:
                # Multi-vector: average pool to single vector
                # ISNE currently expects single vectors
                embedding_vector = np.mean(embeddings, axis=0).tolist()
            else:
                # Single vector
                embedding_vector = embeddings.tolist()
        else:
            embedding_vector = embeddings
            
        chunk_embedding = ChunkEmbedding(
            chunk_id=chunk['id'],
            embedding=embedding_vector,
            embedding_dimension=len(embedding_vector),
            model_name=metadata.get('model_used', 'jina-embeddings-v4'),
            confidence=chunk.get('metadata', {}).get('semantic_density'),
            metadata={
                'keywords': chunk.get('keywords', []),
                'relationships': chunk.get('relationships', []),
                'chunk_metadata': chunk.get('metadata', {}),
                'text': chunk.get('text', '')  # Include text for ISNE context
            }
        )
        
        chunk_embeddings.append(chunk_embedding)
    
    # Extract embeddings and node IDs for GraphEnhancementInput
    embeddings = [ce.embedding for ce in chunk_embeddings]
    node_ids = [NodeID(ce.chunk_id) for ce in chunk_embeddings]
    
    # Create GraphEnhancementInput
    graph_input = GraphEnhancementInput(
        embeddings=embeddings,
        node_ids=node_ids,
        graph_structure={  # Build graph structure from relationships
            'nodes': [{'id': ce.chunk_id, 'data': ce.metadata} for ce in chunk_embeddings],
            'edges': jina_output.get('relationships', [])
        },
        enhancement_type="isne_default",
        options={
            'use_keywords': True,  # Use Jina-extracted keywords
            'use_relationships': True,  # Use pre-computed relationships
            'enhancement_strength': 0.3,
            'edge_weight_threshold': 0.7,
            'max_edges_per_node': 10,
            'use_semantic_similarity': True,
            'source': 'jina_v4',
            'document_metadata': metadata,
            'total_chunks': len(chunk_embeddings)
        }
    )
    
    logger.info(
        f"Converted Jina output to ISNE input: "
        f"{len(chunk_embeddings)} chunks ready for enhancement"
    )
    
    return graph_input


def convert_jina_to_embedding_input(jina_output: Dict[str, Any]) -> EmbeddingInput:
    """
    Convert Jina v4 output to EmbeddingInput format.
    
    This is useful if you need to feed Jina output into other components
    that expect EmbeddingInput format.
    
    Args:
        jina_output: Output from JinaV4Processor.process()
        
    Returns:
        EmbeddingInput format
    """
    
    chunks = jina_output.get('chunks', [])
    metadata = jina_output.get('document_metadata', {})
    
    # Convert to EmbeddingChunk format
    embedding_chunks = []
    
    for chunk in chunks:
        embedding_chunk = DocumentChunk(
            id=chunk['id'],
            content=chunk['text'],
            document_id=metadata.get('file_path', 'unknown'),
            chunk_index=chunk.get('metadata', {}).get('start_token', 0),
            chunk_size=len(chunk['text']),
            metadata={
                **chunk.get('metadata', {}),
                'keywords': chunk.get('keywords', []),
                'relationships': chunk.get('relationships', [])
            }
        )
        embedding_chunks.append(embedding_chunk)
    
    # Extract texts from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Create EmbeddingInput
    embedding_input = EmbeddingInput(
        texts=texts,
        model_name=metadata.get('model_used', 'jina-embeddings-v4'),
        options={
            'already_embedded': True,  # Signal that embeddings exist
            'embeddings': {chunk['id']: chunk['embeddings'] for chunk in chunks},
            'chunk_data': embedding_chunks  # Store chunk data in options
        }
    )
    
    return embedding_input


def extract_graph_structure(jina_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract graph structure from Jina output relationships.
    
    This creates an adjacency list representation of the chunk relationships
    that can be used by ISNE for graph construction.
    
    Args:
        jina_output: Output from JinaV4Processor.process()
        
    Returns:
        Graph structure dictionary
    """
    
    chunks = jina_output.get('chunks', [])
    
    # Build adjacency list
    graph: Dict[str, Any] = {
        'nodes': [],
        'edges': [],
        'node_features': {}
    }
    
    # Add nodes
    for chunk in chunks:
        node_id = chunk['id']
        graph['nodes'].append(node_id)
        
        # Store node features
        graph['node_features'][node_id] = {
            'keywords': chunk.get('keywords', []),
            'semantic_density': chunk.get('metadata', {}).get('semantic_density', 0.5),
            'has_visual_content': chunk.get('metadata', {}).get('has_visual_content', False)
        }
        
        # Add edges from relationships
        for rel in chunk.get('relationships', []):
            edge = {
                'source': node_id,
                'target': rel['target_chunk'],
                'weight': rel['similarity'],
                'type': rel.get('type', 'semantic')
            }
            graph['edges'].append(edge)
    
    logger.info(
        f"Extracted graph structure: "
        f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges"
    )
    
    return graph


def prepare_for_storage(
    jina_output: Dict[str, Any],
    enhanced_output: Any = None
) -> List[Dict[str, Any]]:
    """
    Prepare Jina output (optionally with ISNE enhancement) for storage.
    
    Args:
        jina_output: Output from JinaV4Processor.process()
        enhanced_output: Optional output from ISNE enhancement
        
    Returns:
        List of documents ready for storage
    """
    
    chunks = jina_output.get('chunks', [])
    metadata = jina_output.get('document_metadata', {})
    
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc = {
            'id': chunk['id'],
            'content': chunk['text'],
            'embedding': chunk['embeddings'],  # Original Jina embedding
            'keywords': chunk.get('keywords', []),
            'metadata': {
                **chunk.get('metadata', {}),
                'model': metadata.get('model_used'),
                'processing_timestamp': metadata.get('timestamp'),
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
        }
        
        # Add enhanced embedding if available
        if enhanced_output:
            # Find corresponding enhanced embedding
            for enhanced in enhanced_output.embeddings:
                if enhanced.chunk_id == chunk['id']:
                    doc['enhanced_embedding'] = enhanced.enhanced_embedding
                    doc['graph_features'] = enhanced.graph_features
                    doc['enhancement_score'] = enhanced.enhancement_score
                    break
        
        # Add relationships as edge data
        doc['edges'] = [
            {
                'target': rel['target_chunk'],
                'weight': rel['similarity'],
                'type': rel.get('type', 'semantic')
            }
            for rel in chunk.get('relationships', [])
        ]
        
        documents.append(doc)
    
    return documents