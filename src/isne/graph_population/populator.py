"""
ISNE Graph Populator - Basic Validation Implementation

This module provides a basic implementation of the ISNEGraphPopulator
for testing relationship discovery from trained ISNE models.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from datetime import datetime
import json

from src.isne.models.isne_model import ISNEModel


class ISNEGraphPopulator:
    """
    Basic implementation for validating ISNE relationship discovery.
    
    This is a simplified version focused on testing the concept before
    building the full ArangoDB integration.
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize the graph populator.
        
        Args:
            confidence_threshold: Minimum similarity score for relationships
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def validate_from_trained_model(
        self, 
        model_path: str, 
        test_chunks: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test relationship discovery on a subset of chunks.
        
        Args:
            model_path: Path to trained ISNE model
            test_chunks: List of chunk dictionaries with 'content' and 'metadata'
            output_path: Optional path to save discovered relationships
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Starting validation with {len(test_chunks)} chunks")
        
        # Load ISNE model
        model = self._load_isne_model(model_path)
        
        # Generate embeddings for test chunks
        embeddings = self._generate_embeddings(model, test_chunks)
        
        # Discover relationships
        relationships = self._discover_relationships(
            test_chunks, embeddings, model_version="test_validation"
        )
        
        # Analyze relationship patterns
        analysis = self._analyze_relationships(relationships, test_chunks)
        
        # Save results if requested
        if output_path:
            self._save_results(relationships, analysis, output_path)
        
        return {
            "chunks_processed": len(test_chunks),
            "relationships_discovered": len(relationships),
            "analysis": analysis,
            "relationships": relationships[:10]  # Sample for review
        }
    
    def _load_isne_model(self, model_path: str) -> ISNEModel:
        """Load trained ISNE model."""
        self.logger.info(f"Loading ISNE model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            num_nodes = config.get('num_nodes', 159940)
            embedding_dim = config.get('embedding_dim', 384)
        else:
            # Try to get from checkpoint directly
            num_nodes = checkpoint.get('num_nodes', 159940)
            embedding_dim = checkpoint.get('embedding_dim', 384)
        
        # Initialize model
        model = ISNEModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _generate_embeddings(
        self, 
        model: ISNEModel, 
        chunks: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Generate ISNE-enhanced embeddings for chunks."""
        self.logger.info("Generating ISNE-enhanced embeddings")
        
        # For validation, we'll use the theta parameters directly
        # This is a simplified approach for testing
        num_chunks = len(chunks)
        
        with torch.no_grad():
            # For validation, we'll use the first num_chunks theta parameters
            # In production, we'd map chunks to actual graph nodes
            if num_chunks > model.num_nodes:
                self.logger.warning(f"More chunks ({num_chunks}) than model nodes ({model.num_nodes})")
                num_chunks = model.num_nodes
            
            # Get theta parameters for our test chunks
            # These are the learned ISNE embeddings
            enhanced_embeddings = model.theta[:num_chunks].detach()
        
        return enhanced_embeddings
    
    def _discover_relationships(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: torch.Tensor,
        model_version: str
    ) -> List[Dict[str, Any]]:
        """Discover relationships between chunks using ISNE embeddings."""
        self.logger.info("Discovering relationships")
        
        relationships = []
        num_chunks = len(chunks)
        
        # Convert to numpy for easier computation
        embeddings_np = embeddings.cpu().numpy()
        
        # Compute pairwise similarities
        for i in range(num_chunks):
            for j in range(i + 1, num_chunks):
                # Compute cosine similarity
                similarity = self._cosine_similarity(
                    embeddings_np[i], embeddings_np[j]
                )
                
                if similarity > self.confidence_threshold:
                    # Determine relationship type
                    chunk_a = chunks[i]
                    chunk_b = chunks[j]
                    rel_type = self._classify_relationship_type(
                        chunk_a, chunk_b, similarity
                    )
                    
                    relationships.append({
                        "from_idx": i,
                        "to_idx": j,
                        "from_content": chunk_a['content'][:100] + "...",
                        "to_content": chunk_b['content'][:100] + "...",
                        "type": rel_type,
                        "confidence": float(similarity),
                        "cross_domain": chunk_a.get('source_type') != chunk_b.get('source_type'),
                        "discovered_in_version": model_version,
                        "discovered_at": datetime.utcnow().isoformat()
                    })
        
        # Sort by confidence
        relationships.sort(key=lambda x: x['confidence'], reverse=True)
        
        return relationships
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _classify_relationship_type(
        self,
        chunk_a: Dict[str, Any],
        chunk_b: Dict[str, Any],
        similarity: float
    ) -> str:
        """Classify the type of relationship based on chunks and similarity."""
        source_a = chunk_a.get('source_type', 'unknown')
        source_b = chunk_b.get('source_type', 'unknown')
        
        # Cross-domain relationships
        if source_a != source_b:
            if source_a == 'code' and source_b == 'document':
                return 'implements' if similarity > 0.85 else 'conceptual'
            elif source_a == 'document' and source_b == 'code':
                return 'documented_by' if similarity > 0.85 else 'conceptual'
        
        # Same-domain relationships
        if source_a == 'code' and source_b == 'code':
            if similarity > 0.9:
                return 'similar_implementation'
            elif similarity > 0.8:
                return 'related_functionality'
        
        if source_a == 'document' and source_b == 'document':
            if similarity > 0.85:
                return 'references'
            else:
                return 'conceptual'
        
        return 'similarity'
    
    def _analyze_relationships(
        self,
        relationships: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze discovered relationships for patterns."""
        if not relationships:
            return {"message": "No relationships discovered"}
        
        # Count relationship types
        type_counts = {}
        cross_domain_count = 0
        
        for rel in relationships:
            rel_type = rel['type']
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            
            if rel['cross_domain']:
                cross_domain_count += 1
        
        # Confidence statistics
        confidences = [rel['confidence'] for rel in relationships]
        
        analysis = {
            "total_relationships": len(relationships),
            "relationship_types": type_counts,
            "cross_domain_relationships": cross_domain_count,
            "cross_domain_percentage": (cross_domain_count / len(relationships)) * 100,
            "confidence_stats": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            },
            "chunks_with_relationships": len(set(
                [rel['from_idx'] for rel in relationships] +
                [rel['to_idx'] for rel in relationships]
            )),
            "chunks_total": len(chunks)
        }
        
        return analysis
    
    def _save_results(
        self,
        relationships: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        output_path: str
    ):
        """Save validation results to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save relationships
        relationships_file = output_dir / "discovered_relationships.json"
        with open(relationships_file, 'w') as f:
            json.dump(relationships, f, indent=2)
        
        # Save analysis
        analysis_file = output_dir / "relationship_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}")


def create_test_chunks(data_dir: str, num_chunks: int = 100) -> List[Dict[str, Any]]:
    """
    Create test chunks from existing data.
    
    This is a placeholder - in production, these would come from
    the document processing pipeline.
    """
    chunks = []
    
    # Simulate mixed document and code chunks
    for i in range(num_chunks):
        if i % 3 == 0:
            # Code chunk
            chunks.append({
                "content": f"def function_{i}():\n    # Implementation for feature {i}\n    return result_{i}",
                "source_type": "code",
                "metadata": {
                    "file_type": "python",
                    "function_name": f"function_{i}"
                }
            })
        else:
            # Document chunk
            chunks.append({
                "content": f"This section describes the implementation of feature {i}. "
                          f"The function_{i // 3} handles this functionality by processing result_{i // 3}.",
                "source_type": "document",
                "metadata": {
                    "section": f"Feature {i}",
                    "doc_type": "technical"
                }
            })
    
    return chunks