"""
Theory-Practice Bridge Detection using Jina v4

This module detects semantic bridges between theoretical research and practical
implementations using Jina v4's attention mechanisms instead of a separate LLM.

Bridge Types:
- Theory to Implementation: Research paper → Code
- Documentation Bridge: Code → Documentation → Usage
- Knowledge Inscription: Abstract concepts → Data structures
- Concept Translation: Theory terminology → Practice terminology
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class BridgeType(Enum):
    """Types of theory-practice bridges."""
    THEORY_TO_IMPLEMENTATION = "theory_to_implementation"
    DOCUMENTATION_BRIDGE = "documentation_bridge"
    KNOWLEDGE_INSCRIPTION = "knowledge_inscription"
    CONCEPT_TRANSLATION = "concept_translation"
    ALGORITHMIC_BRIDGE = "algorithmic_bridge"
    METHODOLOGY_BRIDGE = "methodology_bridge"


class TheoryPracticeBridgeDetector:
    """
    Detects theory-practice bridges using Jina v4's capabilities.
    
    Instead of using a separate code LLM (Devstral), this leverages
    Jina v4's attention mechanisms and multimodal understanding to
    identify conceptual bridges.
    """
    
    def __init__(self, jina_processor: Any):
        """
        Initialize bridge detector with Jina v4 processor.
        
        Args:
            jina_processor: Jina v4 processor instance
        """
        self.jina = jina_processor
        
        # Bridge detection thresholds
        self.keyword_overlap_threshold = 0.3
        self.attention_threshold = 0.7
        self.bridge_weight = 0.9  # High weight for theory-practice bridges
        
        # Theory indicators
        self.theory_indicators = {
            'paper', 'research', 'theory', 'abstract', 'methodology',
            'hypothesis', 'experiment', 'results', 'conclusion', 'citation'
        }
        
        # Practice indicators
        self.practice_indicators = {
            'implementation', 'code', 'function', 'class', 'api', 
            'example', 'usage', 'tutorial', 'demo', 'test'
        }
        
        logger.info("Initialized Theory-Practice Bridge Detector with Jina v4")
    
    async def detect_bridges(self, 
                           chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect theory-practice bridges in a set of chunks.
        
        Args:
            chunks: List of processed chunks with embeddings and metadata
            
        Returns:
            List of detected bridges with metadata
        """
        bridges = []
        
        # Classify chunks by type
        theory_chunks = []
        practice_chunks = []
        
        for chunk in chunks:
            chunk_type = self._classify_chunk(chunk)
            if chunk_type == 'theory':
                theory_chunks.append(chunk)
            elif chunk_type == 'practice':
                practice_chunks.append(chunk)
        
        # Find bridges between theory and practice chunks
        for theory_chunk in theory_chunks:
            for practice_chunk in practice_chunks:
                bridge = await self._analyze_potential_bridge(
                    theory_chunk, 
                    practice_chunk
                )
                if bridge:
                    bridges.append(bridge)
        
        logger.info(f"Detected {len(bridges)} theory-practice bridges")
        return bridges
    
    def _classify_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Classify a chunk as theory, practice, or other.
        
        Uses metadata and content analysis.
        """
        text = chunk.get('text', '').lower()
        metadata = chunk.get('metadata', {})
        
        # Check file type from metadata
        file_type = metadata.get('file_type', '')
        if file_type in ['pdf', 'paper', 'article']:
            return 'theory'
        elif file_type in ['py', 'js', 'java', 'code']:
            return 'practice'
        
        # Analyze content
        theory_score = sum(1 for word in self.theory_indicators if word in text)
        practice_score = sum(1 for word in self.practice_indicators if word in text)
        
        if theory_score > practice_score:
            return 'theory'
        elif practice_score > theory_score:
            return 'practice'
        else:
            return 'other'
    
    async def _analyze_potential_bridge(self,
                                      theory_chunk: Dict[str, Any],
                                      practice_chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze if two chunks form a theory-practice bridge.
        
        Uses Jina v4's attention mechanisms to identify semantic connections.
        """
        # Extract keywords using Jina v4's attention
        theory_keywords = await self._extract_keywords_with_attention(theory_chunk)
        practice_keywords = await self._extract_keywords_with_attention(practice_chunk)
        
        # Calculate keyword overlap
        common_keywords = theory_keywords.intersection(practice_keywords)
        if not common_keywords:
            return None
        
        overlap_score = len(common_keywords) / min(len(theory_keywords), len(practice_keywords))
        
        if overlap_score < self.keyword_overlap_threshold:
            return None
        
        # Calculate embedding similarity
        similarity = self._calculate_embedding_similarity(
            theory_chunk.get('embeddings'),
            practice_chunk.get('embeddings')
        )
        
        # Determine bridge type
        bridge_type = self._determine_bridge_type(
            theory_chunk, 
            practice_chunk, 
            common_keywords
        )
        
        # Create bridge if criteria met
        if similarity > self.attention_threshold or overlap_score > 0.5:
            return {
                'type': 'theory_practice_bridge',
                'bridge_type': bridge_type.value,
                'from_chunk': theory_chunk['id'],
                'to_chunk': practice_chunk['id'],
                'weight': self.bridge_weight,
                'confidence': (similarity + overlap_score) / 2,
                'metadata': {
                    'common_keywords': list(common_keywords)[:10],
                    'keyword_overlap': overlap_score,
                    'embedding_similarity': similarity,
                    'theory_context': self._extract_context(theory_chunk),
                    'practice_context': self._extract_context(practice_chunk)
                }
            }
        
        return None
    
    async def _extract_keywords_with_attention(self, 
                                             chunk: Dict[str, Any]) -> Set[str]:
        """
        Extract keywords using Jina v4's attention mechanisms.
        
        This replaces the need for a separate code LLM (Devstral).
        """
        # If chunk already has keywords from AST or Jina processing
        if 'keywords' in chunk:
            return set(chunk['keywords'])
        
        # If AST keywords available (for code)
        if 'ast_keywords' in chunk:
            keywords = set()
            ast_kw = chunk['ast_keywords']
            keywords.update(ast_kw.get('symbols', []))
            keywords.update(ast_kw.get('libraries', []))
            return keywords
        
        # Otherwise, use Jina v4's attention to extract keywords
        # This would use the attention weights from the transformer
        # For now, simple extraction as placeholder
        text = chunk.get('text', '')
        words = text.lower().split()
        
        # Extract technical terms (simple heuristic)
        keywords = set()
        for word in words:
            if len(word) > 4 and word.isalnum():
                keywords.add(word)
        
        return keywords
    
    def _calculate_embedding_similarity(self, 
                                      emb1: Any, 
                                      emb2: Any) -> float:
        """Calculate similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Handle both single and multi-vector embeddings
        if isinstance(emb1, list) and isinstance(emb2, list):
            # Average similarity for multi-vector
            similarities = []
            for e1, e2 in zip(emb1[:5], emb2[:5]):  # Compare first 5
                sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                similarities.append(sim)
            return float(np.mean(similarities))
        else:
            # Single vector similarity
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def _determine_bridge_type(self,
                             theory_chunk: Dict[str, Any],
                             practice_chunk: Dict[str, Any],
                             common_keywords: Set[str]) -> BridgeType:
        """Determine the specific type of theory-practice bridge."""
        # Check keywords for patterns
        keywords_lower = {k.lower() for k in common_keywords}
        
        # Implementation bridge
        if any(word in keywords_lower for word in ['implement', 'code', 'function', 'class']):
            return BridgeType.THEORY_TO_IMPLEMENTATION
        
        # Documentation bridge
        elif any(word in keywords_lower for word in ['document', 'guide', 'tutorial', 'example']):
            return BridgeType.DOCUMENTATION_BRIDGE
        
        # Algorithm bridge
        elif any(word in keywords_lower for word in ['algorithm', 'method', 'approach']):
            return BridgeType.ALGORITHMIC_BRIDGE
        
        # Data structure bridge
        elif any(word in keywords_lower for word in ['structure', 'model', 'schema', 'format']):
            return BridgeType.KNOWLEDGE_INSCRIPTION
        
        # Methodology bridge
        elif any(word in keywords_lower for word in ['methodology', 'framework', 'process']):
            return BridgeType.METHODOLOGY_BRIDGE
        
        # Default to concept translation
        else:
            return BridgeType.CONCEPT_TRANSLATION
    
    def _extract_context(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context from a chunk."""
        metadata = chunk.get('metadata', {})
        
        return {
            'chunk_type': metadata.get('chunk_type', 'unknown'),
            'file_path': metadata.get('file_path', ''),
            'line_range': metadata.get('line_range'),
            'symbols': metadata.get('symbol_name') if metadata.get('chunk_type') == 'code_symbol' else None
        }
    
    def create_bridge_edges(self, bridges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert detected bridges into graph edges.
        
        These high-weight edges (0.9) are critical for PathRAG traversal.
        """
        edges = []
        
        for bridge in bridges:
            edge = {
                '_from': f"chunks/{bridge['from_chunk']}",
                '_to': f"chunks/{bridge['to_chunk']}",
                'type': 'theory_practice_bridge',
                'weight': bridge['weight'],
                'metadata': {
                    'bridge_type': bridge['bridge_type'],
                    'confidence': bridge['confidence'],
                    'common_keywords': bridge['metadata']['common_keywords'],
                    'created_by': 'theory_practice_detector',
                    'detection_method': 'jina_v4_attention'
                }
            }
            edges.append(edge)
            
            # Also create reverse edge for bidirectional traversal
            reverse_edge = {
                '_from': edge['_to'],
                '_to': edge['_from'],
                'type': 'theory_practice_bridge',
                'weight': bridge['weight'] * 0.8,  # Slightly lower for reverse
                'metadata': {
                    **edge['metadata'],
                    'is_reverse': True
                }
            }
            edges.append(reverse_edge)
        
        return edges