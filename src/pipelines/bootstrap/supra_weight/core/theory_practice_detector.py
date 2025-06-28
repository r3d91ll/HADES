"""
Theory-Practice Bridge Detector for ANT/STS enhanced relationships.

This module detects semantic bridges between theoretical research papers,
documentation, and code implementations, following Actor-Network Theory principles.
"""

from typing import Dict, List, Optional, Tuple, Set
import re
import numpy as np
import logging
from pathlib import Path

from .supra_weight_calculator import Relationship, RelationType

logger = logging.getLogger(__name__)


class TheoryPracticeBridgeType:
    """Types of theory-practice bridges based on ANT/STS principles."""
    THEORY_TO_IMPLEMENTATION = "theory_to_implementation"  # Paper → Code
    DOCUMENTATION_BRIDGE = "documentation_bridge"  # Docs connecting theory & code
    CONCEPT_TRANSLATION = "concept_translation"  # Abstract concept → concrete code
    ACTOR_NETWORK_FORMATION = "actor_network_formation"  # Network assembly
    BLACK_BOX_INSCRIPTION = "black_box_inscription"  # Complex theory → simple API
    SOCIOTECHNICAL_ASSEMBLY = "sociotechnical_assembly"  # Human-machine interface
    KNOWLEDGE_INSCRIPTION = "knowledge_inscription"  # Knowledge → data structure


class TheoryPracticeDetector:
    """
    Detects theory-practice bridges following ANT/STS principles.
    
    This detector identifies how theoretical concepts from research papers
    translate into practical code implementations, creating semantic bridges
    that help ISNE understand the relationship between abstract knowledge
    and concrete implementations.
    """
    
    def __init__(self, 
                 semantic_threshold: float = 0.4,
                 cross_domain_threshold: float = 0.6):
        """
        Initialize the detector.
        
        Args:
            semantic_threshold: Minimum similarity for novel bridges (0.4-0.6)
            cross_domain_threshold: Threshold for cross-domain bridges (0.6-0.8)
        """
        self.semantic_threshold = semantic_threshold
        self.cross_domain_threshold = cross_domain_threshold
        
        # Theory indicators (found in papers)
        self.theory_indicators = {
            'actor_network': ['actor-network', 'ANT', 'actant', 'assemblage', 'heterogeneous network'],
            'sts': ['sociotechnical', 'co-construction', 'technological mediation', 'inscription'],
            'complex_systems': ['emergence', 'self-organization', 'adaptive system', 'complexity'],
            'knowledge_rep': ['ontology', 'knowledge graph', 'semantic network', 'representation'],
            'graph_theory': ['graph algorithm', 'network analysis', 'topology', 'connectivity']
        }
        
        # Practice indicators (found in code)
        self.practice_indicators = {
            'implementation': ['class', 'function', 'def', 'implement', 'algorithm'],
            'data_structure': ['dict', 'list', 'array', 'graph', 'node', 'edge'],
            'api': ['interface', 'endpoint', 'method', 'service', 'client'],
            'pipeline': ['process', 'transform', 'pipeline', 'flow', 'stream']
        }
        
        # Documentation bridge indicators
        self.doc_indicators = ['README', 'documentation', 'theory', 'concept', 'explanation']
        
        logger.info("Initialized TheoryPracticeDetector for ANT/STS bridge detection")
        
    def detect_bridges(self, 
                      node_a: Dict, 
                      node_b: Dict,
                      embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[Relationship]:
        """
        Detect theory-practice bridges between nodes.
        
        Args:
            node_a: First node (could be paper, code, or doc)
            node_b: Second node
            embeddings: Optional embeddings for semantic similarity
            
        Returns:
            List of bridge relationships
        """
        bridges: List[Relationship] = []
        
        # Classify nodes
        type_a = self._classify_node(node_a)
        type_b = self._classify_node(node_b)
        
        # Skip if both are same type (except documentation)
        if type_a == type_b and type_a != 'documentation':
            return bridges
            
        # Detect specific bridge types
        if self._is_theory_implementation_bridge(node_a, node_b, type_a, type_b):
            bridges.append(Relationship(
                type=RelationType.REFERENCE,  # Using existing type
                strength=0.9,
                confidence=0.85,
                metadata={
                    'bridge_type': TheoryPracticeBridgeType.THEORY_TO_IMPLEMENTATION,
                    'theory_domain': self._get_theory_domain(node_a if type_a == 'theory' else node_b)
                }
            ))
            
        if self._is_documentation_bridge(node_a, node_b, type_a, type_b):
            bridges.append(Relationship(
                type=RelationType.REFERENCE,
                strength=0.85,
                confidence=0.9,
                metadata={
                    'bridge_type': TheoryPracticeBridgeType.DOCUMENTATION_BRIDGE,
                    'connects': f"{type_a}-{type_b}"
                }
            ))
            
        # Check semantic similarity for concept translation
        if embeddings:
            similarity = self._compute_similarity(embeddings)
            
            # High-confidence bridges (> 0.8)
            if similarity > 0.8 and type_a != type_b:
                bridges.append(Relationship(
                    type=RelationType.SEMANTIC,
                    strength=float(similarity),
                    confidence=0.95,
                    metadata={
                        'bridge_type': TheoryPracticeBridgeType.CONCEPT_TRANSLATION,
                        'bridge_strength': 'high',
                        'similarity': float(similarity)
                    }
                ))
                
            # Cross-domain bridges (0.6-0.8)
            elif similarity > self.cross_domain_threshold:
                bridge_type = self._determine_bridge_type(node_a, node_b, type_a, type_b)
                bridges.append(Relationship(
                    type=RelationType.SEMANTIC,
                    strength=float(similarity),
                    confidence=0.8,
                    metadata={
                        'bridge_type': bridge_type,
                        'bridge_strength': 'cross_domain',
                        'similarity': float(similarity)
                    }
                ))
                
            # Novel discovery bridges (0.4-0.6)
            elif similarity > self.semantic_threshold:
                bridges.append(Relationship(
                    type=RelationType.SEMANTIC,
                    strength=float(similarity),
                    confidence=0.6,
                    metadata={
                        'bridge_type': TheoryPracticeBridgeType.SOCIOTECHNICAL_ASSEMBLY,
                        'bridge_strength': 'novel',
                        'similarity': float(similarity)
                    }
                ))
                
        # Check for actor-network formation
        if self._is_actor_network_formation(node_a, node_b):
            bridges.append(Relationship(
                type=RelationType.STRUCTURAL,
                strength=0.8,
                confidence=0.7,
                metadata={
                    'bridge_type': TheoryPracticeBridgeType.ACTOR_NETWORK_FORMATION,
                    'network_type': 'heterogeneous'
                }
            ))
            
        return bridges
        
    def _classify_node(self, node: Dict) -> str:
        """Classify node as theory, practice, or documentation."""
        content = node.get('content', '').lower()
        file_path = node.get('file_path', '').lower()
        file_name = node.get('file_name', '').lower()
        
        # Check if documentation
        if any(indicator in file_name for indicator in ['readme', 'doc', 'documentation']):
            return 'documentation'
            
        # Check if theory (paper)
        theory_score = sum(
            content.count(term) 
            for terms in self.theory_indicators.values() 
            for term in terms
        )
        
        # Check if practice (code)
        practice_score = sum(
            content.count(term)
            for terms in self.practice_indicators.values()
            for term in terms
        )
        
        # Also check file extension
        if file_path.endswith('.pdf') or 'paper' in file_path:
            theory_score += 10
        elif file_path.endswith(('.py', '.js', '.java', '.cpp')):
            practice_score += 10
            
        if theory_score > practice_score:
            return 'theory'
        elif practice_score > theory_score:
            return 'practice'
        else:
            return 'unknown'
            
    def _is_theory_implementation_bridge(self, 
                                       node_a: Dict, 
                                       node_b: Dict,
                                       type_a: str,
                                       type_b: str) -> bool:
        """Check if nodes form a theory-implementation bridge."""
        # One must be theory, other practice
        if not ((type_a == 'theory' and type_b == 'practice') or 
                (type_a == 'practice' and type_b == 'theory')):
            return False
            
        theory_node = node_a if type_a == 'theory' else node_b
        practice_node = node_b if type_a == 'theory' else node_a
        
        # Check for concept overlap
        theory_content = theory_node.get('content', '').lower()
        practice_content = practice_node.get('content', '').lower()
        
        # Look for shared technical terms
        theory_terms = set(re.findall(r'\b\w+(?:_\w+)*\b', theory_content))
        practice_terms = set(re.findall(r'\b\w+(?:_\w+)*\b', practice_content))
        
        overlap = theory_terms & practice_terms
        
        # Check for significant overlap (excluding common words)
        significant_terms = {term for term in overlap if len(term) > 4}
        
        return len(significant_terms) > 5
        
    def _is_documentation_bridge(self,
                               node_a: Dict,
                               node_b: Dict,
                               type_a: str,
                               type_b: str) -> bool:
        """Check if documentation bridges theory and practice."""
        # At least one should be documentation
        if type_a != 'documentation' and type_b != 'documentation':
            return False
            
        # Documentation often references both theory and implementation
        doc_node = node_a if type_a == 'documentation' else node_b
        doc_content = doc_node.get('content', '').lower()
        
        # Check for theory references
        has_theory_ref = any(
            term in doc_content 
            for terms in self.theory_indicators.values() 
            for term in terms
        )
        
        # Check for practice references
        has_practice_ref = any(
            term in doc_content
            for terms in self.practice_indicators.values()
            for term in terms
        )
        
        return has_theory_ref and has_practice_ref
        
    def _is_actor_network_formation(self, node_a: Dict, node_b: Dict) -> bool:
        """Detect actor-network formation patterns."""
        # Look for network assembly indicators
        content_a = node_a.get('content', '').lower()
        content_b = node_b.get('content', '').lower()
        
        # Actor-network keywords
        ant_keywords = ['actor', 'network', 'assemblage', 'heterogeneous', 'actant']
        
        # Check if nodes participate in network formation
        has_ant_a = any(keyword in content_a for keyword in ant_keywords)
        has_ant_b = any(keyword in content_b for keyword in ant_keywords)
        
        # Also check for import/dependency relationships indicating network
        has_import = 'import' in content_a or 'import' in content_b
        
        return (has_ant_a or has_ant_b) and has_import
        
    def _get_theory_domain(self, theory_node: Dict) -> str:
        """Identify the theoretical domain of a node."""
        content = theory_node.get('content', '').lower()
        
        domain_scores = {}
        for domain, keywords in self.theory_indicators.items():
            score = sum(content.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
            
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=lambda k: domain_scores[k])
        return 'unknown'
        
    def _determine_bridge_type(self, 
                             node_a: Dict, 
                             node_b: Dict,
                             type_a: str,
                             type_b: str) -> str:
        """Determine specific bridge type based on content analysis."""
        content_a = node_a.get('content', '').lower()
        content_b = node_b.get('content', '').lower()
        
        # Check for knowledge inscription patterns
        if ('data' in content_a and 'structure' in content_b) or \
           ('structure' in content_a and 'data' in content_b):
            return TheoryPracticeBridgeType.KNOWLEDGE_INSCRIPTION
            
        # Check for black-boxing patterns
        if ('complex' in content_a and 'simple' in content_b) or \
           ('interface' in content_b and 'theory' in content_a):
            return TheoryPracticeBridgeType.BLACK_BOX_INSCRIPTION
            
        # Default to actor-network formation
        return TheoryPracticeBridgeType.ACTOR_NETWORK_FORMATION
        
    def _compute_similarity(self, embeddings: Tuple[np.ndarray, np.ndarray]) -> float:
        """Compute cosine similarity between embeddings."""
        emb_a, emb_b = embeddings
        
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        
    def get_bridge_statistics(self, relationships: List[Relationship]) -> Dict:
        """Get statistics about detected bridges."""
        bridge_types: Dict[str, int] = {}
        bridge_strengths = {'high': 0, 'cross_domain': 0, 'novel': 0}
        
        for rel in relationships:
            if rel.metadata and 'bridge_type' in rel.metadata:
                bridge_type = rel.metadata['bridge_type']
                bridge_types[bridge_type] = bridge_types.get(bridge_type, 0) + 1
                
            if rel.metadata and 'bridge_strength' in rel.metadata:
                strength = rel.metadata['bridge_strength']
                bridge_strengths[strength] += 1
                
        return {
            'total_bridges': len([r for r in relationships if r.metadata and 'bridge_type' in r.metadata]),
            'bridge_types': bridge_types,
            'bridge_strengths': bridge_strengths,
            'theory_domains': list(set(
                r.metadata.get('theory_domain', 'unknown') 
                for r in relationships 
                if r.metadata and 'theory_domain' in r.metadata
            ))
        }