"""
Supra-Weight Calculator for Multi-dimensional Relationships

This module implements the calculation of multi-dimensional relationship weights
that capture different aspects of connections and their synergistic effects.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""
    CO_LOCATION = "co_location"              # Same directory
    SEQUENTIAL = "sequential"                # Sequential in filesystem
    IMPORT = "import"                        # Code import relationship
    SEMANTIC = "semantic"                    # Semantic similarity
    STRUCTURAL = "structural"                # Structural similarity
    TEMPORAL = "temporal"                    # Temporal relationship
    REFERENCE = "reference"                  # Direct reference
    THEORY_PRACTICE = "theory_practice"      # Theory-practice bridge
    CROSS_MODAL = "cross_modal"             # Between different modalities
    USER_DEFINED = "user_defined"           # User-specified relationship


@dataclass
class RelationshipWeight:
    """Represents a weight component for a relationship."""
    relation_type: RelationType
    weight: float
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SupraWeightCalculator:
    """
    Calculates multi-dimensional weights for relationships.
    
    Key insight: Some relationships amplify each other (synergies).
    For example: co-location + import is stronger than either alone.
    """
    
    def __init__(self):
        """Initialize the supra-weight calculator."""
        # Base weights for different relationship types
        self.base_weights = {
            RelationType.CO_LOCATION: 0.6,
            RelationType.SEQUENTIAL: 0.5,
            RelationType.IMPORT: 0.8,
            RelationType.SEMANTIC: 0.7,
            RelationType.STRUCTURAL: 0.6,
            RelationType.TEMPORAL: 0.4,
            RelationType.REFERENCE: 0.9,
            RelationType.THEORY_PRACTICE: 0.9,
            RelationType.CROSS_MODAL: 0.7,
            RelationType.USER_DEFINED: 0.5
        }
        
        # Synergy multipliers for relationship combinations
        self.synergies = {
            # Co-location + Import = stronger (files that import each other in same directory)
            (RelationType.CO_LOCATION, RelationType.IMPORT): 1.3,
            
            # Sequential + Semantic = narrative flow
            (RelationType.SEQUENTIAL, RelationType.SEMANTIC): 1.2,
            
            # Theory-Practice + Reference = documented implementation
            (RelationType.THEORY_PRACTICE, RelationType.REFERENCE): 1.4,
            
            # Structural + Semantic = true similarity
            (RelationType.STRUCTURAL, RelationType.SEMANTIC): 1.25,
            
            # Cross-modal + Reference = explicit connection across modes
            (RelationType.CROSS_MODAL, RelationType.REFERENCE): 1.35
        }
        
        logger.info("Initialized Supra-Weight Calculator")
    
    def calculate_supra_weight(self, 
                             relationships: List[RelationshipWeight]) -> Dict[str, Any]:
        """
        Calculate the combined supra-weight from multiple relationship components.
        
        Args:
            relationships: List of relationship weight components
            
        Returns:
            Dictionary with total weight and breakdown
        """
        if not relationships:
            return {
                'total_weight': 0.0,
                'components': {},
                'synergies': [],
                'dominant_type': None
            }
        
        # Calculate base component weights
        components: Dict[RelationType, float] = {}
        for rel in relationships:
            weight = rel.weight * rel.confidence
            if rel.relation_type in components:
                # Take maximum if multiple of same type
                components[rel.relation_type] = max(components[rel.relation_type], weight)
            else:
                components[rel.relation_type] = weight
        
        # Apply synergies
        synergy_bonus = 0.0
        applied_synergies = []
        
        relation_types = list(components.keys())
        for i, type1 in enumerate(relation_types):
            for type2 in relation_types[i+1:]:
                # Check both orderings
                if (type1, type2) in self.synergies:
                    multiplier = self.synergies[(type1, type2)]
                    bonus = (components[type1] + components[type2]) * (multiplier - 1) * 0.5
                    synergy_bonus += bonus
                    applied_synergies.append({
                        'types': (type1.value, type2.value),
                        'multiplier': multiplier,
                        'bonus': bonus
                    })
                elif (type2, type1) in self.synergies:
                    multiplier = self.synergies[(type2, type1)]
                    bonus = (components[type1] + components[type2]) * (multiplier - 1) * 0.5
                    synergy_bonus += bonus
                    applied_synergies.append({
                        'types': (type2.value, type1.value),
                        'multiplier': multiplier,
                        'bonus': bonus
                    })
        
        # Calculate total weight
        base_weight = sum(components.values())
        total_weight = min(base_weight + synergy_bonus, 1.0)  # Cap at 1.0
        
        # Find dominant relationship type
        dominant_type = max(components.items(), key=lambda x: x[1])[0] if components else None
        
        # Convert components to serializable format
        components_dict = {rt.value: weight for rt, weight in components.items()}
        
        return {
            'total_weight': total_weight,
            'base_weight': base_weight,
            'synergy_bonus': synergy_bonus,
            'components': components_dict,
            'synergies': applied_synergies,
            'dominant_type': dominant_type.value if dominant_type else None,
            'num_dimensions': len(components)
        }
    
    def detect_relationships(self,
                           node1: Dict[str, Any],
                           node2: Dict[str, Any],
                           embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[RelationshipWeight]:
        """
        Detect all applicable relationships between two nodes.
        
        Args:
            node1: First node
            node2: Second node  
            embeddings: Optional embeddings for similarity calculation
            
        Returns:
            List of detected relationship weights
        """
        relationships = []
        
        # 1. Co-location check
        co_location = self._check_co_location(node1, node2)
        if co_location:
            relationships.append(co_location)
        
        # 2. Sequential check
        sequential = self._check_sequential(node1, node2)
        if sequential:
            relationships.append(sequential)
        
        # 3. Import relationship (for code)
        import_rel = self._check_import_relationship(node1, node2)
        if import_rel:
            relationships.append(import_rel)
        
        # 4. Semantic similarity
        if embeddings:
            semantic = self._calculate_semantic_similarity(node1, node2, embeddings)
            if semantic:
                relationships.append(semantic)
        
        # 5. Structural similarity
        structural = self._check_structural_similarity(node1, node2)
        if structural:
            relationships.append(structural)
        
        # 6. Reference check
        reference = self._check_reference(node1, node2)
        if reference:
            relationships.append(reference)
        
        # 7. Cross-modal check
        cross_modal = self._check_cross_modal(node1, node2)
        if cross_modal:
            relationships.append(cross_modal)
        
        return relationships
    
    def _check_co_location(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check if nodes are co-located (same directory)."""
        path1 = node1.get('file_path', '')
        path2 = node2.get('file_path', '')
        
        if not path1 or not path2:
            return None
        
        # Get directory paths
        dir1 = '/'.join(path1.split('/')[:-1])
        dir2 = '/'.join(path2.split('/')[:-1])
        
        if dir1 == dir2:
            # Calculate weight based on directory depth (deeper = more specific)
            depth = len(dir1.split('/'))
            weight = self.base_weights[RelationType.CO_LOCATION] * (1 + depth * 0.05)
            
            return RelationshipWeight(
                relation_type=RelationType.CO_LOCATION,
                weight=min(weight, 0.9),
                confidence=1.0,
                metadata={'directory': dir1, 'depth': depth}
            )
        
        return None
    
    def _check_sequential(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check if nodes are sequential in filesystem."""
        path1 = node1.get('file_path', '')
        path2 = node2.get('file_path', '')
        
        if not path1 or not path2:
            return None
        
        # Check if in same directory first
        dir1 = '/'.join(path1.split('/')[:-1])
        dir2 = '/'.join(path2.split('/')[:-1])
        
        if dir1 != dir2:
            return None
        
        # Get filenames
        file1 = path1.split('/')[-1]
        file2 = path2.split('/')[-1]
        
        # Check for sequential patterns
        sequential_patterns = [
            # Numbered sequences
            (r'(\d+)', lambda n1, n2: abs(int(n1) - int(n2)) == 1),
            # Part sequences
            (r'part[_-]?(\d+)', lambda n1, n2: abs(int(n1) - int(n2)) == 1),
            # Chapter sequences
            (r'chapter[_-]?(\d+)', lambda n1, n2: abs(int(n1) - int(n2)) == 1),
        ]
        
        import re
        for pattern, check_func in sequential_patterns:
            match1 = re.search(pattern, file1, re.IGNORECASE)
            match2 = re.search(pattern, file2, re.IGNORECASE)
            
            if match1 and match2:
                if check_func(match1.group(1), match2.group(1)):
                    return RelationshipWeight(
                        relation_type=RelationType.SEQUENTIAL,
                        weight=self.base_weights[RelationType.SEQUENTIAL],
                        confidence=0.9,
                        metadata={'pattern': pattern, 'files': [file1, file2]}
                    )
        
        return None
    
    def _check_import_relationship(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check for import relationships in code."""
        # Check if both are code files
        if node1.get('type') != 'code' or node2.get('type') != 'code':
            return None
        
        # Check AST analysis for imports
        ast1 = node1.get('ast_analysis', {})
        ast2 = node2.get('ast_analysis', {})
        
        imports1 = set(ast1.get('imports', {}).values())
        imports2 = set(ast2.get('imports', {}).values())
        
        file1 = node1.get('file_path', '').replace('.py', '').replace('/', '.')
        file2 = node2.get('file_path', '').replace('.py', '').replace('/', '.')
        
        # Check if node1 imports node2 or vice versa
        if file2 in imports1 or file1 in imports2:
            return RelationshipWeight(
                relation_type=RelationType.IMPORT,
                weight=self.base_weights[RelationType.IMPORT],
                confidence=1.0,
                metadata={
                    'imports': list(imports1 & imports2),
                    'direction': 'bidirectional' if file2 in imports1 and file1 in imports2 else 'unidirectional'
                }
            )
        
        return None
    
    def _calculate_semantic_similarity(self, 
                                     node1: Dict[str, Any], 
                                     node2: Dict[str, Any],
                                     embeddings: Tuple[np.ndarray, np.ndarray]) -> Optional[RelationshipWeight]:
        """Calculate semantic similarity from embeddings."""
        emb1, emb2 = embeddings
        
        # Cosine similarity
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        if similarity > 0.7:  # Threshold for semantic relationship
            return RelationshipWeight(
                relation_type=RelationType.SEMANTIC,
                weight=self.base_weights[RelationType.SEMANTIC] * similarity,
                confidence=similarity,
                metadata={'cosine_similarity': similarity}
            )
        
        return None
    
    def _check_structural_similarity(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check structural similarity (for code)."""
        if node1.get('type') != 'code' or node2.get('type') != 'code':
            return None
        
        ast1 = node1.get('ast_analysis', {})
        ast2 = node2.get('ast_analysis', {})
        
        if not ast1 or not ast2:
            return None
        
        # Compare structure
        structure1 = set(ast1.get('symbols', {}).keys())
        structure2 = set(ast2.get('symbols', {}).keys())
        
        if not structure1 or not structure2:
            return None
        
        # Jaccard similarity of symbols
        intersection = len(structure1 & structure2)
        union = len(structure1 | structure2)
        
        if union > 0:
            similarity = intersection / union
            
            if similarity > 0.3:  # Threshold for structural similarity
                return RelationshipWeight(
                    relation_type=RelationType.STRUCTURAL,
                    weight=self.base_weights[RelationType.STRUCTURAL] * similarity,
                    confidence=similarity,
                    metadata={
                        'jaccard_similarity': similarity,
                        'common_symbols': list(structure1 & structure2)[:5]
                    }
                )
        
        return None
    
    def _check_reference(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check for direct references between nodes."""
        # This would check for explicit references like citations, links, etc.
        # For now, simplified check
        
        content1 = node1.get('content', '').lower()
        content2 = node2.get('content', '').lower()
        
        name1 = node1.get('name', '').lower()
        name2 = node2.get('name', '').lower()
        
        # Check if one references the other by name
        if name1 and name2:
            if name1 in content2 or name2 in content1:
                return RelationshipWeight(
                    relation_type=RelationType.REFERENCE,
                    weight=self.base_weights[RelationType.REFERENCE],
                    confidence=0.9,
                    metadata={'reference_type': 'name_mention'}
                )
        
        return None
    
    def _check_cross_modal(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> Optional[RelationshipWeight]:
        """Check for cross-modal relationships."""
        type1 = node1.get('type', 'unknown')
        type2 = node2.get('type', 'unknown')
        
        # Different types indicate cross-modal
        if type1 != type2 and type1 != 'unknown' and type2 != 'unknown':
            # Special handling for theory-practice
            if {type1, type2} == {'theory', 'practice'} or {type1, type2} == {'documentation', 'code'}:
                return None  # These are handled by theory-practice detector
            
            return RelationshipWeight(
                relation_type=RelationType.CROSS_MODAL,
                weight=self.base_weights[RelationType.CROSS_MODAL],
                confidence=0.8,
                metadata={
                    'modalities': [type1, type2],
                    'cross_type': f"{type1}_to_{type2}"
                }
            )
        
        return None
    
    def create_edge_from_supra_weight(self,
                                    from_node: str,
                                    to_node: str,
                                    supra_weight: Dict[str, Any],
                                    relationships: List[RelationshipWeight]) -> Dict[str, Any]:
        """
        Create a graph edge with supra-weight information.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            supra_weight: Calculated supra-weight
            relationships: Component relationships
            
        Returns:
            Edge dictionary for graph storage
        """
        edge = {
            '_from': from_node,
            '_to': to_node,
            'weight': supra_weight['total_weight'],
            'supra_weight': supra_weight,
            'relationships': [
                {
                    'type': rel.relation_type.value,
                    'weight': rel.weight,
                    'confidence': rel.confidence,
                    'metadata': rel.metadata
                }
                for rel in relationships
            ],
            'metadata': {
                'num_dimensions': supra_weight['num_dimensions'],
                'has_synergy': len(supra_weight['synergies']) > 0,
                'dominant_type': supra_weight['dominant_type']
            }
        }
        
        return edge