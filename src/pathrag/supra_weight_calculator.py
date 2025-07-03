"""
Supra-weight calculator for multi-dimensional relationship aggregation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between nodes in the graph."""
    CO_LOCATION = "co_location"      # Same directory
    SEQUENTIAL = "sequential"        # Sequential chunks  
    IMPORT = "import"               # Code dependencies
    SEMANTIC = "semantic"           # Content similarity
    STRUCTURAL = "structural"       # AST-based relationships
    TEMPORAL = "temporal"          # Time-based relationships
    REFERENCE = "reference"        # Documentation references
    THEORY_PRACTICE = "theory_practice"  # Theory-practice bridges (ANT/STS)
    

@dataclass
class Relationship:
    """Single relationship between two nodes."""
    type: RelationType
    strength: float  # [0, 1]
    confidence: float = 1.0  # How certain we are about this relationship
    metadata: Optional[Dict] = None
    
    def __post_init__(self) -> None:
        """Validate relationship values."""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Strength must be in [0, 1], got {self.strength}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class SupraWeightCalculator:
    """
    Calculates aggregated weights from multiple relationships.
    
    This class implements various aggregation strategies for combining
    multiple relationship types into supra-weights.
    """
    
    def __init__(self, method: str = 'adaptive', importance_weights: Optional[Dict[Union[RelationType, str], float]] = None):
        """
        Initialize the calculator.
        
        Args:
            method: Aggregation method ('adaptive', 'weighted_sum', 'max', 'harmonic_mean')
            importance_weights: Optional weights for each relationship type
        """
        self.method = method
        
        # Default importance weights based on empirical observations
        default_weights = {
            RelationType.CO_LOCATION: 1.0,      # Strong signal
            RelationType.IMPORT: 0.95,          # Very strong for code
            RelationType.STRUCTURAL: 0.9,       # Strong for code structure
            RelationType.THEORY_PRACTICE: 0.9,  # Strong for ANT/STS bridges
            RelationType.REFERENCE: 0.85,       # Important for docs
            RelationType.SEMANTIC: 0.7,         # Good but can be noisy
            RelationType.SEQUENTIAL: 0.6,       # Moderate importance
            RelationType.TEMPORAL: 0.5          # Weakest signal
        }
        
        # Handle importance weights - convert string keys to enum if needed
        if importance_weights:
            self.importance_weights = {}
            for key, value in importance_weights.items():
                if isinstance(key, str):
                    # Convert string to enum
                    try:
                        enum_key = RelationType(key)
                        self.importance_weights[enum_key] = value
                    except ValueError:
                        logger.warning(f"Unknown relationship type: {key}")
                else:  # isinstance(key, RelationType)
                    self.importance_weights[key] = value
        else:
            self.importance_weights = default_weights
        
        logger.info(f"Initialized SupraWeightCalculator with method: {method}")
        
    def calculate(self, relationships: List[Relationship]) -> Tuple[float, np.ndarray]:
        """
        Calculate both aggregated weight and weight vector.
        
        Args:
            relationships: List of relationships between two nodes
            
        Returns:
            Tuple of (aggregated_weight, weight_vector)
        """
        if not relationships:
            return 0.0, np.zeros(len(RelationType))
            
        # Create weight vector
        weight_vector = self._create_weight_vector(relationships)
        
        # Calculate aggregated weight based on method
        if self.method == 'adaptive':
            aggregated = self._adaptive_aggregation(relationships, weight_vector)
        elif self.method == 'weighted_sum':
            aggregated = self._weighted_sum_aggregation(relationships, weight_vector)
        elif self.method == 'max':
            aggregated = float(np.max(weight_vector))
        elif self.method == 'harmonic_mean':
            aggregated = self._harmonic_mean_aggregation(weight_vector)
        else:
            # Default to mean
            aggregated = float(np.mean(weight_vector[weight_vector > 0]))
            
        # Ensure aggregated weight is in [0, 1]
        aggregated = np.clip(aggregated, 0.0, 1.0)
        
        return float(aggregated), weight_vector
        
    def _create_weight_vector(self, relationships: List[Relationship]) -> np.ndarray:
        """Create a weight vector from relationships."""
        weight_vector = np.zeros(len(RelationType))
        
        for rel in relationships:
            idx = list(RelationType).index(rel.type)
            # Use maximum if multiple relationships of same type
            current_weight = rel.strength * rel.confidence
            weight_vector[idx] = max(weight_vector[idx], current_weight)
            
        return weight_vector
        
    def _adaptive_aggregation(self, relationships: List[Relationship], 
                            weight_vector: np.ndarray) -> float:
        """
        Adaptive aggregation that considers relationship synergies.
        
        Some relationships amplify each other (e.g., co-location + import),
        while others are redundant.
        """
        rel_types = {r.type for r in relationships}
        
        # Check for synergistic relationships
        has_structural = any(t in rel_types for t in [
            RelationType.CO_LOCATION, 
            RelationType.IMPORT, 
            RelationType.STRUCTURAL
        ])
        has_semantic = RelationType.SEMANTIC in rel_types
        has_reference = RelationType.REFERENCE in rel_types
        
        # Base weight is weighted average
        base_weight = self._weighted_sum_aggregation(relationships, weight_vector)
        
        # Apply synergy bonuses
        synergy_bonus = 0.0
        
        # Code relationships reinforce each other
        if has_structural and has_semantic:
            synergy_bonus += 0.1
            
        # Documentation + code is very strong signal
        if has_reference and has_structural:
            synergy_bonus += 0.15
            
        # Multiple structural relationships indicate strong coupling
        structural_count = sum(1 for r in relationships 
                             if r.type in [RelationType.CO_LOCATION, 
                                         RelationType.IMPORT, 
                                         RelationType.STRUCTURAL])
        if structural_count >= 2:
            synergy_bonus += 0.05 * (structural_count - 1)
            
        return min(base_weight + synergy_bonus, 1.0)
        
    def _weighted_sum_aggregation(self, relationships: List[Relationship], 
                                weight_vector: np.ndarray) -> float:
        """Weighted sum aggregation using importance weights."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for rel in relationships:
            importance = self.importance_weights[rel.type]
            value = rel.strength * rel.confidence
            weighted_sum += importance * value
            total_weight += importance
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def _harmonic_mean_aggregation(self, weight_vector: np.ndarray) -> float:
        """Harmonic mean aggregation (good for averaging rates)."""
        non_zero = weight_vector[weight_vector > 0]
        if len(non_zero) == 0:
            return 0.0
            
        return float(len(non_zero) / np.sum(1.0 / non_zero))
        
    def get_statistics(self, relationships: List[Relationship]) -> Dict:
        """Get statistics about the relationships."""
        if not relationships:
            return {
                'count': 0,
                'types': [],
                'avg_strength': 0.0,
                'avg_confidence': 0.0
            }
            
        return {
            'count': len(relationships),
            'types': list({r.type.value for r in relationships}),
            'avg_strength': np.mean([r.strength for r in relationships]),
            'avg_confidence': np.mean([r.confidence for r in relationships]),
            'strongest_type': max(relationships, key=lambda r: r.strength).type.value
        }