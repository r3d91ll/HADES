"""
HADES Innovative Concepts

This module contains implementations of advanced concepts that transform HADES
from a simple RAG system into a sophisticated knowledge representation system.

Key Concepts:
- Query as Node: Treating queries as first-class citizens in the knowledge graph
- Theory-Practice Bridges: Semantic connections between research and implementation
- ANT Integration: Actor-Network Theory for heterogeneous knowledge networks
- LCC Classification: Library of Congress Classification for semantic categorization
- Temporal Relationships: Time-bound connections in the knowledge graph
- Supra-Weight Synergies: Multi-dimensional relationship weights
"""

from .query_as_node import QueryAsNode, QueryNode
from .theory_practice_bridge import TheoryPracticeBridgeDetector, BridgeType
from .lcc_classifier import LCCClassifier
from .ant_validator import ANTValidator
from .temporal_relationships import TemporalRelationship
from .supra_weight_calculator import SupraWeightCalculator

__all__ = [
    "QueryAsNode",
    "QueryNode",
    "TheoryPracticeBridgeDetector",
    "BridgeType",
    "LCCClassifier",
    "ANTValidator",
    "TemporalRelationship",
    "SupraWeightCalculator"
]