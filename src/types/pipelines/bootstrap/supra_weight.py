"""Type definitions for supra-weight bootstrap pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime


class RelationType(Enum):
    """Types of relationships between nodes in the graph."""
    CO_LOCATION = "co_location"      # Same directory
    SEQUENTIAL = "sequential"        # Sequential chunks  
    IMPORT = "import"               # Code dependencies
    SEMANTIC = "semantic"           # Content similarity
    STRUCTURAL = "structural"       # AST-based relationships
    TEMPORAL = "temporal"          # Time-based relationships
    REFERENCE = "reference"        # Documentation references


@dataclass
class Relationship:
    """Single relationship between two nodes."""
    type: RelationType
    strength: float  # [0, 1]
    confidence: float = 1.0  # How certain we are about this relationship
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate relationship values."""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Strength must be in [0, 1], got {self.strength}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class EdgeCandidate:
    """Candidate edge with its weight and metadata."""
    from_node: str
    to_node: str
    weight: float
    relationships: List[Relationship]
    weight_vector: np.ndarray


class ThermalStatus(Enum):
    """Thermal status levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThermalMetrics:
    """Thermal monitoring metrics."""
    max_temp_c: float
    device_temps: Dict[str, float]
    status: ThermalStatus
    timestamp: float


@dataclass
class WriteBatch:
    """Container for batch write operations."""
    nodes: List[Dict[str, Any]]
    edges: List[EdgeCandidate]
    metadata: Dict[str, Any]
    
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.metadata = {}
        
    def size(self) -> int:
        """Get total size of batch."""
        return len(self.nodes) + len(self.edges)
        
    def clear(self) -> None:
        """Clear the batch."""
        self.nodes.clear()
        self.edges.clear()
        self.metadata.clear()


class TheoryPracticeBridgeType:
    """Types of theory-practice bridges based on ANT/STS principles."""
    THEORY_TO_IMPLEMENTATION = "theory_to_implementation"  # Paper → Code
    DOCUMENTATION_BRIDGE = "documentation_bridge"  # Docs connecting theory & code
    CONCEPT_TRANSLATION = "concept_translation"  # Abstract concept → concrete code
    ACTOR_NETWORK_FORMATION = "actor_network_formation"  # Network assembly
    BLACK_BOX_INSCRIPTION = "black_box_inscription"  # Complex theory → simple API
    SOCIOTECHNICAL_ASSEMBLY = "sociotechnical_assembly"  # Human-machine interface
    KNOWLEDGE_INSCRIPTION = "knowledge_inscription"  # Knowledge → data structure


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    THEORY = "theory"          # Research papers, academic content
    PRACTICE = "practice"      # Code implementations
    DOCUMENTATION = "documentation"  # READMEs, guides, tutorials
    UNKNOWN = "unknown"        # Unclassified


@dataclass
class BridgeStatistics:
    """Statistics about theory-practice bridges."""
    total_bridges: int
    bridge_types: Dict[str, int]
    bridge_strengths: Dict[str, int]
    theory_domains: List[str]


@dataclass
class DensityStatistics:
    """Statistics from density control."""
    edges_proposed: int
    edges_accepted: int
    edges_rejected_weight: int
    edges_rejected_degree: int
    edges_rejected_density: int
    acceptance_rate: float
    weight_rejection_rate: float
    degree_rejection_rate: float
    density_rejection_rate: float
    avg_node_degree: float
    max_node_degree: int
    num_nodes_with_edges: int


@dataclass
class PipelineStatistics:
    """Overall pipeline execution statistics."""
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    nodes_processed: int
    edges_created: int
    relationships_detected: Dict[str, int]
    errors: List[str]
    density_control: Optional[DensityStatistics]
    bridge_statistics: Optional[BridgeStatistics]