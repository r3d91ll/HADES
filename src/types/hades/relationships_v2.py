"""
Enhanced .hades/relationships.json schema for theory-practice bridges.

This module defines the v2 schema that includes explicit bridges between
theoretical concepts and practical implementations.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class SourceTarget(BaseModel):
    """Source or target in a theory-practice bridge."""
    type: Literal["research_paper", "code", "documentation", "configuration", "notebook"]
    path: str = Field(..., description="Relative or absolute path to the resource")
    section: Optional[str] = Field(None, description="Section/heading for documents")
    symbol: Optional[str] = Field(None, description="Symbol name for code")
    lines: Optional[List[int]] = Field(None, description="Line range [start, end] for code")


class TheoryPracticeBridge(BaseModel):
    """Bridge between theory and practice."""
    source: SourceTarget
    target: SourceTarget
    relationship: Literal[
        "implements",       # Code implements theoretical concept
        "algorithm_of",     # Direct algorithm implementation
        "based_on",        # Inspired by or adapted from
        "documented_in",   # Where this is documented
        "api_reference",   # API documentation
        "tutorial_in",     # Tutorial featuring this
        "cites",          # Academic citation
        "references",      # General reference
        "extends",        # Extension of existing work
        "used_by",        # Code that uses this
        "example_in",     # Example usage
        "tested_in",      # Test coverage
        "evaluates",      # Evaluation/benchmark of theory
        "proves",         # Mathematical proof or verification
    ]
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence score")
    bidirectional: bool = Field(False, description="Whether relationship goes both ways")
    notes: Optional[str] = Field(None, description="Additional context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra metadata")


class SemanticRelationship(BaseModel):
    """Traditional semantic relationships between directories/files."""
    source: str = Field("current_directory", description="Source path (current_directory for self)")
    target: str = Field(..., description="Target path")
    type: Literal[
        "depends_on",
        "integrates_with",
        "extends",
        "implements_interface",
        "provides_service",
        "consumes_service",
        "tests",
        "documents",
        "configures"
    ]
    strength: float = Field(0.7, ge=0.0, le=1.0)
    description: Optional[str] = None


class HadesRelationshipsV2(BaseModel):
    """Enhanced .hades/relationships.json schema."""
    version: Literal["2.0"] = "2.0"
    
    # Theory-practice bridges
    bridges: Dict[str, List[TheoryPracticeBridge]] = Field(
        default_factory=lambda: {"theory_practice": []},  # type: ignore
        description="Theory-practice bridge relationships"
    )
    
    # Traditional semantic relationships
    semantic: List[SemanticRelationship] = Field(
        default_factory=list,
        description="Semantic relationships between components"
    )
    
    # Auto-discovered relationships (populated by system)
    discovered: Optional[Dict[str, Any]] = Field(
        None,
        description="System-discovered relationships"
    )
    
    # Metadata about this relationships file
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata about relationship definitions"
    )


# Example usage
example_relationships_v2 = {
    "version": "2.0",
    "bridges": {
        "theory_practice": [
            {
                "source": {
                    "type": "research_paper",
                    "path": "research/jina-embeddings-v4.pdf",
                    "section": "3.2 Late Chunking"
                },
                "target": {
                    "type": "code",
                    "path": "src/jina_v4/jina_processor.py",
                    "symbol": "_perform_late_chunking",
                    "lines": [493, 614]
                },
                "relationship": "implements",
                "confidence": 0.95,
                "notes": "Direct implementation of late chunking algorithm from paper"
            },
            {
                "source": {
                    "type": "code",
                    "path": "src/pathrag/pathrag_rag_strategy.py",
                    "symbol": "PathRAG"
                },
                "target": {
                    "type": "documentation",
                    "path": "docs/concepts/CORE_CONCEPTS.md",
                    "section": "PathRAG Algorithm"
                },
                "relationship": "documented_in",
                "bidirectional": True,
                "confidence": 1.0
            },
            {
                "source": {
                    "type": "code",
                    "path": "src/isne/models/isne_model.py",
                    "symbol": "ISNEModel"
                },
                "target": {
                    "type": "research_paper",
                    "path": "research/isne_paper.pdf",
                    "section": "Section 4: Method"
                },
                "relationship": "algorithm_of",
                "confidence": 0.9,
                "notes": "Implementation follows paper's skip-gram approach"
            }
        ]
    },
    "semantic": [
        {
            "source": "current_directory",
            "target": "/src/storage/arangodb",
            "type": "depends_on",
            "strength": 0.9,
            "description": "Requires ArangoDB for graph storage"
        },
        {
            "source": "current_directory",
            "target": "/src/types",
            "type": "implements_interface",
            "strength": 0.8
        }
    ],
    "metadata": {
        "created": "2024-01-20",
        "last_updated": "2024-01-20",
        "auto_discover": True,
        "discovery_rules": [
            "match_algorithm_names",
            "detect_citations",
            "analyze_imports"
        ]
    }
}