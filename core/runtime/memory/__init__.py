"""
Experiential Memory System - Observations, Reflections & Knowledge Graph

WHAT: Local library for experiential memory operations (no network services)
WHERE: core/runtime/memory/ - runtime orchestration subsystem
WHO: Agents storing observations, reflections, and building knowledge graphs
TIME: User-agent latency target <100ms avg, recall p99 ≤250ms

Local Python library implementing experiential memory architecture.
Prompt engineering patterns inspired by:
https://github.com/doobidoo/mcp-memory-service

Infrastructure powered by HADES stack:
- ArangoDB graph database (Unix socket connections)
- PathRAG retrieval with graph traversal
- Jina v4 embeddings (32k context, 2048 dimensions)

Memory Types:
- observations: Raw experiences/events witnessed by the agent
- reflections: Higher-order insights synthesized from observations
- entities: People/concepts/components extracted from experiences
- relationships: Graph edges connecting entities and memories

Operations (local library - no network services):
- store_observation(content, tags, type): Persist experiential memories
- retrieve_memories(query, filters): PathRAG-powered semantic search
- search_by_tag(tags): Structured category-based retrieval
- list_observations(): List recent memories with pagination
- archive_observation(key): Soft delete memories

Consolidation (separate engine):
- consolidate_observations(ids): Generate reflections from observations
- find_consolidation_candidates(): Auto-detect related memories

Prompt engineering patterns inspired by:
https://github.com/doobidoo/mcp-memory-service

Boundary Notes:
- Package boundary defines the agent capability (H) scope
- All submodules contribute to overall conveyance C = W·R·H/T
- Protocol compatibility P_ij enforced at package interfaces
- Memory operations tracked for conveyance analysis
"""

from .boundary_metrics import BoundaryMetricsCalculator  # noqa: F401
from .model_engine import (  # noqa: F401
    MissingDependencyError,
    ModelNotLoadedError,
    QwenModelConfig,
    QwenModelEngine,
)
from .orchestrator import MemGPTOrchestrator, OrchestratorConfig, PromptEnvelope  # noqa: F401
from .telemetry import (  # noqa: F401
    ConsoleTelemetryClient,
    NoOpTelemetryClient,
    TelemetryClient,
    TelemetrySpan,
)
from .turn_manager import ConversationTurn, TurnManager  # noqa: F401

__all__ = [
    "BoundaryMetricsCalculator",
    "ModelNotLoadedError",
    "MissingDependencyError",
    "MemGPTOrchestrator",
    "OrchestratorConfig",
    "PromptEnvelope",
    "TelemetryClient",
    "NoOpTelemetryClient",
    "ConsoleTelemetryClient",
    "TelemetrySpan",
    "QwenModelConfig",
    "QwenModelEngine",
    "ConversationTurn",
    "TurnManager",
]
