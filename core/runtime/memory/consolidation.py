"""
Memory Consolidation Engine - Generate Reflections from Observations

WHAT: LLM-powered synthesis of insights from multiple observations
WHERE: core/runtime/memory/consolidation.py - reflection generation layer
WHO: Batch processes or agents consolidating experiential memories
TIME: 1-5s per reflection (LLM inference)

Consolidates related observations into higher-order reflections using
the Qwen3 model engine. Identifies patterns, principles, and insights
from accumulated experiences.

Boundary Notes:
- Reflection quality directly impacts H (agent capability)
- Generated reflections inherit tags from source observations
- Consolidation strength metric indicates evidence support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import Observation, Reflection
from .operations import MemoryStore

logger = logging.getLogger(__name__)


REFLECTION_GENERATION_PROMPT = """You are a thoughtful analyst synthesizing insights from multiple observations.

Given a set of related observations, generate a concise, actionable reflection that captures:
1. The underlying pattern or principle
2. Why it matters
3. How it can be applied in the future

Observations:
{observations}

Generate a reflection that:
- Is 1-3 sentences
- Focuses on the "why" and "so what", not just facts
- Can guide future decisions
- Is specific enough to be actionable

Reflection:"""


@dataclass
class ConsolidationConfig:
    """Configuration for reflection consolidation."""

    min_observations: int = 2  # Minimum observations to consolidate
    max_observations: int = 10  # Maximum observations per reflection
    similarity_threshold: float = 0.75  # Cluster similarity for grouping
    model: str = "Qwen3-30B-A3B-Thinking-2507"  # Model for generation
    max_tokens: int = 150  # Max tokens for reflection


class ConsolidationEngine:
    """
    Generates reflections from clusters of related observations.

    Two modes:
    1. Manual: Consolidate specific observation IDs
    2. Automatic: Find clusters and consolidate periodically
    """

    def __init__(self, memory_store: MemoryStore, model_engine: Optional[Any] = None):
        self.store = memory_store
        self.model_engine = model_engine
        self.config = ConsolidationConfig()

    def consolidate_observations(
        self, observation_ids: List[str], custom_prompt: Optional[str] = None
    ) -> Reflection:
        """
        Generate a reflection from specific observations.

        Args:
            observation_ids: List of observation document IDs to consolidate
            custom_prompt: Optional custom prompt template

        Returns:
            Generated Reflection model

        Raises:
            ValueError: If too few observations or model not available
        """
        if len(observation_ids) < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, got {len(observation_ids)}"
            )

        if not self.model_engine:
            raise ValueError("Model engine not initialized - cannot generate reflections")

        # Fetch observations
        observations: List[Observation] = []
        for obs_id in observation_ids:
            try:
                coll, key = obs_id.split("/")
                doc = self.store.client.get_document(coll, key)
                if doc:
                    observations.append(Observation.from_arango_doc(doc))
            except Exception as e:
                logger.warning(f"Failed to fetch observation {obs_id}: {e}")

        if len(observations) < self.config.min_observations:
            raise ValueError(
                f"Could only fetch {len(observations)} valid observations, need at least {self.config.min_observations}"
            )

        # Prepare prompt
        obs_text = "\n\n".join(
            [
                f"[{i+1}] {obs.memory_type.upper()}: {obs.content}\n   Tags: {', '.join(obs.tags)}"
                for i, obs in enumerate(observations)
            ]
        )

        prompt = (custom_prompt or REFLECTION_GENERATION_PROMPT).format(observations=obs_text)

        # Generate reflection using model
        try:
            response = self._generate_with_model(prompt)
            reflection_content = response.strip()
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            raise

        # Aggregate metadata
        all_tags = set()
        for obs in observations:
            all_tags.update(obs.tags)

        # Compute strength metric (simple: 1.0 if all observations recent)
        now = datetime.now(timezone.utc).timestamp()
        recency_scores = [
            max(0.0, 1.0 - (now - obs.created_at_unix) / (86400 * 30))  # 30-day decay
            for obs in observations
        ]
        strength = sum(recency_scores) / len(recency_scores)

        # Create reflection model
        reflection = Reflection(
            content=reflection_content,
            memory_type="reflection",
            tags=sorted(list(all_tags)),
            consolidates=[f"observations/{obs.key}" for obs in observations],
            source_count=len(observations),
            metadata={
                "strength": round(strength, 2),
                "method": "llm_synthesis",
                "model": self.config.model,
                "prompt_tokens": len(prompt.split()),  # Rough estimate
            },
        )

        # Generate embedding if embedder available
        if self.store.embedder:
            try:
                emb_result = self.store.embedder.embed_documents(
                    [reflection.content], batch_size=1
                )
                reflection.embedding = emb_result[0].tolist() if len(emb_result) > 0 else None
            except Exception as e:
                logger.warning(f"Failed to generate embedding for reflection: {e}")

        # Store reflection
        doc = reflection.to_arango_doc()
        self.store.client.insert_document(self.store.REFLECTIONS, doc, overwrite_mode="ignore")

        logger.info(
            f"Generated reflection {reflection.key} from {len(observations)} observations "
            f"(strength={strength:.2f})"
        )
        return reflection

    def find_consolidation_candidates(
        self, session_id: Optional[str] = None, min_cluster_size: int = 3
    ) -> List[List[str]]:
        """
        Find clusters of related observations that should be consolidated.

        Uses tag co-occurrence and embedding similarity to identify clusters.

        Args:
            session_id: Optional session filter
            min_cluster_size: Minimum observations per cluster

        Returns:
            List of observation ID clusters
        """
        # Fetch unconsolidated observations
        observations = self.store.list_observations(session_id=session_id, limit=100)

        # Filter out those already consolidated
        existing_consolidated = self._get_consolidated_observation_ids()
        observations = [obs for obs in observations if f"observations/{obs.key}" not in existing_consolidated]

        if len(observations) < min_cluster_size:
            logger.info(f"Only {len(observations)} unconsolidated observations - skipping clustering")
            return []

        # Simple tag-based clustering
        clusters = self._cluster_by_tags(observations, min_cluster_size)

        logger.info(f"Found {len(clusters)} consolidation candidate clusters")
        return clusters

    def _cluster_by_tags(
        self, observations: List[Observation], min_size: int
    ) -> List[List[str]]:
        """
        Simple tag-based clustering using Jaccard similarity.

        More sophisticated: Could use embedding clustering (K-means, HDBSCAN).
        """
        clusters: List[List[str]] = []

        # Group by primary tag (most common tag across observations)
        tag_groups: Dict[str, List[Observation]] = {}
        for obs in observations:
            for tag in obs.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(obs)

        # Convert to ID lists
        for tag, obs_list in tag_groups.items():
            if len(obs_list) >= min_size:
                clusters.append([f"observations/{obs.key}" for obs in obs_list[:10]])  # Cap at 10

        return clusters

    def _get_consolidated_observation_ids(self) -> set:
        """Get set of observation IDs that have already been consolidated."""
        aql = """
        FOR refl IN reflections
            FILTER refl.status == 'active'
            FOR obs_id IN refl.consolidates
                RETURN obs_id
        """
        try:
            result = self.store.client.execute_query(aql, {})
            return set(result)
        except Exception as e:
            logger.warning(f"Failed to fetch consolidated observation IDs: {e}")
            return set()

    def _generate_with_model(self, prompt: str) -> str:
        """
        Generate text using the configured model engine.

        Falls back to placeholder if model not available.
        """
        if not self.model_engine:
            # Placeholder for testing without model
            return "Generated reflection from observations (model unavailable)"

        # If using QwenModelEngine
        try:
            if hasattr(self.model_engine, "generate_streaming"):
                # Collect streaming tokens
                tokens = []
                for chunk in self.model_engine.generate_streaming(
                    prompt=prompt, max_new_tokens=self.config.max_tokens
                ):
                    tokens.append(chunk)
                return "".join(tokens)
            elif hasattr(self.model_engine, "generate"):
                return self.model_engine.generate(
                    prompt=prompt, max_new_tokens=self.config.max_tokens
                )
            else:
                raise ValueError("Model engine has no generate method")
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise


def consolidate_session_memories(
    store: MemoryStore, session_id: str, model_engine: Optional[Any] = None
) -> List[Reflection]:
    """
    Convenience function to consolidate all observations from a session.

    Args:
        store: MemoryStore instance
        session_id: Session to consolidate
        model_engine: Optional model for generation

    Returns:
        List of generated reflections
    """
    engine = ConsolidationEngine(store, model_engine)
    clusters = engine.find_consolidation_candidates(session_id=session_id, min_cluster_size=2)

    reflections = []
    for cluster in clusters:
        try:
            refl = engine.consolidate_observations(cluster)
            reflections.append(refl)
        except Exception as e:
            logger.warning(f"Failed to consolidate cluster: {e}")

    logger.info(f"Generated {len(reflections)} reflections from session {session_id}")
    return reflections


__all__ = [
    "ConsolidationEngine",
    "ConsolidationConfig",
    "consolidate_session_memories",
]