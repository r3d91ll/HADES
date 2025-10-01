"""
MemGPT Orchestrator - Central Coordination Point

WHAT: High-level orchestrator tying turns, model engine, and telemetry
WHERE: core/runtime/memgpt/orchestrator.py - top of the runtime stack
WHO: Entry point for all MemGPT agent interactions
TIME: End-to-end latency target <100ms avg for user-agent exchanges

Central coordination point that ties together conversation management,
model generation, and telemetry collection. This is where conveyance
metrics are measured at the user↔agent boundary.

Boundary Notes:
- Primary measurement point for W·R·H/T conveyance calculation
- Aggregates metrics from all subsystems
- Enforces protocol compatibility checks (P_ij)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .boundary_metrics import BoundaryInputs, BoundaryMetricsCalculator
from .model_engine import QwenModelConfig, QwenModelEngine
from .telemetry import NoOpTelemetryClient, TelemetryClient
from .turn_manager import ConversationTurn, TurnManager


@dataclass(slots=True)
class OrchestratorConfig:
    max_turns: int = 100
    token_budget: Optional[int] = 8192
    auto_load_model: bool = False


@dataclass(slots=True)
class PromptEnvelope:
    """Encapsulates a prepared model prompt and generation overrides."""

    prompt: str
    stops: tuple[str, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemGPTOrchestrator:
    """Facade that coordinates conversation turns, model inference, and metrics."""

    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        model_config: QwenModelConfig | None = None,
        turn_manager: TurnManager | None = None,
        model_engine: QwenModelEngine | None = None,
        metrics_calculator: BoundaryMetricsCalculator | None = None,
        telemetry: TelemetryClient | None = None,
    ) -> None:
        cfg = config or OrchestratorConfig()
        self._turn_manager = turn_manager or TurnManager(
            max_turns=cfg.max_turns,
            token_budget=cfg.token_budget,
        )
        self._model_engine = model_engine or QwenModelEngine(model_config)
        self._metrics_calculator = metrics_calculator or BoundaryMetricsCalculator()
        self._telemetry = telemetry or NoOpTelemetryClient()
        if cfg.auto_load_model and not self._model_engine.is_loaded():
            self._model_engine.load()

    @property
    def turn_manager(self) -> TurnManager:
        return self._turn_manager

    @property
    def model_engine(self) -> QwenModelEngine:
        return self._model_engine

    @property
    def metrics_calculator(self) -> BoundaryMetricsCalculator:
        return self._metrics_calculator

    @property
    def telemetry(self) -> TelemetryClient:
        return self._telemetry

    def ingest_turn(
        self,
        *,
        role: str,
        content: str,
        tokens: int | None = None,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        turn = ConversationTurn.create(role=role, content=content, tokens=tokens, metadata=metadata)
        self._turn_manager.add_turn(turn)
        return turn

    def build_prompt(self, turns: Iterable[ConversationTurn] | None = None) -> str:
        snapshot = turns if turns is not None else self._turn_manager.turns
        return self._model_engine.build_prompt(snapshot)

    def prepare_envelope(
        self,
        *,
        turns: Iterable[ConversationTurn] | None = None,
        stop: Iterable[str] | None = None,
        overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptEnvelope:
        prompt = self.build_prompt(turns)
        stops = tuple(stop or ())
        return PromptEnvelope(
            prompt=prompt,
            stops=stops,
            overrides=dict(overrides or {}),
            metadata=metadata or {},
        )

    def _merge_overrides(
        self,
        envelope: PromptEnvelope,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        merged = dict(envelope.overrides)
        if extra:
            merged.update(extra)
        if envelope.stops and "stop" not in merged:
            merged["stop"] = envelope.stops
        return merged

    def generate_turn(self, envelope: PromptEnvelope | None = None, **overrides: Any) -> ConversationTurn:
        env = envelope or self.prepare_envelope()
        merged_overrides = self._merge_overrides(env, overrides)
        turn_summary = self._turn_manager.summarize()
        span_attributes = {
            "prompt_chars": len(env.prompt),
            "stop_count": len(env.stops),
            "turn_count": turn_summary.get("turn_count"),
            "token_budget": turn_summary.get("token_budget"),
            "override_keys": sorted(merged_overrides.keys()),
        }

        with self._telemetry.span(
            "memgpt.model_generate",
            attributes=span_attributes,
        ) as span:
            reply_text = self._model_engine.generate(env.prompt, **merged_overrides)
            span.set_attribute("response_chars", len(reply_text))

        reply_turn = ConversationTurn.create(
            role="assistant",
            content=reply_text,
            metadata=dict(env.metadata),
        )
        self._turn_manager.add_turn(reply_turn)
        return reply_turn

    def generate_reply(self, **overrides: Any) -> ConversationTurn:
        return self.generate_turn(**overrides)

    def compute_boundary_payload(self, *, label: str, inputs: BoundaryInputs) -> dict[str, object]:
        return self._metrics_calculator.build_payload(label, inputs)
