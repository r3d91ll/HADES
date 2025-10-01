"""
Telemetry Collection - Runtime Performance Monitoring

WHAT: Lightweight telemetry for tracking runtime performance and events
WHERE: core/runtime/memgpt/telemetry.py - observability layer
WHO: All runtime components emitting performance data
TIME: Zero-overhead when disabled, <0.1ms overhead when enabled

Provides telemetry interfaces for Phoenix/Arize export and local analysis.
Supports both no-op and active collection modes based on configuration.

Boundary Notes:
- Telemetry events include boundary markers for conveyance analysis
- Stage-level metrics (reads_stage, latency_stage) for optimization
- Supports α learning through historical data collection
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Optional


class TelemetrySpan(AbstractContextManager["TelemetrySpan"]):
    """Context manager capturing span metadata and duration."""

    def __init__(
        self,
        client: "TelemetryClient",
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self.name = name
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self._start: float = 0.0

    def __enter__(self) -> "TelemetrySpan":
        self._start = time.perf_counter()
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        """Update span attributes while running."""

        self.attributes[key] = value

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        self.attributes.setdefault("success", exc is None)
        self.attributes["duration_ms"] = duration_ms
        self._client.emit_span(self.name, self.attributes)
        return False


class TelemetryClient:
    """Base telemetry client; override `emit_span` for custom sinks."""

    def span(
        self,
        name: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TelemetrySpan:
        return TelemetrySpan(self, name, attributes)

    def emit_span(self, name: str, attributes: Dict[str, Any]) -> None:
        """Handle span completion. Subclasses override this hook."""

        raise NotImplementedError


@dataclass(slots=True)
class NoOpTelemetryClient(TelemetryClient):
    """Telemetry client that silently discards spans."""

    def emit_span(self, name: str, attributes: Dict[str, Any]) -> None:  # noqa: D401 - intentionally empty
        pass


class ConsoleTelemetryClient(TelemetryClient):
    """Simple client that prints spans for debugging."""

    def emit_span(self, name: str, attributes: Dict[str, Any]) -> None:
        payload = {k: attributes[k] for k in sorted(attributes)}
        print(f"[telemetry] {name}: {payload}")
