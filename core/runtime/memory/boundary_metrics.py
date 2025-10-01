"""
Boundary Metrics - Conveyance Measurement at User↔Agent Interface

WHAT: Calculates W·R·H/T conveyance metrics at conversation boundaries
WHERE: core/runtime/memgpt/boundary_metrics.py - measurement layer
WHO: MemGPT orchestrator collecting metrics for analysis
TIME: Metric computation <1ms per turn

Implements the Conveyance Framework measurement protocol, computing metrics
ONLY at user↔agent boundaries, not internal operations. Bridges turn manager
data with conveyance logging infrastructure.

Boundary Notes:
- Primary implementation of boundary-only measurement principle
- Computes W (semantic), R (structure), H (capability), T (latency)
- Context amplification α estimated from historical data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logging.conveyance import ConveyanceContext, compute_pair_conveyance


@dataclass(slots=True)
class BoundaryInputs:
    """Inputs required to compute pair-level conveyance for a single turn."""

    W_out: float
    R_encode: float
    H_out: float
    T_out: float
    W_in: float
    R_decode: float
    H_in: float
    T_in: float
    C_ext: ConveyanceContext
    P_ij: float = 1.0
    alpha: float = 1.7
    view: str = "efficiency"


class BoundaryMetricsCalculator:
    """Thin wrapper to compute conveyance metrics for orchestration telemetry."""

    def build_payload(self, label: str, inputs: BoundaryInputs) -> dict[str, Any]:
        context_value = inputs.C_ext.weighted_sum()
        result = compute_pair_conveyance(
            W_out=inputs.W_out,
            R_encode=inputs.R_encode,
            H_out=inputs.H_out,
            T_out=inputs.T_out,
            W_in=inputs.W_in,
            R_decode=inputs.R_decode,
            H_in=inputs.H_in,
            T_in=inputs.T_in,
            C_ext=context_value,
            P_ij=inputs.P_ij,
            alpha=inputs.alpha,
            view=inputs.view,
        )
        payload = {
            "label": label,
            "context": inputs.C_ext.as_dict(),
            "P_ij": inputs.P_ij,
            "alpha": inputs.alpha,
        }
        payload.update(result)
        return payload
