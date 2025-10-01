"""Utilities for recording Conveyance-aligned benchmark metrics.

This module provides two layers of helpers:

1) Legacy single-sided helpers based on the original formulation
   C = (W·R·H/T) · Ctx^alpha, which are still useful for simple
   single-agent measurements derived from benchmark JSON artefacts.

2) Dyadic (pair-level) helpers that align with the updated framework,
   combining outbound and inbound bottlenecks via a harmonic mean and
   applying the super-linear amplifier to shared external context only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

TIME_UNITS: Mapping[str, float] = {
    "seconds": 1.0,
    "s": 1.0,
    "milliseconds": 1e3,
    "ms": 1e3,
    "microseconds": 1e6,
    "us": 1e6,
}


@dataclass(frozen=True)
class ConveyanceContext:
    """Context (C_ext) components and weights for the Conveyance calculation."""

    L: float
    I: float
    A: float
    G: float
    weight_L: float = 0.25
    weight_I: float = 0.25
    weight_A: float = 0.25
    weight_G: float = 0.25

    def weighted_sum(self) -> float:
        """Return the scalar C_ext value (wL·L + wI·I + wA·A + wG·G)."""

        return (
            self.weight_L * self.L
            + self.weight_I * self.I
            + self.weight_A * self.A
            + self.weight_G * self.G
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the context components, weights, and total C_ext (plus legacy alias)."""

        return {
            "components": {"L": self.L, "I": self.I, "A": self.A, "G": self.G},
            "weights": {
                "L": self.weight_L,
                "I": self.weight_I,
                "A": self.weight_A,
                "G": self.weight_G,
            },
            "C_ext": self.weighted_sum(),
            "Ctx": self.weighted_sum(),  # legacy alias for backward compatibility
        }


def load_metric(
    input_path: Path,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
) -> float:
    """Extract a latency metric from a benchmark JSON artefact."""

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    benchmark: MutableMapping[str, Any] | None = data.get(benchmark_key)
    if benchmark is None:
        raise KeyError(f"Benchmark key '{benchmark_key}' not found in {input_path}")

    stats = benchmark.get("stats")
    if stats is None:
        raise KeyError(f"Stats not present under '{benchmark_key}' in {input_path}")

    source = stats.get(time_source)
    if source is None:
        raise KeyError(
            f"Time source '{time_source}' missing under stats for '{benchmark_key}'"
        )

    if time_metric not in source:
        raise KeyError(
            f"Metric '{time_metric}' missing under '{time_source}' in {input_path}"
        )

    return float(source[time_metric])


def compute_conveyance(
    what: float,
    where: float,
    who: float,
    time_seconds: float,
    ctx_value: float,
    alpha: float,
) -> Dict[str, Any]:
    """Return efficiency/capability Conveyance values and zero-gate.

    Legacy single-sided formulation retained for backward compatibility.
    Interprets `(what·where·who)/time` as both `C_out` and `C_in` under a
    symmetric assumption with `P_ij = 1`, so the result approximates
    `C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij` when the two legs share the
    same bottlenecks.
    """

    zero_gate = (
        what <= 0
        or where <= 0
        or who <= 0
        or time_seconds <= 0
        or ctx_value <= 0
    )
    if zero_gate:
        return {
            "conveyance_efficiency": 0.0,
            "conveyance_capability": 0.0,
            "zero_propagation": True,
        }

    efficiency = (what * where * who / time_seconds) * (ctx_value**alpha)
    capability = (what * where * who) * (ctx_value**alpha)
    return {
        "conveyance_efficiency": efficiency,
        "conveyance_capability": capability,
        "zero_propagation": False,
    }


# -----------------------------
# Dyadic (pair-level) utilities
# -----------------------------

def hmean(x: float, y: float) -> float:
    """Harmonic mean for positive values; 0 if either side is non-positive."""

    if x <= 0 or y <= 0:
        return 0.0
    return 2.0 * x * y / (x + y)


def _validate_alpha(alpha: float) -> float:
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a numeric value between 1.5 and 2.0") from exc
    if not 1.5 <= alpha_value <= 2.0:
        raise ValueError("alpha must be between 1.5 and 2.0 (inclusive)")
    return alpha_value


def compute_pair_bottlenecks_efficiency(
    *,
    W_out: float,
    R_encode: float,
    H_out: float,
    T_out: float,
    W_in: float,
    R_decode: float,
    H_in: float,
    T_in: float,
) -> Tuple[float, float]:
    """Compute outbound/inbound bottlenecks for the efficiency view.

    Returns a tuple (C_out, C_in).
    """

    C_out = 0.0 if (W_out <= 0 or R_encode <= 0 or H_out <= 0 or T_out <= 0) else (
        (W_out * R_encode * H_out) / T_out
    )
    C_in = 0.0 if (W_in <= 0 or R_decode <= 0 or H_in <= 0 or T_in <= 0) else (
        (W_in * R_decode * H_in) / T_in
    )
    return C_out, C_in


def compute_pair_bottlenecks_capability(
    *,
    W_out: float,
    R_encode: float,
    H_out: float,
    W_in: float,
    R_decode: float,
    H_in: float,
) -> Tuple[float, float]:
    """Compute outbound/inbound bottlenecks for the capability view.

    Returns a tuple (C_out_cap, C_in_cap) without time terms.
    """

    C_out = 0.0 if (W_out <= 0 or R_encode <= 0 or H_out <= 0) else (
        W_out * R_encode * H_out
    )
    C_in = 0.0 if (W_in <= 0 or R_decode <= 0 or H_in <= 0) else (
        W_in * R_decode * H_in
    )
    return C_out, C_in


def compute_pair_conveyance(
    *,
    # Per-side factors (efficiency view requires T terms)
    W_out: float,
    R_encode: float,
    H_out: float,
    T_out: float,
    W_in: float,
    R_decode: float,
    H_in: float,
    T_in: float,
    # Shared external context and protocol gate
    C_ext: float,
    P_ij: float = 1.0,
    alpha: float = 1.7,
    view: str = "efficiency",
    monolithic: bool = False,
) -> Dict[str, Any]:
    """Compute pair-level conveyance per the updated framework.

    - view: one of {"efficiency", "capability"}
    - monolithic: if True, apply the rare monolithic alternative using the
      bottlenecks for the selected view.
    """

    alpha_value = _validate_alpha(alpha)

    # Protocol gate sanity
    if P_ij < 0 or P_ij > 1:
        raise ValueError("P_ij must be in [0, 1]")

    # Zero-propagation gates on C_ext and P_ij
    if C_ext <= 0 or P_ij == 0:
        return {
            "C_out": 0.0,
            "C_in": 0.0,
            "C_pair": 0.0,
            "zero_propagation": True,
            "gate": "C_ext_or_P_ij",
            "view": view,
            "monolithic": monolithic,
        }

    if view not in {"efficiency", "capability"}:
        raise ValueError("view must be 'efficiency' or 'capability'")

    if view == "efficiency":
        C_out, C_in = compute_pair_bottlenecks_efficiency(
            W_out=W_out,
            R_encode=R_encode,
            H_out=H_out,
            T_out=T_out,
            W_in=W_in,
            R_decode=R_decode,
            H_in=H_in,
            T_in=T_in,
        )
    else:
        C_out, C_in = compute_pair_bottlenecks_capability(
            W_out=W_out,
            R_encode=R_encode,
            H_out=H_out,
            W_in=W_in,
            R_decode=R_decode,
            H_in=H_in,
        )

    # Structural zero if either bottleneck collapses (harmonic-mean behavior)
    Hm = hmean(C_out, C_in)
    if Hm <= 0:
        return {
            "C_out": C_out,
            "C_in": C_in,
            "C_pair": 0.0,
            "zero_propagation": True,
            "gate": "bottleneck",
            "view": view,
            "monolithic": monolithic,
        }

    if monolithic:
        C_pair = (Hm * C_ext) ** alpha_value * P_ij
    else:
        C_pair = Hm * (C_ext ** alpha_value) * P_ij

    return {
        "C_out": C_out,
        "C_in": C_in,
        "C_pair": C_pair,
        "zero_propagation": False,
        "view": view,
        "monolithic": monolithic,
    }


def compute_acm(
    *,
    delta_W_rel: float,
    delta_R_connect: float,
    H: float,
    delta_T: float,
    C_ext_pre: float,
    alpha: float,
) -> float:
    """Compute Agentic Context Move (ACM) score for efficiency view.

    ACM = (ΔW_rel · ΔR_connect · H / ΔT) · C_ext_pre^(alpha - 1)
    """

    alpha_value = _validate_alpha(alpha)
    if delta_T <= 0 or H <= 0 or delta_W_rel <= 0 or delta_R_connect <= 0 or C_ext_pre <= 0:
        return 0.0
    return (
        (delta_W_rel * delta_R_connect * H) / delta_T
        * (C_ext_pre ** (alpha_value - 1.0))
    )


def build_record(
    *,
    input_path: Path,
    label: str,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
    time_units: str,
    what: float,
    where: float,
    who: float,
    context: ConveyanceContext,
    alpha: float,
    notes: str = "",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Construct a structured Conveyance record."""

    raw_value = load_metric(input_path, benchmark_key, time_source, time_metric)
    scale = TIME_UNITS.get(time_units)
    if scale is None:
        raise ValueError(f"Unsupported time unit '{time_units}'")

    time_seconds = raw_value / scale if scale != 0 else raw_value
    time_seconds = max(time_seconds, 1e-9)

    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a numeric value between 1.5 and 2.0") from exc

    if not 1.5 <= alpha_value <= 2.0:
        raise ValueError("alpha must be between 1.5 and 2.0 (inclusive)")

    ctx_value = context.weighted_sum()
    conveyance_payload = compute_conveyance(
        what=what,
        where=where,
        who=who,
        time_seconds=time_seconds,
        ctx_value=ctx_value,
        alpha=alpha_value,
    )

    record = {
        "label": label,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "input": str(input_path),
        "benchmark_key": benchmark_key,
        "time_source": time_source,
        "time_metric": time_metric,
        "time_units": time_units,
        "time_value": raw_value,
        "time_seconds": time_seconds,
        "what": what,
        "where": where,
        "who": who,
        "alpha": alpha_value,
        "context": context.as_dict(),
        "notes": notes,
    }
    record.update(conveyance_payload)
    return record


def append_record(output_path: Path, record: Mapping[str, Any]) -> None:
    """Append a record to a JSONL file."""

    with output_path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False)
        fh.write("\n")


def log_conveyance(
    *,
    input_path: Path,
    label: str,
    benchmark_key: str,
    time_source: str,
    time_metric: str,
    time_units: str,
    what: float,
    where: float,
    who: float,
    context: ConveyanceContext,
    alpha: float = 1.7,
    notes: str = "",
    output_path: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build (and optionally persist) a Conveyance benchmark record."""

    record = build_record(
        input_path=input_path,
        label=label,
        benchmark_key=benchmark_key,
        time_source=time_source,
        time_metric=time_metric,
        time_units=time_units,
        what=what,
        where=where,
        who=who,
        context=context,
        alpha=alpha,
        notes=notes,
    )

    if output_path is not None and not dry_run:
        append_record(output_path, record)

    return record


def build_pair_record(
    *,
    label: str,
    # Outbound
    W_out: float,
    R_encode: float,
    H_out: float,
    T_out: float,
    # Inbound
    W_in: float,
    R_decode: float,
    H_in: float,
    T_in: float,
    # Shared context and protocol
    context: ConveyanceContext,
    P_ij: float = 1.0,
    alpha: float = 1.7,
    view: str = "efficiency",
    monolithic: bool = False,
    notes: str = "",
    timestamp: datetime | None = None,
) -> Dict[str, Any]:
    """Construct a dyadic Conveyance record aligned to the updated framework."""

    ctx_value = context.weighted_sum()
    result = compute_pair_conveyance(
        W_out=W_out,
        R_encode=R_encode,
        H_out=H_out,
        T_out=T_out,
        W_in=W_in,
        R_decode=R_decode,
        H_in=H_in,
        T_in=T_in,
        C_ext=ctx_value,
        P_ij=P_ij,
        alpha=alpha,
        view=view,
        monolithic=monolithic,
    )

    record = {
        "label": label,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "alpha": float(alpha),
        "P_ij": float(P_ij),
        "view": view,
        "monolithic": monolithic,
        # Per-side inputs
        "outbound": {
            "W_out": W_out,
            "R_encode": R_encode,
            "H_out": H_out,
            "T_out": T_out,
        },
        "inbound": {
            "W_in": W_in,
            "R_decode": R_decode,
            "H_in": H_in,
            "T_in": T_in,
        },
        # Context surface
        "context": context.as_dict(),
        # Derived bottlenecks and outcome
        "C_out": result["C_out"],
        "C_in": result["C_in"],
        "C_pair": result["C_pair"],
        "zero_propagation": result["zero_propagation"],
    }

    if result.get("zero_propagation"):
        record["gate"] = result.get("gate", "")

    if notes:
        record["notes"] = notes

    return record


def log_pair_conveyance(
    *,
    label: str,
    W_out: float,
    R_encode: float,
    H_out: float,
    T_out: float,
    W_in: float,
    R_decode: float,
    H_in: float,
    T_in: float,
    context: ConveyanceContext,
    P_ij: float = 1.0,
    alpha: float = 1.7,
    view: str = "efficiency",
    monolithic: bool = False,
    notes: str = "",
    output_path: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build (and optionally persist) a dyadic Conveyance record."""

    record = build_pair_record(
        label=label,
        W_out=W_out,
        R_encode=R_encode,
        H_out=H_out,
        T_out=T_out,
        W_in=W_in,
        R_decode=R_decode,
        H_in=H_in,
        T_in=T_in,
        context=context,
        P_ij=P_ij,
        alpha=alpha,
        view=view,
        monolithic=monolithic,
        notes=notes,
    )

    if output_path is not None and not dry_run:
        append_record(output_path, record)

    return record


__all__ = [
    "ConveyanceContext",
    "TIME_UNITS",
    "build_record",
    "compute_conveyance",
    "hmean",
    "compute_pair_bottlenecks_efficiency",
    "compute_pair_bottlenecks_capability",
    "compute_pair_conveyance",
    "compute_acm",
    "load_metric",
    "log_conveyance",
    "build_pair_record",
    "log_pair_conveyance",
]
