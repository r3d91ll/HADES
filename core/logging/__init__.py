"""Logging utilities for HADES.

The structured logging helper (`core.logging.logging`) depends on optional
third-party packages such as ``structlog``. Import it lazily so callers who only
need Conveyance helpers do not require those extras. Production deployments can
opt-in by installing the structured logging dependencies.
"""

from __future__ import annotations

from .conveyance import (  # noqa: F401
    TIME_UNITS,
    ConveyanceContext,
    build_record,
    compute_conveyance,
    load_metric,
    log_conveyance,
)

try:  # pragma: no cover - optional dependency
    from .logging import LogManager  # type: ignore[attr-defined] # noqa: F401
except Exception:  # pragma: no cover - structlog not installed
    LogManager = None  # type: ignore[assignment]

__all__ = [
    "LogManager",
    "ConveyanceContext",
    "TIME_UNITS",
    "build_record",
    "compute_conveyance",
    "load_metric",
    "log_conveyance",
]
