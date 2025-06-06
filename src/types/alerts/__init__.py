"""
Alert type definitions for HADES-PathRAG.

This package contains type definitions for the alert system,
including alert levels, models, and data structures.
"""

from .base import AlertLevel
from .models import Alert

__all__ = [
    "AlertLevel",
    "Alert",
]