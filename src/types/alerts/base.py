"""
Base types and enums for the alert system.

This module contains foundational types used throughout the alert system.
"""

from enum import Enum, auto


class AlertLevel(Enum):
    """Alert severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()