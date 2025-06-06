"""
Alert data models and structures.

This module contains the core Alert class and related data structures.
"""

import time
from typing import Dict, Any, Optional

from .base import AlertLevel


class Alert:
    """
    Alert class representing a single alert.
    
    This class encapsulates all information about an alert, including
    its message, level, source, and timestamp.
    """
    
    def __init__(
            self,
            message: str,
            level: AlertLevel,
            source: str,
            context: Optional[Dict[str, Any]] = None,
            timestamp: Optional[float] = None
        ):
        """
        Initialize an alert.
        
        Args:
            message: The alert message
            level: Alert severity level
            source: Component or module that generated the alert
            context: Additional context data related to the alert
            timestamp: Alert timestamp (defaults to current time)
        """
        self.message = message
        self.level = level
        self.source = source
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        self.id = f"{int(self.timestamp)}_{hash(message) % 10000:04d}"
    
    def __str__(self) -> str:
        """Return string representation of the alert."""
        return (
            f"Alert[{self.level.name}] {self.source}: {self.message} "
            f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level.name,
            "source": self.source,
            "context": self.context,
            "timestamp": self.timestamp
        }