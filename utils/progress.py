"""
Progress Tracking for Long-Running Operations

Provides consistent progress tracking across all infrastructure components.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress of long-running operations with ETA calculation.
    
    Features:
    - Automatic ETA calculation
    - Rate limiting for updates
    - Multiple progress bars
    - Callback support
    
    Example:
        tracker = ProgressTracker(total=1000, desc="Processing documents")
        
        for i in range(1000):
            # Do work...
            tracker.update(1)
        
        tracker.close()
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "items",
        disable: bool = False,
        update_interval: float = 0.1,
        callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description of the operation
            unit: Unit name for items
            disable: Whether to disable progress display
            update_interval: Minimum seconds between updates
            callback: Optional callback for progress updates
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.update_interval = update_interval
        self.callback = callback
        
        # Initialize tqdm progress bar
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            disable=disable
        )
        
        # Tracking variables
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.rates = []  # Moving average of rates
        
    def update(self, n: int = 1, **kwargs):
        """
        Update progress by n items.
        
        Args:
            n: Number of items completed
            **kwargs: Additional info to pass to callback
        """
        self.current += n
        current_time = time.time()
        
        # Update progress bar
        self.pbar.update(n)
        
        # Rate limiting for callback
        if current_time - self.last_update_time >= self.update_interval:
            self._calculate_stats()
            
            if self.callback:
                stats = self.get_stats()
                stats.update(kwargs)
                self.callback(stats)
            
            self.last_update_time = current_time
    
    def _calculate_stats(self):
        """Calculate current statistics."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            rate = self.current / elapsed
            self.rates.append(rate)
            
            # Keep only recent rates for moving average
            if len(self.rates) > 100:
                self.rates.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current progress statistics.
        
        Returns:
            Dictionary with progress stats
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate rate
        if elapsed > 0:
            rate = self.current / elapsed
            avg_rate = sum(self.rates) / len(self.rates) if self.rates else rate
        else:
            rate = avg_rate = 0
        
        # Calculate ETA
        if avg_rate > 0 and self.current < self.total:
            remaining = self.total - self.current
            eta_seconds = remaining / avg_rate
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = None
        
        return {
            "current": self.current,
            "total": self.total,
            "percentage": (self.current / self.total * 100) if self.total > 0 else 0,
            "elapsed": timedelta(seconds=int(elapsed)),
            "rate": avg_rate,
            "rate_unit": f"{self.unit}/s",
            "eta": eta,
            "description": self.desc
        }
    
    def close(self):
        """Close the progress tracker."""
        self.pbar.close()
        
        # Final callback
        if self.callback:
            stats = self.get_stats()
            stats["finished"] = True
            self.callback(stats)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MultiProgressTracker:
    """
    Manages multiple progress bars for complex operations.
    
    Example:
        tracker = MultiProgressTracker()
        
        # Add progress bars
        tracker.add_bar("extraction", total=1000, desc="Extracting")
        tracker.add_bar("embedding", total=1000, desc="Embedding")
        
        # Update specific bars
        tracker.update("extraction", 10)
        tracker.update("embedding", 5)
        
        tracker.close_all()
    """
    
    def __init__(self):
        """Initialize multi-progress tracker."""
        self.bars = {}
        self.positions = {}
        self.next_position = 0
    
    def add_bar(
        self,
        name: str,
        total: int,
        desc: Optional[str] = None,
        **kwargs
    ) -> ProgressTracker:
        """
        Add a new progress bar.
        
        Args:
            name: Unique name for the bar
            total: Total items for this bar
            desc: Description (defaults to name)
            **kwargs: Additional arguments for ProgressTracker
            
        Returns:
            The created ProgressTracker
        """
        if name in self.bars:
            raise ValueError(f"Progress bar '{name}' already exists")
        
        desc = desc or name
        
        # Create tracker with position for multi-bar display
        tracker = ProgressTracker(
            total=total,
            desc=desc,
            **kwargs
        )
        
        self.bars[name] = tracker
        self.positions[name] = self.next_position
        self.next_position += 1
        
        return tracker
    
    def update(self, name: str, n: int = 1, **kwargs):
        """Update a specific progress bar."""
        if name not in self.bars:
            raise ValueError(f"Progress bar '{name}' not found")
        
        self.bars[name].update(n, **kwargs)
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for one or all progress bars.
        
        Args:
            name: Specific bar name, or None for all
            
        Returns:
            Statistics dictionary
        """
        if name:
            if name not in self.bars:
                raise ValueError(f"Progress bar '{name}' not found")
            return self.bars[name].get_stats()
        else:
            return {
                name: bar.get_stats()
                for name, bar in self.bars.items()
            }
    
    def close_all(self):
        """Close all progress bars."""
        for bar in self.bars.values():
            bar.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()