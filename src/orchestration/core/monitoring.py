"""Performance monitoring for parallel processing pipelines.

This module provides monitoring and metrics collection for pipeline stages,
worker pools, and queue performance to help diagnose bottlenecks and
optimize resource allocation.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Monitors performance of pipeline components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline monitor with configuration.
        
        Args:
            config: Configuration options for monitoring
        """
        self.config = config or {}
        self.start_time = time.time()
        self.components: Dict[str, Dict[str, Any]] = {}
        self.metrics_log_interval = self.config.get("metrics_log_interval_seconds", 30)
        self.enable_memory_warnings = self.config.get("enable_memory_warnings", True)
        self.memory_warning_threshold = self.config.get("memory_warning_threshold_percent", 85)
        
        # Lock for thread-safe updates
        self._lock = threading.Lock()
        
        # Worker thread for periodic monitoring
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def register_component(self, component_type: str, component_name: str, 
                          component: Any) -> None:
        """Register a component to be monitored.
        
        Args:
            component_type: Type of component (queue, worker, pipeline)
            component_name: Name of the component
            component: Component instance with get_metrics() method
        """
        with self._lock:
            component_id = f"{component_type}.{component_name}"
            self.components[component_id] = {
                "type": component_type,
                "name": component_name,
                "instance": component,
                "last_metrics": None,
                "registration_time": time.time()
            }
            logger.info(f"Registered {component_type} '{component_name}' for monitoring")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered components."""
        metrics = {}
        with self._lock:
            for component_id, component_info in self.components.items():
                instance = component_info["instance"]
                if hasattr(instance, "get_metrics") and callable(instance.get_metrics):
                    try:
                        metrics[component_id] = instance.get_metrics()
                        component_info["last_metrics"] = metrics[component_id]
                    except Exception as e:
                        logger.error(f"Error getting metrics for {component_id}: {e}")
                        metrics[component_id] = {"error": str(e)}
        
        # Add overall metrics
        metrics["_overall"] = {
            "uptime_seconds": time.time() - self.start_time,
            "component_count": len(self.components),
            "timestamp": time.time()
        }
        
        return metrics


# Export the class
__all__ = ["PipelineMonitor"]
