"""
Performance benchmarks for the alert system.

This module measures the performance of the alert system to ensure 
minimal impact on pipeline processing speed.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pytest

from src.alerts import AlertManager, AlertLevel

# Number of alerts to create in bulk tests
SMALL_BATCH = 100
MEDIUM_BATCH = 1000
LARGE_BATCH = 10000


def setup_test_environment(disable_handlers: bool = False) -> tuple:
    """Set up a test environment with a temporary directory."""
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    alert_dir = Path(temp_dir.name) / "alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    
    # Create alert manager with minimal handlers if requested
    handlers = None
    if disable_handlers:
        handlers = {"null_handler": lambda _: None}
        
    alert_manager = AlertManager(
        alert_dir=alert_dir,
        handlers=handlers
    )
    
    return alert_manager, temp_dir


def cleanup_environment(temp_dir: tempfile.TemporaryDirectory):
    """Clean up the test environment."""
    temp_dir.cleanup()


def benchmark_single_alert_creation(disable_handlers: bool = False) -> float:
    """
    Benchmark the creation of a single alert.
    
    Args:
        disable_handlers: Whether to disable alert handlers for pure creation timing
        
    Returns:
        Average time in milliseconds to create an alert
    """
    alert_manager, temp_dir = setup_test_environment(disable_handlers)
    
    # Warm-up
    for _ in range(10):
        alert_manager.alert(
            message="Warm-up alert",
            level=AlertLevel.LOW,
            source="benchmark"
        )
    
    # Timing runs
    times = []
    for i in range(100):
        start_time = time.time()
        
        alert_manager.alert(
            message=f"Benchmark alert {i}",
            level=AlertLevel.LOW,
            source="benchmark"
        )
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    cleanup_environment(temp_dir)
    
    # Return average time
    return sum(times) / len(times)


def benchmark_batch_alert_creation(batch_size: int, disable_handlers: bool = False) -> float:
    """
    Benchmark the creation of a batch of alerts.
    
    Args:
        batch_size: Number of alerts to create
        disable_handlers: Whether to disable alert handlers
        
    Returns:
        Alerts per second
    """
    alert_manager, temp_dir = setup_test_environment(disable_handlers)
    
    # Prepare messages and levels
    messages = [f"Benchmark batch alert {i}" for i in range(batch_size)]
    levels = [
        AlertLevel.LOW, 
        AlertLevel.MEDIUM, 
        AlertLevel.HIGH, 
        AlertLevel.CRITICAL
    ]
    
    # Timing
    start_time = time.time()
    
    for i in range(batch_size):
        level = levels[i % len(levels)]
        alert_manager.alert(
            message=messages[i],
            level=level,
            source="benchmark"
        )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    cleanup_environment(temp_dir)
    
    # Return alerts per second
    return batch_size / elapsed


def benchmark_alert_filtering(num_alerts: int) -> float:
    """
    Benchmark the filtering of alerts.
    
    Args:
        num_alerts: Number of alerts to create before filtering
        
    Returns:
        Time in milliseconds to filter alerts
    """
    alert_manager, temp_dir = setup_test_environment(disable_handlers=True)
    
    # Create alerts with different sources and levels
    sources = ["source_a", "source_b", "source_c", "source_d"]
    levels = list(AlertLevel)
    
    for i in range(num_alerts):
        source = sources[i % len(sources)]
        level = levels[i % len(levels)]
        
        alert_manager.alert(
            message=f"Alert {i}",
            level=level,
            source=source
        )
    
    # Benchmark different filtering operations
    filter_times = []
    
    # Filter by level
    start_time = time.time()
    alert_manager.get_alerts(min_level=AlertLevel.HIGH)
    filter_times.append((time.time() - start_time) * 1000)
    
    # Filter by source
    start_time = time.time()
    alert_manager.get_alerts(source="source_a")
    filter_times.append((time.time() - start_time) * 1000)
    
    # Combined filtering
    start_time = time.time()
    alert_manager.get_alerts(min_level=AlertLevel.MEDIUM, source="source_b")
    filter_times.append((time.time() - start_time) * 1000)
    
    cleanup_environment(temp_dir)
    
    # Return average filtering time
    return sum(filter_times) / len(filter_times)


def benchmark_file_handler(num_alerts: int) -> float:
    """
    Benchmark the file handler.
    
    Args:
        num_alerts: Number of alerts to process
        
    Returns:
        Alerts per second
    """
    alert_manager, temp_dir = setup_test_environment()
    
    # Only keep the file handler
    alert_manager.handlers = {"file": alert_manager._file_alert}
    
    # Timing
    start_time = time.time()
    
    for i in range(num_alerts):
        level = AlertLevel.LOW if i % 10 != 0 else AlertLevel.HIGH
        alert_manager.alert(
            message=f"File handler benchmark alert {i}",
            level=level,
            source="benchmark"
        )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    cleanup_environment(temp_dir)
    
    # Return alerts per second
    return num_alerts / elapsed


def run_all_benchmarks():
    """Run all benchmarks and print results."""
    print("\n===== Alert System Performance Benchmarks =====\n")
    
    # Single alert creation
    pure_creation_time = benchmark_single_alert_creation(disable_handlers=True)
    full_creation_time = benchmark_single_alert_creation(disable_handlers=False)
    
    print(f"Single Alert Creation (pure, no handlers): {pure_creation_time:.3f} ms")
    print(f"Single Alert Creation (with handlers): {full_creation_time:.3f} ms")
    print(f"Handler overhead: {(full_creation_time - pure_creation_time):.3f} ms")
    
    # Batch alert creation
    for batch_size in [SMALL_BATCH, MEDIUM_BATCH]:
        pure_rate = benchmark_batch_alert_creation(batch_size, disable_handlers=True)
        full_rate = benchmark_batch_alert_creation(batch_size, disable_handlers=False)
        
        print(f"\nBatch Creation ({batch_size} alerts):")
        print(f"  Pure rate (no handlers): {pure_rate:.2f} alerts/sec")
        print(f"  Full rate (with handlers): {full_rate:.2f} alerts/sec")
    
    # Alert filtering
    for num_alerts in [SMALL_BATCH, MEDIUM_BATCH]:
        filter_time = benchmark_alert_filtering(num_alerts)
        print(f"\nAlert Filtering ({num_alerts} alerts): {filter_time:.3f} ms")
    
    # File handler
    file_rate = benchmark_file_handler(SMALL_BATCH)
    print(f"\nFile Handler: {file_rate:.2f} alerts/sec")
    
    print("\n=== Benchmark Results Analysis ===")
    
    # Analyze results and provide summary
    if pure_creation_time < 0.5:
        print("✅ Alert creation is very fast (< 0.5 ms)")
    elif pure_creation_time < 1.0:
        print("✅ Alert creation is fast (< 1.0 ms)")
    else:
        print("⚠️ Alert creation might be slow (> 1.0 ms)")
    
    # Handlers overhead
    handler_ratio = full_creation_time / pure_creation_time
    if handler_ratio < 2:
        print("✅ Handler overhead is minimal (< 2x)")
    elif handler_ratio < 5:
        print("✅ Handler overhead is acceptable (< 5x)")
    else:
        print("⚠️ Handler overhead is significant (> 5x)")
    
    print("\n===== End of Benchmarks =====")


if __name__ == "__main__":
    run_all_benchmarks()
