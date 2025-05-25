"""
Unit tests for the worker schemas in the HADES-PathRAG system.

Tests worker status, types, configuration, statistics, and pool management
functionality for pipeline orchestration.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

from pydantic import ValidationError

from src.schemas.pipeline.workers import (
    WorkerStatus,
    WorkerType,
    WorkerConfigSchema,
    WorkerStatsSchema,
    WorkerPoolConfigSchema
)


class TestWorkerStatus(unittest.TestCase):
    """Test the WorkerStatus enumeration."""
    
    def test_worker_statuses(self):
        """Test all defined worker statuses."""
        expected_statuses = [
            "idle", "busy", "error", "terminated"
        ]
        
        # Check that all expected statuses are defined
        for status in expected_statuses:
            self.assertTrue(hasattr(WorkerStatus, status.upper()))
            self.assertEqual(getattr(WorkerStatus, status.upper()).value, status)
        
        # Check total number of statuses
        self.assertEqual(len(WorkerStatus), len(expected_statuses))


class TestWorkerType(unittest.TestCase):
    """Test the WorkerType enumeration."""
    
    def test_worker_types(self):
        """Test all defined worker types."""
        expected_types = [
            "processor", "loader", "chunker", "embedder", 
            "indexer", "storage", "generic"
        ]
        
        # Check that all expected types are defined
        for worker_type in expected_types:
            self.assertTrue(hasattr(WorkerType, worker_type.upper()))
            self.assertEqual(getattr(WorkerType, worker_type.upper()).value, worker_type)
        
        # Check total number of types
        self.assertEqual(len(WorkerType), len(expected_types))


class TestWorkerConfigSchema(unittest.TestCase):
    """Test the WorkerConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that WorkerConfigSchema can be instantiated with required attributes."""
        # Test minimal config
        config = WorkerConfigSchema(
            worker_type=WorkerType.PROCESSOR
        )
        
        self.assertEqual(config.worker_type, WorkerType.PROCESSOR)
        self.assertIsNone(config.worker_id)  # default value
        self.assertEqual(config.max_batch_size, 10)  # default value
        self.assertEqual(config.max_retries, 3)  # default value
        self.assertEqual(config.timeout_seconds, 300)  # default value
        self.assertEqual(config.priority, 0)  # default value
        self.assertEqual(config.resource_limits, {})  # default value
        self.assertEqual(config.metadata, {})  # default value
        
        # Test with all attributes
        config = WorkerConfigSchema(
            worker_type=WorkerType.EMBEDDER,
            worker_id="embedder-001",
            max_batch_size=32,
            max_retries=5,
            timeout_seconds=600,
            priority=10,
            resource_limits={"memory": "4G", "cpu": 2},
            metadata={"model": "sentence-transformers", "device": "cpu"}
        )
        
        self.assertEqual(config.worker_type, WorkerType.EMBEDDER)
        self.assertEqual(config.worker_id, "embedder-001")
        self.assertEqual(config.max_batch_size, 32)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.timeout_seconds, 600)
        self.assertEqual(config.priority, 10)
        self.assertEqual(config.resource_limits, {"memory": "4G", "cpu": 2})
        self.assertEqual(config.metadata, {"model": "sentence-transformers", "device": "cpu"})
    
    def test_positive_validation(self):
        """Test validation of positive integer values."""
        # Test max_batch_size validation
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                max_batch_size=0
            )
        
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                max_batch_size=-10
            )
        
        # Test max_retries validation
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                max_retries=0
            )
        
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                max_retries=-5
            )
        
        # Test timeout_seconds validation
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                timeout_seconds=0
            )
        
        with self.assertRaises(ValueError):
            WorkerConfigSchema(
                worker_type=WorkerType.PROCESSOR,
                timeout_seconds=-30
            )


class TestWorkerStatsSchema(unittest.TestCase):
    """Test the WorkerStatsSchema functionality."""
    
    def test_stats_instantiation(self):
        """Test that WorkerStatsSchema can be instantiated with required attributes."""
        # Test minimal stats
        stats = WorkerStatsSchema(
            worker_id="worker-001",
            worker_type=WorkerType.PROCESSOR
        )
        
        self.assertEqual(stats.worker_id, "worker-001")
        self.assertEqual(stats.worker_type, WorkerType.PROCESSOR)
        self.assertEqual(stats.status, WorkerStatus.IDLE)  # default value
        self.assertIsNotNone(stats.start_time)
        self.assertIsNotNone(stats.last_active)
        self.assertEqual(stats.tasks_processed, 0)
        self.assertEqual(stats.tasks_failed, 0)
        self.assertEqual(stats.avg_processing_time_ms, 0.0)
        self.assertIsNone(stats.current_task)
        self.assertEqual(stats.errors, [])
    
    def test_update_status(self):
        """Test updating worker status."""
        stats = WorkerStatsSchema(
            worker_id="worker-001",
            worker_type=WorkerType.PROCESSOR
        )
        
        # Record initial last_active time
        initial_last_active = stats.last_active
        
        # Wait a short time
        datetime.now()  # just to add a small delay
        
        # Update status
        stats.update_status(WorkerStatus.BUSY)
        
        # Verify status updated
        self.assertEqual(stats.status, WorkerStatus.BUSY)
        self.assertGreater(stats.last_active, initial_last_active)
    
    def test_record_task_processed(self):
        """Test recording a processed task."""
        stats = WorkerStatsSchema(
            worker_id="worker-001",
            worker_type=WorkerType.PROCESSOR
        )
        
        # Initial state
        self.assertEqual(stats.tasks_processed, 0)
        self.assertEqual(stats.avg_processing_time_ms, 0.0)
        
        # Record first task
        stats.record_task_processed(100.0)
        
        # Verify stats updated
        self.assertEqual(stats.tasks_processed, 1)
        self.assertEqual(stats.avg_processing_time_ms, 100.0)
        
        # Record second task
        stats.record_task_processed(200.0)
        
        # Verify average calculation
        self.assertEqual(stats.tasks_processed, 2)
        self.assertEqual(stats.avg_processing_time_ms, 150.0)  # (100 + 200) / 2
        
        # Record third task
        stats.record_task_processed(300.0)
        
        # Verify average calculation
        self.assertEqual(stats.tasks_processed, 3)
        self.assertEqual(stats.avg_processing_time_ms, 200.0)  # (100 + 200 + 300) / 3
    
    def test_record_error(self):
        """Test recording an error."""
        stats = WorkerStatsSchema(
            worker_id="worker-001",
            worker_type=WorkerType.PROCESSOR
        )
        
        # Initial state
        self.assertEqual(stats.tasks_failed, 0)
        self.assertEqual(len(stats.errors), 0)
        self.assertEqual(stats.status, WorkerStatus.IDLE)
        
        # Record error without task ID
        stats.record_error("An error occurred")
        
        # Verify error recorded
        self.assertEqual(stats.tasks_failed, 1)
        self.assertEqual(len(stats.errors), 1)
        self.assertEqual(stats.status, WorkerStatus.ERROR)
        self.assertEqual(stats.errors[0]["error"], "An error occurred")
        self.assertIn("timestamp", stats.errors[0])
        self.assertNotIn("task_id", stats.errors[0])
        
        # Record error with task ID
        stats.record_error("Another error", task_id="task-123")
        
        # Verify error recorded
        self.assertEqual(stats.tasks_failed, 2)
        self.assertEqual(len(stats.errors), 2)
        self.assertEqual(stats.errors[1]["error"], "Another error")
        self.assertEqual(stats.errors[1]["task_id"], "task-123")


class TestWorkerPoolConfigSchema(unittest.TestCase):
    """Test the WorkerPoolConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that WorkerPoolConfigSchema can be instantiated with default attributes."""
        # Test default config
        config = WorkerPoolConfigSchema()
        
        self.assertEqual(config.min_workers, 1)
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.worker_configs, {})
        self.assertTrue(config.scaling_enabled)
        self.assertEqual(config.scaling_cooldown_seconds, 60)
        self.assertEqual(config.task_queue_size, 100)
        
        # Test with custom attributes
        processor_config = WorkerConfigSchema(
            worker_type=WorkerType.PROCESSOR,
            max_batch_size=20
        )
        
        embedder_config = WorkerConfigSchema(
            worker_type=WorkerType.EMBEDDER,
            max_batch_size=32,
            timeout_seconds=600
        )
        
        config = WorkerPoolConfigSchema(
            min_workers=2,
            max_workers=8,
            worker_configs={
                WorkerType.PROCESSOR: processor_config,
                WorkerType.EMBEDDER: embedder_config
            },
            scaling_enabled=True,
            scaling_cooldown_seconds=120,
            task_queue_size=200
        )
        
        self.assertEqual(config.min_workers, 2)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(len(config.worker_configs), 2)
        self.assertEqual(config.worker_configs[WorkerType.PROCESSOR], processor_config)
        self.assertEqual(config.worker_configs[WorkerType.EMBEDDER], embedder_config)
        self.assertTrue(config.scaling_enabled)
        self.assertEqual(config.scaling_cooldown_seconds, 120)
        self.assertEqual(config.task_queue_size, 200)
    
    def test_worker_count_validation(self):
        """Test validation of worker count."""
        # Test min_workers validation
        with self.assertRaises(ValueError):
            WorkerPoolConfigSchema(
                min_workers=0
            )
        
        with self.assertRaises(ValueError):
            WorkerPoolConfigSchema(
                min_workers=-1
            )
        
        # Test max_workers validation
        with self.assertRaises(ValueError):
            WorkerPoolConfigSchema(
                max_workers=0
            )
        
        with self.assertRaises(ValueError):
            WorkerPoolConfigSchema(
                max_workers=-5
            )
    
    def test_cooldown_validation(self):
        """Test validation of cooldown period."""
        # Valid cooldown
        config = WorkerPoolConfigSchema(
            scaling_cooldown_seconds=0
        )
        self.assertEqual(config.scaling_cooldown_seconds, 0)
        
        # Invalid cooldown
        with self.assertRaises(ValueError):
            WorkerPoolConfigSchema(
                scaling_cooldown_seconds=-1
            )


if __name__ == "__main__":
    unittest.main()
