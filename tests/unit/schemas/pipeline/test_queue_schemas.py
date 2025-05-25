"""
Unit tests for the queue management schemas in the HADES-PathRAG system.

Tests task queue configuration, task schema, and queue statistics functionality.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

from pydantic import ValidationError

from src.schemas.pipeline.queue import (
    TaskPriority,
    TaskStatus,
    TaskSchema,
    QueueConfigSchema,
    QueueStatsSchema
)


class TestTaskPriority(unittest.TestCase):
    """Test the TaskPriority enumeration."""
    
    def test_task_priorities(self):
        """Test all defined task priorities."""
        expected_priorities = [
            ("LOW", 0),
            ("NORMAL", 1),
            ("HIGH", 2),
            ("CRITICAL", 3)
        ]
        
        # Check that all expected priorities are defined
        for name, value in expected_priorities:
            self.assertTrue(hasattr(TaskPriority, name))
            self.assertEqual(getattr(TaskPriority, name).value, value)
        
        # Check total number of priorities
        self.assertEqual(len(TaskPriority), len(expected_priorities))
        
        # Test ordering
        self.assertTrue(TaskPriority.LOW < TaskPriority.NORMAL)
        self.assertTrue(TaskPriority.NORMAL < TaskPriority.HIGH)
        self.assertTrue(TaskPriority.HIGH < TaskPriority.CRITICAL)


class TestTaskStatus(unittest.TestCase):
    """Test the TaskStatus enumeration."""
    
    def test_task_statuses(self):
        """Test all defined task statuses."""
        expected_statuses = [
            "pending", "processing", "completed", 
            "failed", "retrying", "canceled"
        ]
        
        # Check that all expected statuses are defined
        for status in expected_statuses:
            self.assertTrue(hasattr(TaskStatus, status.upper()))
            self.assertEqual(getattr(TaskStatus, status.upper()).value, status)
        
        # Check total number of statuses
        self.assertEqual(len(TaskStatus), len(expected_statuses))


class TestTaskSchema(unittest.TestCase):
    """Test the TaskSchema functionality."""
    
    def test_task_instantiation(self):
        """Test that TaskSchema can be instantiated with required attributes."""
        # Test minimal task
        task = TaskSchema(
            task_id="task123",
            task_type="document_processing",
            payload={"document_id": "doc123"}
        )
        
        self.assertEqual(task.task_id, "task123")
        self.assertEqual(task.task_type, "document_processing")
        self.assertEqual(task.priority, TaskPriority.NORMAL)  # default value
        self.assertEqual(task.status, TaskStatus.PENDING)  # default value
        self.assertEqual(task.payload, {"document_id": "doc123"})
        self.assertIsNotNone(task.created_at)
        self.assertIsNone(task.started_at)
        self.assertIsNone(task.completed_at)
        self.assertIsNone(task.worker_id)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.max_retries, 3)
        self.assertIsNone(task.error)
        self.assertIsNone(task.parent_task_id)
        self.assertEqual(task.depends_on, [])
        self.assertEqual(task.metadata, {})
        
        # Test with all attributes
        created_time = datetime.now() - timedelta(minutes=10)
        started_time = datetime.now() - timedelta(minutes=5)
        completed_time = datetime.now()
        
        task = TaskSchema(
            task_id="task456",
            task_type="embedding_generation",
            priority=TaskPriority.HIGH,
            status=TaskStatus.COMPLETED,
            payload={"document_id": "doc456", "chunk_ids": ["chunk1", "chunk2"]},
            created_at=created_time,
            started_at=started_time,
            completed_at=completed_time,
            worker_id="worker1",
            retry_count=1,
            max_retries=5,
            parent_task_id="parent_task",
            depends_on=["dependency1", "dependency2"],
            metadata={"processor": "text_embedder", "model": "modernbert"}
        )
        
        self.assertEqual(task.task_id, "task456")
        self.assertEqual(task.task_type, "embedding_generation")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.payload, {"document_id": "doc456", "chunk_ids": ["chunk1", "chunk2"]})
        self.assertEqual(task.created_at, created_time)
        self.assertEqual(task.started_at, started_time)
        self.assertEqual(task.completed_at, completed_time)
        self.assertEqual(task.worker_id, "worker1")
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.max_retries, 5)
        self.assertIsNone(task.error)
        self.assertEqual(task.parent_task_id, "parent_task")
        self.assertEqual(task.depends_on, ["dependency1", "dependency2"])
        self.assertEqual(task.metadata, {"processor": "text_embedder", "model": "modernbert"})
    
    def test_status_consistency_completed(self):
        """Test status consistency validation with completed_at timestamp."""
        # Creating a task with completed_at but status not set to completed
        task = TaskSchema(
            task_id="task123",
            task_type="test",
            payload={},
            status=TaskStatus.PROCESSING,
            completed_at=datetime.now()
        )
        
        # Status should be automatically updated to COMPLETED
        self.assertEqual(task.status, TaskStatus.COMPLETED)
    
    def test_status_consistency_started(self):
        """Test status consistency validation with started_at timestamp."""
        # Creating a task with started_at but status set to PENDING
        task = TaskSchema(
            task_id="task123",
            task_type="test",
            payload={},
            status=TaskStatus.PENDING,
            started_at=datetime.now()
        )
        
        # Status should be automatically updated to PROCESSING
        self.assertEqual(task.status, TaskStatus.PROCESSING)
    
    def test_status_consistency_error(self):
        """Test status consistency validation with error field."""
        # Creating a task with error but status not set to FAILED
        task = TaskSchema(
            task_id="task123",
            task_type="test",
            payload={},
            status=TaskStatus.PROCESSING,
            error="Something went wrong"
        )
        
        # Status should be automatically updated to FAILED
        self.assertEqual(task.status, TaskStatus.FAILED)


class TestQueueConfigSchema(unittest.TestCase):
    """Test the QueueConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that QueueConfigSchema can be instantiated with required attributes."""
        # Test minimal config
        config = QueueConfigSchema(
            queue_name="test-queue"
        )
        
        self.assertEqual(config.queue_name, "test-queue")
        self.assertEqual(config.max_size, 1000)  # default value
        self.assertEqual(config.dequeue_batch_size, 10)  # default value
        self.assertTrue(config.enable_priority)  # default value
        self.assertTrue(config.enable_persistence)  # default value
        self.assertIsNone(config.persistence_path)  # default value
        self.assertTrue(config.enable_dead_letter)  # default value
        self.assertEqual(config.max_retry_delay_seconds, 300)  # default value
        self.assertEqual(config.metadata, {})  # default value
        
        # Test with all attributes
        config = QueueConfigSchema(
            queue_name="full-queue",
            max_size=500,
            dequeue_batch_size=5,
            enable_priority=False,
            enable_persistence=True,
            persistence_path="/path/to/queue/storage",
            enable_dead_letter=False,
            max_retry_delay_seconds=600,
            metadata={"owner": "test-user"}
        )
        
        self.assertEqual(config.queue_name, "full-queue")
        self.assertEqual(config.max_size, 500)
        self.assertEqual(config.dequeue_batch_size, 5)
        self.assertFalse(config.enable_priority)
        self.assertTrue(config.enable_persistence)
        self.assertEqual(config.persistence_path, "/path/to/queue/storage")
        self.assertFalse(config.enable_dead_letter)
        self.assertEqual(config.max_retry_delay_seconds, 600)
        self.assertEqual(config.metadata, {"owner": "test-user"})
    
    def test_positive_validation(self):
        """Test validation of positive integer values."""
        # Test max_size validation
        with self.assertRaises(ValidationError):
            QueueConfigSchema(
                queue_name="test",
                max_size=0
            )
        
        with self.assertRaises(ValidationError):
            QueueConfigSchema(
                queue_name="test",
                max_size=-100
            )
        
        # Test dequeue_batch_size validation
        with self.assertRaises(ValidationError):
            QueueConfigSchema(
                queue_name="test",
                dequeue_batch_size=0
            )
        
        with self.assertRaises(ValidationError):
            QueueConfigSchema(
                queue_name="test",
                dequeue_batch_size=-5
            )


class TestQueueStatsSchema(unittest.TestCase):
    """Test the QueueStatsSchema functionality."""
    
    def test_stats_instantiation(self):
        """Test that QueueStatsSchema can be instantiated with required attributes."""
        # Test minimal stats
        stats = QueueStatsSchema(
            queue_name="test-queue"
        )
        
        self.assertEqual(stats.queue_name, "test-queue")
        self.assertEqual(stats.current_size, 0)
        self.assertEqual(stats.peak_size, 0)
        self.assertEqual(stats.total_enqueued, 0)
        self.assertEqual(stats.total_dequeued, 0)
        self.assertEqual(stats.total_completed, 0)
        self.assertEqual(stats.total_failed, 0)
        self.assertEqual(stats.total_retries, 0)
        self.assertEqual(stats.average_wait_time_ms, 0.0)
        self.assertEqual(stats.average_processing_time_ms, 0.0)
        self.assertEqual(stats.tasks_by_priority, {})
        self.assertEqual(stats.tasks_by_status, {})
    
    def test_update_stats_enqueue(self):
        """Test updating stats for enqueue event."""
        stats = QueueStatsSchema(queue_name="test-queue")
        
        # Create a task with string values instead of enum objects
        # This matches how the actual implementation expects them
        task = TaskSchema(
            task_id="task1",
            task_type="test",
            payload={},
            priority=2,  # HIGH value
            status="pending"
        )
        
        # Update stats with enqueue event
        stats.update_stats(task, 'enqueue')
        
        # Check updated values
        self.assertEqual(stats.total_enqueued, 1)
        self.assertEqual(stats.current_size, 1)
        self.assertEqual(stats.peak_size, 1)
        # Priority is stored as int value in tests
        self.assertEqual(stats.tasks_by_priority.get("2", 0), 1)
        self.assertEqual(stats.tasks_by_status, {"pending": 1})
        
        # Enqueue another task
        task2 = TaskSchema(
            task_id="task2",
            task_type="test",
            payload={},
            priority=1,  # NORMAL value
            status="pending"
        )
        
        stats.update_stats(task2, 'enqueue')
        
        # Check updated values
        self.assertEqual(stats.total_enqueued, 2)
        self.assertEqual(stats.current_size, 2)
        self.assertEqual(stats.peak_size, 2)
        # Check both priorities
        self.assertEqual(stats.tasks_by_priority.get("2", 0), 1)
        self.assertEqual(stats.tasks_by_priority.get("1", 0), 1)
        self.assertEqual(stats.tasks_by_status, {"pending": 2})
    
    def test_update_stats_dequeue(self):
        """Test updating stats for dequeue event."""
        stats = QueueStatsSchema(queue_name="test-queue")
        
        # Setup: enqueue a task first
        task = TaskSchema(
            task_id="task1",
            task_type="test",
            payload={},
            status="pending",
            priority=1,  # NORMAL value
            created_at=datetime.now() - timedelta(seconds=10)
        )
        
        stats.update_stats(task, 'enqueue')
        
        # Now dequeue the task
        task.started_at = datetime.now()
        task.status = "processing"  # Use string directly
        
        stats.update_stats(task, 'dequeue')
        
        # Check updated values
        self.assertEqual(stats.total_dequeued, 1)
        self.assertEqual(stats.current_size, 0)
        self.assertGreater(stats.average_wait_time_ms, 0)  # Should have some wait time
        self.assertEqual(stats.tasks_by_status, {"pending": 0, "processing": 1})
    
    def test_update_stats_complete(self):
        """Test updating stats for complete event."""
        stats = QueueStatsSchema(queue_name="test-queue")
        
        # Setup: enqueue and dequeue a task first
        task = TaskSchema(
            task_id="task1",
            task_type="test",
            payload={},
            status="pending", 
            priority=1,  # NORMAL value
            created_at=datetime.now() - timedelta(seconds=20)
        )
        
        stats.update_stats(task, 'enqueue')
        
        # Update for dequeue
        task.started_at = datetime.now() - timedelta(seconds=10)
        task.status = "processing"
        stats.update_stats(task, 'dequeue')
        
        # Now complete the task
        task.completed_at = datetime.now()
        task.status = "completed"
        
        stats.update_stats(task, 'complete')
        
        # Check updated values
        self.assertEqual(stats.total_completed, 1)
        self.assertGreater(stats.average_processing_time_ms, 0)  # Should have some processing time
        self.assertEqual(stats.tasks_by_status, {"pending": 0, "processing": 0, "completed": 1})
    
    def test_update_stats_fail(self):
        """Test updating stats for fail event."""
        stats = QueueStatsSchema(queue_name="test-queue")
        
        # Setup: enqueue and dequeue a task first
        task = TaskSchema(
            task_id="task1",
            task_type="test",
            payload={},
            status="pending",
            priority=1  # NORMAL value
        )
        
        stats.update_stats(task, 'enqueue')
        
        # Update for dequeue
        task.status = "processing"
        task.started_at = datetime.now()
        stats.update_stats(task, 'dequeue')
        
        # Now fail the task
        task.error = "Test error"
        task.status = "failed"
        
        stats.update_stats(task, 'fail')
        
        # Check updated values
        self.assertEqual(stats.total_failed, 1)
        self.assertEqual(stats.tasks_by_status, {"pending": 0, "processing": 0, "failed": 1})
    
    def test_update_stats_retry(self):
        """Test updating stats for retry event."""
        stats = QueueStatsSchema(queue_name="test-queue")
        
        # Setup: enqueue, dequeue, and fail a task first
        task = TaskSchema(
            task_id="task1",
            task_type="test",
            payload={},
            status="pending",
            priority=1,  # NORMAL value
            error=None
        )
        
        stats.update_stats(task, 'enqueue')
        
        # Update for dequeue
        task.status = "processing"
        task.started_at = datetime.now()
        stats.update_stats(task, 'dequeue')
        
        # Update for fail
        task.status = "failed"
        task.error = "Test error"
        stats.update_stats(task, 'fail')
        
        # Now retry the task
        task.status = "retrying"
        task.retry_count = 1
        
        stats.update_stats(task, 'retry')
        
        # Check updated values
        self.assertEqual(stats.total_retries, 1)
        self.assertEqual(stats.tasks_by_status, {"pending": 0, "processing": 0, "failed": 0, "retrying": 1})


if __name__ == "__main__":
    unittest.main()
