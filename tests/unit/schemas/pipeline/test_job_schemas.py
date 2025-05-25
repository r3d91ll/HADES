"""
Unit tests for the job management schemas in the HADES-PathRAG system.

Tests job execution, result tracking, and batch processing schemas 
for the pipeline orchestration system.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

from pydantic import ValidationError

from src.schemas.pipeline.jobs import (
    JobStatus,
    JobSchema,
    JobResultSchema,
    BatchJobSchema
)


class TestJobStatus(unittest.TestCase):
    """Test the JobStatus enumeration."""
    
    def test_job_statuses(self):
        """Test all defined job statuses."""
        expected_statuses = [
            "pending", "scheduled", "running", "completed",
            "failed", "cancelled", "timeout", "retry"
        ]
        
        # Check that all expected statuses are defined
        for status in expected_statuses:
            self.assertTrue(hasattr(JobStatus, status.upper()))
            self.assertEqual(getattr(JobStatus, status.upper()).value, status)
        
        # Check total number of statuses
        self.assertEqual(len(JobStatus), len(expected_statuses))


class TestJobSchema(unittest.TestCase):
    """Test the JobSchema functionality."""
    
    def test_job_instantiation(self):
        """Test that JobSchema can be instantiated with required attributes."""
        # Test minimal job
        job = JobSchema(
            name="test-job",
            pipeline="text-processing"
        )
        
        self.assertIsNotNone(job.id)
        self.assertEqual(job.name, "test-job")
        self.assertEqual(job.pipeline, "text-processing")
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.params, {})
        self.assertEqual(job.priority, 0)
        self.assertIsNotNone(job.created_at)
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.completed_at)
        self.assertIsNone(job.scheduled_at)
        self.assertEqual(job.max_retries, 0)
        self.assertEqual(job.retry_count, 0)
        self.assertIsNone(job.timeout)
        self.assertIsNone(job.owner)
        self.assertEqual(job.tags, [])
        self.assertEqual(job.metadata, {})
        
        # Test job with all attributes
        created_time = datetime.now() - timedelta(hours=1)
        job = JobSchema(
            id="job-123",
            name="full-job",
            pipeline="embedding-pipeline",
            status=JobStatus.RUNNING,
            params={"model": "sentence-transformers", "batch_size": 32},
            priority=5,
            created_at=created_time,
            started_at=datetime.now() - timedelta(minutes=30),
            scheduled_at=created_time + timedelta(minutes=10),
            max_retries=3,
            retry_count=1,
            timeout=3600,
            owner="test-user",
            tags=["embedding", "production"],
            metadata={"source": "test-dataset"}
        )
        
        self.assertEqual(job.id, "job-123")
        self.assertEqual(job.name, "full-job")
        self.assertEqual(job.pipeline, "embedding-pipeline")
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertEqual(job.params, {"model": "sentence-transformers", "batch_size": 32})
        self.assertEqual(job.priority, 5)
        self.assertEqual(job.created_at, created_time)
        self.assertIsNotNone(job.started_at)
        self.assertIsNone(job.completed_at)
        self.assertIsNotNone(job.scheduled_at)
        self.assertEqual(job.max_retries, 3)
        self.assertEqual(job.retry_count, 1)
        self.assertEqual(job.timeout, 3600)
        self.assertEqual(job.owner, "test-user")
        self.assertEqual(job.tags, ["embedding", "production"])
        self.assertEqual(job.metadata, {"source": "test-dataset"})
    
    def test_positive_int_validation(self):
        """Test validation of positive integer values."""
        # Valid values
        job = JobSchema(
            name="test-job",
            pipeline="text-processing",
            priority=-5,  # Priority can be negative
            max_retries=0,  # Zero is allowed for max_retries
            retry_count=0,  # Zero is allowed for retry_count
            timeout=0  # Zero is allowed for timeout
        )
        
        self.assertEqual(job.priority, -5)
        self.assertEqual(job.max_retries, 0)
        self.assertEqual(job.retry_count, 0)
        self.assertEqual(job.timeout, 0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            JobSchema(
                name="test-job",
                pipeline="text-processing",
                max_retries=-1
            )
        
        with self.assertRaises(ValueError):
            JobSchema(
                name="test-job",
                pipeline="text-processing",
                retry_count=-1
            )
        
        with self.assertRaises(ValueError):
            JobSchema(
                name="test-job",
                pipeline="text-processing",
                timeout=-1
            )


class TestJobResultSchema(unittest.TestCase):
    """Test the JobResultSchema functionality."""
    
    def test_result_instantiation(self):
        """Test that JobResultSchema can be instantiated with required attributes."""
        # Test completed job result
        result = JobResultSchema(
            job_id="job-123",
            status=JobStatus.COMPLETED,
            output={"documents_processed": 100}
        )
        
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(result.output, {"documents_processed": 100})
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_details)
        self.assertIsNotNone(result.completed_at)
        self.assertIsNone(result.duration)
        self.assertEqual(result.metrics, {})
        
        # Test failed job result
        error_details = {
            "exception": "ValueError",
            "traceback": "Traceback information...",
            "stage": "preprocessing"
        }
        
        result = JobResultSchema(
            job_id="job-456",
            status=JobStatus.FAILED,
            error="Document processing failed",
            error_details=error_details,
            duration=120.5,
            metrics={"memory_usage": "2.5GB"}
        )
        
        self.assertEqual(result.job_id, "job-456")
        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertEqual(result.output, {})  # Now defaults to empty dict instead of None
        self.assertEqual(result.error, "Document processing failed")
        self.assertEqual(result.error_details, error_details)
        self.assertIsNotNone(result.completed_at)
        self.assertEqual(result.duration, 120.5)
        self.assertEqual(result.metrics, {"memory_usage": "2.5GB"})
    
    def test_validate_result(self):
        """Test validation of job result schema."""
        # Failed status requires error message
        with self.assertRaises(ValueError):
            JobResultSchema(
                job_id="job-123",
                status="failed",  # Use string instead of enum
                error=None
            )
        
        # Completed status with missing output gets empty dict
        result = JobResultSchema(
            job_id="job-123",
            status="completed",  # Use string instead of enum
            output=None
        )
        
        self.assertEqual(result.output, {})


class TestBatchJobSchema(unittest.TestCase):
    """Test the BatchJobSchema functionality."""
    
    def test_batch_job_instantiation(self):
        """Test that BatchJobSchema can be instantiated with required attributes."""
        # Create individual jobs for the batch
        job1 = JobSchema(
            name="job1",
            pipeline="text-processing"
        )
        
        job2 = JobSchema(
            name="job2",
            pipeline="text-processing"
        )
        
        # Test minimal batch job
        batch = BatchJobSchema(
            name="test-batch",
            jobs=[job1, job2]
        )
        
        self.assertIsNotNone(batch.id)
        self.assertEqual(batch.name, "test-batch")
        self.assertEqual(len(batch.jobs), 2)
        self.assertEqual(batch.jobs[0], job1)
        self.assertEqual(batch.jobs[1], job2)
        self.assertEqual(batch.strategy, "parallel")  # default value
        self.assertEqual(batch.max_concurrent, 4)  # default value
        self.assertEqual(batch.priority, 0)  # default value
        self.assertIsNone(batch.owner)  # default value
        self.assertEqual(batch.tags, [])  # default value
        self.assertEqual(batch.metadata, {})  # default value
        self.assertIsNotNone(batch.created_at)
        self.assertIsNone(batch.timeout)  # default value
        self.assertFalse(batch.retries_enabled)  # default value
        
        # Test batch job with all attributes
        batch = BatchJobSchema(
            id="batch-123",
            name="full-batch",
            jobs=[job1, job2],
            strategy="sequential",
            max_concurrent=2,
            priority=10,
            owner="test-user",
            tags=["daily", "production"],
            metadata={"source": "scheduled-task"},
            created_at=datetime.now(),
            timeout=7200,
            retries_enabled=True
        )
        
        self.assertEqual(batch.id, "batch-123")
        self.assertEqual(batch.name, "full-batch")
        self.assertEqual(len(batch.jobs), 2)
        self.assertEqual(batch.strategy, "sequential")
        self.assertEqual(batch.max_concurrent, 2)
        self.assertEqual(batch.priority, 10)
        self.assertEqual(batch.owner, "test-user")
        self.assertEqual(batch.tags, ["daily", "production"])
        self.assertEqual(batch.metadata, {"source": "scheduled-task"})
        self.assertIsNotNone(batch.created_at)
        self.assertEqual(batch.timeout, 7200)
        self.assertTrue(batch.retries_enabled)
    
    def test_strategy_validation(self):
        """Test validation of batch execution strategy."""
        job = JobSchema(name="job1", pipeline="test")
        
        # Valid strategies
        for strategy in ["parallel", "sequential", "dependency"]:
            batch = BatchJobSchema(
                name="test-batch",
                jobs=[job],
                strategy=strategy
            )
            self.assertEqual(batch.strategy, strategy)
        
        # Invalid strategy
        with self.assertRaises(ValueError):
            BatchJobSchema(
                name="test-batch",
                jobs=[job],
                strategy="invalid_strategy"
            )
    
    def test_max_concurrent_validation(self):
        """Test validation of max_concurrent value."""
        job = JobSchema(name="job1", pipeline="test")
        
        # Valid value
        batch = BatchJobSchema(
            name="test-batch",
            jobs=[job],
            max_concurrent=1
        )
        self.assertEqual(batch.max_concurrent, 1)
        
        # Invalid values
        with self.assertRaises(ValueError):
            BatchJobSchema(
                name="test-batch",
                jobs=[job],
                max_concurrent=0
            )
        
        with self.assertRaises(ValueError):
            BatchJobSchema(
                name="test-batch",
                jobs=[job],
                max_concurrent=-1
            )
    
    def test_jobs_validation(self):
        """Test validation that batch must contain at least one job."""
        # No jobs - should raise validation error
        with self.assertRaises(ValueError):
            BatchJobSchema(
                name="test-batch",
                jobs=[]
            )


if __name__ == "__main__":
    unittest.main()
