"""
Unit tests for the base pipeline schemas in the HADES-PathRAG system.

Tests the core pipeline functionality including pipeline stages, status tracking,
configuration, and execution statistics.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any

from pydantic import ValidationError

from src.schemas.pipeline.base import (
    PipelineStage,
    PipelineStatus,
    PipelineConfigSchema,
    PipelineStatsSchema
)


class TestPipelineStage(unittest.TestCase):
    """Test the PipelineStage enumeration."""
    
    def test_pipeline_stages(self):
        """Test all defined pipeline stages."""
        expected_stages = [
            "load", "preprocess", "chunk", "embed", 
            "index", "store", "complete", "failed"
        ]
        
        # Check that all expected stages are defined
        for stage in expected_stages:
            self.assertTrue(hasattr(PipelineStage, stage.upper()))
            self.assertEqual(getattr(PipelineStage, stage.upper()).value, stage)
        
        # Check total number of stages
        self.assertEqual(len(PipelineStage), len(expected_stages))


class TestPipelineStatus(unittest.TestCase):
    """Test the PipelineStatus enumeration."""
    
    def test_pipeline_statuses(self):
        """Test all defined pipeline statuses."""
        expected_statuses = [
            "pending", "running", "paused", 
            "completed", "failed", "canceled"
        ]
        
        # Check that all expected statuses are defined
        for status in expected_statuses:
            self.assertTrue(hasattr(PipelineStatus, status.upper()))
            self.assertEqual(getattr(PipelineStatus, status.upper()).value, status)
        
        # Check total number of statuses
        self.assertEqual(len(PipelineStatus), len(expected_statuses))


class TestPipelineConfigSchema(unittest.TestCase):
    """Test the PipelineConfigSchema functionality."""
    
    def test_config_instantiation(self):
        """Test that PipelineConfigSchema can be instantiated with required attributes."""
        # Test minimal config
        config = PipelineConfigSchema(
            name="test-pipeline"
        )
        
        self.assertEqual(config.name, "test-pipeline")
        self.assertIsNone(config.description)
        self.assertEqual(config.version, "1.0.0")  # default value
        self.assertEqual(len(config.enabled_stages), len(PipelineStage))  # all stages enabled by default
        self.assertFalse(config.parallel)  # default value
        self.assertEqual(config.max_workers, 4)  # default value
        self.assertEqual(config.timeout_seconds, 3600)  # default value
        self.assertEqual(config.retry_count, 3)  # default value
        self.assertEqual(config.metadata, {})  # default value
        
        # Test with all attributes
        config = PipelineConfigSchema(
            name="full-pipeline",
            description="A full pipeline configuration",
            version="2.0.0",
            enabled_stages=[PipelineStage.LOAD, PipelineStage.PREPROCESS, PipelineStage.CHUNK],
            parallel=True,
            max_workers=8,
            timeout_seconds=7200,
            retry_count=5,
            metadata={"owner": "test-user"}
        )
        
        self.assertEqual(config.name, "full-pipeline")
        self.assertEqual(config.description, "A full pipeline configuration")
        self.assertEqual(config.version, "2.0.0")
        self.assertEqual(len(config.enabled_stages), 3)
        self.assertIn(PipelineStage.LOAD, config.enabled_stages)
        self.assertIn(PipelineStage.PREPROCESS, config.enabled_stages)
        self.assertIn(PipelineStage.CHUNK, config.enabled_stages)
        self.assertTrue(config.parallel)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.timeout_seconds, 7200)
        self.assertEqual(config.retry_count, 5)
        self.assertEqual(config.metadata, {"owner": "test-user"})
    
    def test_max_workers_validation(self):
        """Test validation of max_workers parameter."""
        # Valid values
        config = PipelineConfigSchema(
            name="test",
            max_workers=1
        )
        self.assertEqual(config.max_workers, 1)
        
        config = PipelineConfigSchema(
            name="test",
            max_workers=10
        )
        self.assertEqual(config.max_workers, 10)
        
        # Invalid values
        with self.assertRaises(ValidationError):
            PipelineConfigSchema(
                name="test",
                max_workers=0
            )
        
        with self.assertRaises(ValidationError):
            PipelineConfigSchema(
                name="test",
                max_workers=-1
            )
    
    def test_retry_count_validation(self):
        """Test validation of retry_count parameter."""
        # Valid values
        config = PipelineConfigSchema(
            name="test",
            retry_count=0
        )
        self.assertEqual(config.retry_count, 0)
        
        config = PipelineConfigSchema(
            name="test",
            retry_count=10
        )
        self.assertEqual(config.retry_count, 10)
        
        # Invalid values
        with self.assertRaises(ValidationError):
            PipelineConfigSchema(
                name="test",
                retry_count=-1
            )


class TestPipelineStatsSchema(unittest.TestCase):
    """Test the PipelineStatsSchema functionality."""
    
    def test_stats_instantiation(self):
        """Test that PipelineStatsSchema can be instantiated with default values."""
        stats = PipelineStatsSchema()
        
        self.assertIsNone(stats.start_time)
        self.assertIsNone(stats.end_time)
        self.assertIsNone(stats.duration_seconds)
        self.assertEqual(stats.documents_processed, 0)
        self.assertEqual(stats.documents_failed, 0)
        self.assertEqual(stats.documents_skipped, 0)
        self.assertIsNone(stats.current_stage)
        self.assertEqual(stats.stage_stats, {})
        self.assertEqual(stats.errors, [])
    
    def test_record_start(self):
        """Test recording of pipeline start time."""
        stats = PipelineStatsSchema()
        self.assertIsNone(stats.start_time)
        
        # Record start time
        before = datetime.now()
        stats.record_start()
        after = datetime.now()
        
        # Verify start time was recorded
        self.assertIsNotNone(stats.start_time)
        self.assertTrue(before <= stats.start_time <= after)
    
    def test_record_end(self):
        """Test recording of pipeline end time and duration calculation."""
        stats = PipelineStatsSchema()
        
        # Record start time
        stats.record_start()
        start_time = stats.start_time
        
        # Wait a short time
        datetime.now()  # just to add a small delay
        
        # Record end time
        stats.record_end()
        
        # Verify end time and duration
        self.assertIsNotNone(stats.end_time)
        self.assertIsNotNone(stats.duration_seconds)
        self.assertTrue(stats.duration_seconds >= 0)
        self.assertTrue(stats.end_time > start_time)
        
        # Verify duration calculation
        expected_duration = (stats.end_time - start_time).total_seconds()
        self.assertEqual(stats.duration_seconds, expected_duration)
    
    def test_update_stage(self):
        """Test updating the current pipeline stage."""
        stats = PipelineStatsSchema()
        
        # Initial state
        self.assertIsNone(stats.current_stage)
        self.assertEqual(stats.stage_stats, {})
        
        # Update to LOAD stage - use string instead of enum
        stats.update_stage("load")
        
        # Verify stage update
        self.assertEqual(stats.current_stage, "load")
        self.assertIn("load", stats.stage_stats)
        self.assertIn("start_time", stats.stage_stats["load"])
        self.assertEqual(stats.stage_stats["load"]["documents_processed"], 0)
        self.assertEqual(stats.stage_stats["load"]["errors"], 0)
        
        # Update to PREPROCESS stage - use string instead of enum
        stats.update_stage("preprocess")
        
        # Verify stage update
        self.assertEqual(stats.current_stage, "preprocess")
        self.assertIn("preprocess", stats.stage_stats)
        self.assertEqual(len(stats.stage_stats), 2)  # Should have both stages
    
    def test_record_document_processed(self):
        """Test recording processed documents."""
        stats = PipelineStatsSchema()
        
        # Initial state
        self.assertEqual(stats.documents_processed, 0)
        
        # Record without stage (should not increment stage stats)
        stats.record_document_processed()
        self.assertEqual(stats.documents_processed, 1)
        
        # Set current stage and record - use string value instead of enum
        stats.update_stage("load")
        stats.record_document_processed()
        
        # Verify document count
        self.assertEqual(stats.documents_processed, 2)
        self.assertEqual(stats.stage_stats["load"]["documents_processed"], 1)
        
        # Record for specific stage - use string value instead of enum
        stats.record_document_processed(stage="preprocess")
        
        # Verify document count
        self.assertEqual(stats.documents_processed, 3)
        self.assertIn("preprocess", stats.stage_stats)
        self.assertEqual(stats.stage_stats["preprocess"]["documents_processed"], 1)
        self.assertEqual(stats.stage_stats["load"]["documents_processed"], 1)  # unchanged
    
    def test_record_error(self):
        """Test recording pipeline errors."""
        stats = PipelineStatsSchema()
        
        # Initial state
        self.assertEqual(len(stats.errors), 0)
        
        # Record error without stage or document
        stats.record_error("Test error")
        
        # Verify error recorded
        self.assertEqual(len(stats.errors), 1)
        self.assertEqual(stats.errors[0]["error"], "Test error")
        self.assertIn("timestamp", stats.errors[0])
        
        # Set current stage and record error with document ID - use string instead of enum
        stats.update_stage("load")
        stats.record_error("Document load error", document_id="doc123")
        
        # Verify error recorded with proper metadata
        self.assertEqual(len(stats.errors), 2)
        self.assertEqual(stats.errors[1]["error"], "Document load error")
        self.assertEqual(stats.errors[1]["document_id"], "doc123")
        self.assertEqual(stats.errors[1]["stage"], "load")
        self.assertEqual(stats.stage_stats["load"]["errors"], 1)
        
        # Record error for specific stage - use string instead of enum
        stats.record_error(
            "Preprocessing error", 
            document_id="doc456",
            stage="preprocess"
        )
        
        # Verify error recorded with proper stage
        self.assertEqual(len(stats.errors), 3)
        self.assertEqual(stats.errors[2]["stage"], "preprocess")
        self.assertIn("preprocess", stats.stage_stats)
        self.assertEqual(stats.stage_stats["preprocess"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
