"""
Unit tests for the pipeline orchestrator.

These tests verify the functionality of the Pipeline orchestrator class
that coordinates the execution of multiple pipeline stages.
"""

import unittest
import time
from typing import Dict, Any, List, Tuple

from src.pipeline.stages import PipelineStage, PipelineStageError, PipelineStageResult
from src.pipeline.stages.base import PipelineStageStatus
from src.pipeline.orchestrator import Pipeline, PipelineConfiguration


class TestStage(PipelineStage):
    """Test implementation of PipelineStage for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, validation_errors: List[str] = None, delay: float = 0):
        """Initialize test stage.
        
        Args:
            name: Stage name
            should_fail: Whether the stage should fail execution
            validation_errors: List of validation errors to return
            delay: Execution delay in seconds
        """
        super().__init__(name, {})
        self.should_fail = should_fail
        self.validation_errors = validation_errors or []
        self.delay = delay
        self.was_run = False
        self.execution_count = 0
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the test stage.
        
        Args:
            input_data: Input data for the stage
            
        Returns:
            Modified input data with stage name added
            
        Raises:
            PipelineStageError: If should_fail is True
        """
        self.was_run = True
        self.execution_count += 1
        
        # Add artificial delay if specified
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.should_fail:
            raise PipelineStageError(self.name, "Test stage failure")
        
        # Modify the input data to indicate this stage was run
        result = input_data.copy()
        result["stages_run"] = result.get("stages_run", []) + [self.name]
        
        return result
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data for the test stage.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if self.validation_errors:
            return False, self.validation_errors
        
        return True, []


class TransformStage(TestStage):
    """Test stage that transforms data in a specific way."""
    
    def __init__(self, name: str, transform_key: str, transform_value: Any):
        """Initialize transform stage.
        
        Args:
            name: Stage name
            transform_key: Key to transform
            transform_value: Value to set
        """
        super().__init__(name)
        self.transform_key = transform_key
        self.transform_value = transform_value
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the transform stage.
        
        Args:
            input_data: Input data for the stage
            
        Returns:
            Modified input data with transformation applied
        """
        result = super().run(input_data)
        result[self.transform_key] = self.transform_value
        return result


class PipelineOrchestratorTests(unittest.TestCase):
    """Tests for the Pipeline orchestrator."""
    
    def test_simple_pipeline_execution(self):
        """Test execution of a simple pipeline with multiple stages."""
        # Create test stages
        stage1 = TestStage("stage1")
        stage2 = TestStage("stage2")
        stage3 = TestStage("stage3")
        
        # Create pipeline
        pipeline = Pipeline(
            stages=[stage1, stage2, stage3],
            name="test_pipeline"
        )
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = pipeline.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.data["test"], "data")
        self.assertEqual(result.data["stages_run"], ["stage1", "stage2", "stage3"])
        
        # Verify stages were run
        self.assertTrue(stage1.was_run)
        self.assertTrue(stage2.was_run)
        self.assertTrue(stage3.was_run)
    
    def test_pipeline_with_failing_stage(self):
        """Test pipeline execution with a failing stage."""
        # Create test stages
        stage1 = TestStage("stage1")
        stage2 = TestStage("stage2", should_fail=True)
        stage3 = TestStage("stage3")
        
        # Create pipeline with default configuration (stop_on_error=True)
        pipeline = Pipeline(
            stages=[stage1, stage2, stage3],
            name="failing_pipeline"
        )
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = pipeline.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.FAILED)
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)
        
        # Verify stages execution
        self.assertTrue(stage1.was_run)
        self.assertTrue(stage2.was_run)
        self.assertFalse(stage3.was_run)  # Should not run after failure
    
    def test_pipeline_continue_on_error(self):
        """Test pipeline execution with continue_on_error=True."""
        # Create test stages
        stage1 = TestStage("stage1")
        stage2 = TestStage("stage2", should_fail=True)
        stage3 = TestStage("stage3")
        
        # Create pipeline configured to continue on error
        pipeline = Pipeline(
            stages=[stage1, stage2, stage3],
            config=PipelineConfiguration(stop_on_error=False),
            name="continue_pipeline"
        )
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = pipeline.execute(input_data)
        
        # Verify the result (should be from the last stage that ran successfully)
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertIsNotNone(result.data)
        
        # Verify all stages were attempted
        self.assertTrue(stage1.was_run)
        self.assertTrue(stage2.was_run)
        self.assertTrue(stage3.was_run)
    
    def test_pipeline_with_data_transformation(self):
        """Test pipeline that transforms data through stages."""
        # Create test stages that transform data
        stage1 = TransformStage("stage1", "key1", "value1")
        stage2 = TransformStage("stage2", "key2", "value2")
        stage3 = TransformStage("stage3", "key3", "value3")
        
        # Create pipeline
        pipeline = Pipeline(
            stages=[stage1, stage2, stage3],
            name="transform_pipeline"
        )
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = pipeline.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertIsNotNone(result.data)
        
        # Check transformed data
        self.assertEqual(result.data["test"], "data")
        self.assertEqual(result.data["key1"], "value1")
        self.assertEqual(result.data["key2"], "value2")
        self.assertEqual(result.data["key3"], "value3")
        self.assertEqual(result.data["stages_run"], ["stage1", "stage2", "stage3"])
    
    def test_pipeline_retry_logic(self):
        """Test pipeline retry logic for failing stages."""
        # Create a test stage that fails on first attempt but succeeds on retry
        class RetryStage(TestStage):
            def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                self.execution_count += 1
                if self.execution_count == 1:
                    raise PipelineStageError(self.name, "First attempt failure")
                return super().run(input_data)
        
        # Create stages
        stage1 = TestStage("stage1")
        retry_stage = RetryStage("retry_stage")
        stage3 = TestStage("stage3")
        
        # Create pipeline with retry configuration
        pipeline = Pipeline(
            stages=[stage1, retry_stage, stage3],
            config=PipelineConfiguration(max_retries=2, retry_delay=0.1),
            name="retry_pipeline"
        )
        
        # Execute pipeline
        input_data = {"test": "data"}
        result = pipeline.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertIsNotNone(result.data)
        
        # Verify stages execution
        self.assertTrue(stage1.was_run)
        self.assertEqual(retry_stage.execution_count, 3)  # Should have been executed with 1 failure + 1 retry + 1 success
        self.assertTrue(stage3.was_run)
        
        # Verify the final data includes all stages
        self.assertEqual(result.data["stages_run"], ["stage1", "retry_stage", "stage3"])
    
    def test_batch_execution(self):
        """Test batch execution of the pipeline."""
        # Create test stages
        stage1 = TestStage("stage1")
        stage2 = TestStage("stage2")
        
        # Create pipeline
        pipeline = Pipeline(
            stages=[stage1, stage2],
            name="batch_pipeline"
        )
        
        # Execute pipeline with batch
        batch_data = [
            {"id": 1, "test": "data1"},
            {"id": 2, "test": "data2"},
            {"id": 3, "test": "data3"}
        ]
        results = pipeline.execute_batch(batch_data)
        
        # Verify results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
            self.assertIsNotNone(result.data)
            self.assertEqual(result.data["id"], i+1)
            self.assertEqual(result.data["test"], f"data{i+1}")
            self.assertEqual(result.data["stages_run"], ["stage1", "stage2"])
        
        # Verify stages execution counts
        self.assertEqual(stage1.execution_count, 3)
        self.assertEqual(stage2.execution_count, 3)
        
        # Verify stats
        self.assertEqual(pipeline.stats.total_documents, 3)


if __name__ == "__main__":
    unittest.main()
