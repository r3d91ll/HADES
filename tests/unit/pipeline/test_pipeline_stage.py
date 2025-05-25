"""
Unit tests for the pipeline stage base classes.

These tests verify the functionality of the PipelineStage abstract base class
and its supporting classes for error handling and result tracking.
"""

import unittest
import time
from typing import Dict, Any, List, Tuple

from src.pipeline.stages import PipelineStage, PipelineStageError, PipelineStageResult
from src.pipeline.stages.base import PipelineStageStatus


class TestStage(PipelineStage):
    """Test implementation of PipelineStage for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, validation_errors: List[str] = None):
        """Initialize test stage.
        
        Args:
            name: Stage name
            should_fail: Whether the stage should fail execution
            validation_errors: List of validation errors to return
        """
        super().__init__(name, {})
        self.should_fail = should_fail
        self.validation_errors = validation_errors or []
        self.was_run = False
    
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


class PipelineStageTests(unittest.TestCase):
    """Tests for the PipelineStage class and related components."""
    
    def test_successful_stage_execution(self):
        """Test successful execution of a pipeline stage."""
        # Create a test stage
        stage = TestStage("test_stage")
        
        # Execute the stage
        input_data = {"test": "data"}
        result = stage.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertTrue(stage.was_run)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.data["test"], "data")
        self.assertEqual(result.data["stages_run"], ["test_stage"])
        self.assertGreater(result.execution_time, 0)
    
    def test_failed_stage_execution(self):
        """Test failed execution of a pipeline stage."""
        # Create a test stage that will fail
        stage = TestStage("failing_stage", should_fail=True)
        
        # Execute the stage
        input_data = {"test": "data"}
        result = stage.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.FAILED)
        self.assertTrue(stage.was_run)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.data)
        self.assertEqual(result.error.stage_name, "failing_stage")
        self.assertIn("Test stage failure", str(result.error))
        self.assertGreater(result.execution_time, 0)
    
    def test_validation_failure(self):
        """Test stage execution with validation failures."""
        # Create a test stage with validation errors
        stage = TestStage("validation_stage", validation_errors=["Invalid data format"])
        
        # Execute the stage
        input_data = {"test": "data"}
        result = stage.execute(input_data)
        
        # Verify the result
        self.assertEqual(result.status, PipelineStageStatus.FAILED)
        self.assertFalse(stage.was_run)  # Stage should not run if validation fails
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.data)
        self.assertEqual(result.error.stage_name, "validation_stage")
        self.assertIn("Input validation failed", str(result.error))
        self.assertIn("Invalid data format", str(result.error))
    
    def test_pipeline_stage_error(self):
        """Test PipelineStageError functionality."""
        # Create a pipeline stage error
        original_error = ValueError("Original error")
        context = {"test_key": "test_value"}
        error = PipelineStageError(
            stage_name="test_stage",
            message="Test error message",
            original_error=original_error,
            context=context
        )
        
        # Verify error properties
        self.assertEqual(error.stage_name, "test_stage")
        self.assertEqual(error.original_error, original_error)
        self.assertEqual(error.context, context)
        self.assertIn("test_stage", str(error))
        self.assertIn("Test error message", str(error))
        self.assertIn("Original error", str(error))
        
        # Test conversion to dictionary
        error_dict = error.to_dict()
        self.assertEqual(error_dict["stage_name"], "test_stage")
        self.assertIn("message", error_dict)
        self.assertIn("timestamp", error_dict)
        self.assertEqual(error_dict["context"], context)
        self.assertIn("original_error", error_dict)
        self.assertEqual(error_dict["original_error"]["type"], "ValueError")
        self.assertEqual(error_dict["original_error"]["message"], "Original error")
    
    def test_pipeline_stage_result(self):
        """Test PipelineStageResult functionality."""
        # Create a successful result
        start_time = time.time()
        end_time = start_time + 1.5
        result = PipelineStageResult(
            stage_name="test_stage",
            status=PipelineStageStatus.COMPLETED,
            data={"test": "data"},
            start_time=start_time,
            end_time=end_time,
            metadata={"test_meta": "value"}
        )
        
        # Verify result properties
        self.assertEqual(result.stage_name, "test_stage")
        self.assertEqual(result.status, PipelineStageStatus.COMPLETED)
        self.assertEqual(result.data, {"test": "data"})
        self.assertIsNone(result.error)
        self.assertEqual(result.start_time, start_time)
        self.assertEqual(result.end_time, end_time)
        self.assertAlmostEqual(result.execution_time, 1.5, places=1)
        self.assertEqual(result.metadata, {"test_meta": "value"})
        
        # Test boolean evaluation
        self.assertTrue(result)
        
        # Test conversion to dictionary
        result_dict = result.to_dict()
        self.assertEqual(result_dict["stage_name"], "test_stage")
        self.assertEqual(result_dict["status"], "completed")
        self.assertIn("timestamp", result_dict)
        self.assertAlmostEqual(result_dict["execution_time"], 1.5, places=1)
        self.assertEqual(result_dict["metadata"], {"test_meta": "value"})
        
        # Create a failed result
        error = PipelineStageError("test_stage", "Test error")
        failed_result = PipelineStageResult(
            stage_name="test_stage",
            status=PipelineStageStatus.FAILED,
            error=error,
            start_time=start_time,
            end_time=end_time
        )
        
        # Verify failed result properties
        self.assertEqual(failed_result.status, PipelineStageStatus.FAILED)
        self.assertIsNone(failed_result.data)
        self.assertEqual(failed_result.error, error)
        
        # Test boolean evaluation
        self.assertFalse(failed_result)
        
        # Test conversion to dictionary
        failed_dict = failed_result.to_dict()
        self.assertEqual(failed_dict["status"], "failed")
        self.assertIn("error", failed_dict)


if __name__ == "__main__":
    unittest.main()
