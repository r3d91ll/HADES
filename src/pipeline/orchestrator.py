"""
Pipeline orchestration for HADES-PathRAG.

This module provides the Pipeline class that coordinates the execution
of multiple pipeline stages, handling data flow, error recovery, and
resource management.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, TypeVar, Generic, Sequence

from src.alerts import AlertManager, AlertLevel
from .stages import PipelineStage, PipelineStageResult, PipelineStageError
from .stages.base import PipelineStageStatus
from .schema import ValidationResult, ValidationSeverity, ValidationIssue, BatchProcessingStats

T = TypeVar('T')  # Type variable for pipeline input
U = TypeVar('U')  # Type variable for pipeline output

logger = logging.getLogger(__name__)


class PipelineExecutionMode(str):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"  # Process each stage sequentially
    BATCH = "batch"            # Process all documents in a batch
    STREAMING = "streaming"    # Process documents in a streaming fashion


class PipelineConfiguration:
    """Configuration for pipeline execution."""
    
    def __init__(
        self,
        execution_mode: str = PipelineExecutionMode.SEQUENTIAL,
        stop_on_error: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug_mode: bool = False,
        alert_threshold: AlertLevel = AlertLevel.MEDIUM,
        alert_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 10,
        output_dir: Optional[str] = None
    ):
        """Initialize pipeline configuration.
        
        Args:
            execution_mode: Mode of pipeline execution
            stop_on_error: Whether to stop pipeline on first error
            max_retries: Maximum number of retries for failed stages
            retry_delay: Delay between retries in seconds
            debug_mode: Whether to enable debug mode with artifacts
            alert_threshold: Minimum alert level to report
            alert_config: Configuration for AlertManager
            batch_size: Size of document batches for processing
            output_dir: Directory for output artifacts
        """
        self.execution_mode = execution_mode
        self.stop_on_error = stop_on_error
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug_mode = debug_mode
        self.alert_threshold = alert_threshold
        self.alert_config = alert_config or {}
        self.batch_size = batch_size
        self.output_dir = output_dir


class Pipeline(Generic[T, U]):
    """Pipeline for coordinating multiple processing stages.
    
    This class orchestrates the execution of a sequence of pipeline stages,
    handling data flow between stages, error recovery, and resource management.
    
    The pipeline can operate in different modes:
    - Sequential: Process each document through all stages before starting the next
    - Batch: Process all documents in a batch before moving to the next stage
    - Streaming: Process documents in a streaming fashion as they become available
    """
    
    def __init__(
        self,
        stages: List[PipelineStage],
        config: Optional[PipelineConfiguration] = None,
        name: str = "Pipeline"
    ):
        """Initialize pipeline with stages.
        
        Args:
            stages: List of pipeline stages to execute
            config: Configuration for pipeline execution
            name: Name for this pipeline instance
        """
        self.stages = stages
        self.config = config or PipelineConfiguration()
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize alert manager if needed
        if self.config.alert_config:
            self.alert_manager = AlertManager(
                alert_dir=self.config.alert_config.get("alert_dir", "./alerts"),
                min_level=self.config.alert_threshold,
                email_config=self.config.alert_config.get("email_config")
            )
        else:
            self.alert_manager = None
        
        # Statistics
        self.stats = BatchProcessingStats()
    
    def execute(self, input_data: T) -> PipelineStageResult[U]:
        """Execute the pipeline on input data.
        
        This method runs the input data through all stages in sequence,
        passing the output of each stage as input to the next.
        
        Args:
            input_data: Input data for the pipeline
            
        Returns:
            Result from the final stage
        """
        start_time = time.time()
        self.logger.info(f"Starting pipeline: {self.name}")
        
        # Track stage results
        stage_results = []
        current_data = input_data
        
        # Execute each stage in sequence
        for stage in self.stages:
            self.logger.info(f"Executing stage: {stage.name}")
            
            # Try to execute with retries
            result = self._execute_with_retries(stage, current_data)
            stage_results.append(result)
            
            # Check if stage failed
            if result.status != PipelineStageStatus.COMPLETED:
                self.logger.error(f"Stage {stage.name} failed, stopping pipeline")
                
                # Create alert if enabled
                if self.alert_manager and result.error:
                    self._create_alert_for_error(result.error)
                
                # Stop pipeline if configured to do so
                if self.config.stop_on_error:
                    break
            
            # Update current data for next stage
            if result.data is not None:
                current_data = result.data
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log completion
        self.logger.info(f"Pipeline {self.name} completed in {execution_time:.2f}s")
        
        # Return result from last stage or error result
        final_result = stage_results[-1] if stage_results else None
        if not final_result:
            # Create a failed result if no stages were executed
            final_result = PipelineStageResult(
                stage_name=self.name,
                status=PipelineStageStatus.FAILED,
                error=PipelineStageError(
                    stage_name=self.name,
                    message="No stages were executed"
                ),
                start_time=start_time,
                end_time=end_time
            )
        
        return final_result
    
    def execute_batch(self, batch_data: Sequence[T]) -> List[PipelineStageResult[U]]:
        """Execute the pipeline on a batch of input data.
        
        This method processes each item in the batch through the complete
        pipeline before moving to the next item.
        
        Args:
            batch_data: Sequence of input data items
            
        Returns:
            List of results from final stage for each input item
        """
        start_time = time.time()
        self.logger.info(f"Starting batch execution of pipeline: {self.name}")
        self.logger.info(f"Batch size: {len(batch_data)}")
        
        # Initialize stats
        self.stats.total_documents = len(batch_data)
        
        # Process each item in the batch
        results = []
        for i, item in enumerate(batch_data):
            self.logger.info(f"Processing item {i+1}/{len(batch_data)}")
            result = self.execute(item)
            results.append(result)
            
            # Update stats for successful processing
            if result.status == PipelineStageStatus.COMPLETED and result.data:
                try:
                    # Check if the data is a DocumentSchema or has the necessary attributes
                    if hasattr(result.data, 'total_chunks'):
                        self.stats.add_document_stats(result.data)
                    else:
                        # Just increment the count for non-document objects
                        self.stats.documents_processed += 1
                except Exception as e:
                    self.logger.warning(f"Failed to update stats: {str(e)}")
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        self.stats.processing_times["total"] = execution_time
        
        # Log completion
        self.logger.info(f"Batch execution completed in {execution_time:.2f}s")
        self.logger.info(f"Processed {self.stats.documents_processed}/{self.stats.total_documents} documents")
        
        return results
    
    def _execute_with_retries(
        self, 
        stage: PipelineStage, 
        input_data: Any
    ) -> PipelineStageResult:
        """Execute a stage with retry logic.
        
        Args:
            stage: Pipeline stage to execute
            input_data: Input data for the stage
            
        Returns:
            Stage execution result
        """
        retries = 0
        result = None
        
        while retries <= self.config.max_retries:
            # Execute the stage
            result = stage.execute(input_data)
            
            # Return if successful
            if result.status == PipelineStageStatus.COMPLETED:
                return result
            
            # Increment retry counter
            retries += 1
            
            # Check if we should retry
            if retries <= self.config.max_retries:
                retry_delay = self.config.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                self.logger.warning(
                    f"Stage {stage.name} failed, retrying in {retry_delay:.2f}s "
                    f"(attempt {retries}/{self.config.max_retries})"
                )
                time.sleep(retry_delay)
            else:
                self.logger.error(
                    f"Stage {stage.name} failed after {retries} attempts, giving up"
                )
        
        return result
    
    def _create_alert_for_error(self, error: PipelineStageError):
        """Create an alert for a pipeline error.
        
        Args:
            error: The error that occurred
        """
        if not self.alert_manager:
            return
            
        # Determine alert level based on severity
        alert_level = AlertLevel.HIGH
        
        # Create alert context
        context = {
            "pipeline_name": self.name,
            "stage_name": error.stage_name,
            "timestamp": datetime.now().isoformat(),
            "error_message": str(error),
        }
        
        # Add original error information if available
        if error.original_error:
            context["original_error"] = {
                "type": type(error.original_error).__name__,
                "message": str(error.original_error)
            }
        
        # Add any additional context from the error
        if error.context:
            context.update(error.context)
        
        # Create the alert
        self.alert_manager.create_alert(
            level=alert_level,
            message=f"Pipeline error in stage {error.stage_name}: {str(error)}",
            context=context
        )
