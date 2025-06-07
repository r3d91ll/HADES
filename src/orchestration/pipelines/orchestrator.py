"""
Pipeline orchestrator for HADES-PathRAG.

This module provides stage-based pipeline orchestration with 
parallel processing capabilities.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, TypeVar, Generic, Sequence

from src.alerts import AlertManager, AlertLevel
from src.orchestration.core.parallel_worker import WorkerPool
from src.orchestration.core.queue.queue_manager import QueueManager
from src.orchestration.core.monitoring import PipelineMonitor

from .stages.base import PipelineStage, PipelineStageResult, PipelineStageError, PipelineStageStatus
from .schema import (
    ValidationResult, 
    ValidationSeverity, 
    ValidationIssue, 
    BatchProcessingStats,
    PipelineExecutionContext,
    PipelineExecutionResult
)

T = TypeVar('T')  # Type variable for pipeline input
U = TypeVar('U')  # Type variable for pipeline output

logger = logging.getLogger(__name__)


class PipelineExecutionMode(str):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"  # Process each stage sequentially
    BATCH = "batch"            # Process all documents in a batch
    PARALLEL = "parallel"      # Process documents in parallel
    STREAMING = "streaming"    # Process documents in a streaming fashion


class PipelineConfiguration:
    """Configuration for pipeline execution."""
    
    def __init__(
        self,
        execution_mode: str = PipelineExecutionMode.PARALLEL,
        stop_on_error: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        debug_mode: bool = False,
        alert_threshold: AlertLevel = AlertLevel.MEDIUM,
        alert_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 10,
        output_dir: Optional[str] = None,
        enable_parallel: bool = True,
        max_workers: int = 4,
        queue_size: int = 100,
        enable_monitoring: bool = True
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
            enable_parallel: Whether to enable parallel processing (default: True)
            max_workers: Maximum number of parallel workers
            queue_size: Size of task queues
            enable_monitoring: Whether to enable pipeline monitoring
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
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_monitoring = enable_monitoring


class Pipeline(Generic[T, U]):
    """Pipeline for coordinating multiple processing stages.
    
    The pipeline can operate in multiple modes:
    - Sequential: Process each document through all stages before starting the next
    - Batch: Process all documents in a batch before moving to the next stage
    - Parallel: Process documents in parallel using worker pools
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
        
        # Initialize orchestration components
        self._initialize_parallel_components()
        
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
        
        # Configure stages for parallel execution
        for stage in self.stages:
            if hasattr(stage, 'enable_parallel'):
                stage.enable_parallel = self.config.enable_parallel
            if hasattr(stage, 'worker_pool') and self.worker_pool:
                stage.worker_pool = self.worker_pool
            if hasattr(stage, 'queue_manager') and self.queue_manager:
                stage.queue_manager = self.queue_manager
    
    def _initialize_parallel_components(self):
        """Initialize parallel processing components."""
        if not self.config.enable_parallel:
            self.worker_pool = None
            self.queue_manager = None
            self.monitor = None
            return
            
        try:
            self.worker_pool = WorkerPool(
                max_workers=self.config.max_workers,
                name=f"{self.name}_workers"
            )
            
            self.queue_manager = QueueManager(
                name=f"{self.name}_queue",
                config={"max_size": self.config.queue_size}
            )
            
            if self.config.enable_monitoring:
                self.monitor = PipelineMonitor(
                    pipeline_name=self.name,
                    worker_pool=self.worker_pool,
                    queue_manager=self.queue_manager
                )
            else:
                self.monitor = None
            
            self.logger.info(f"Initialized parallel components for {self.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize parallel components: {str(e)}")
            self.worker_pool = None
            self.queue_manager = None
            self.monitor = None
    
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
        
        # Create execution context
        context = PipelineExecutionContext(
            pipeline_name=self.name,
            execution_mode=self.config.execution_mode,
            batch_size=self.config.batch_size,
            enable_monitoring=self.config.enable_monitoring,
            enable_alerts=bool(self.alert_manager),
            output_dir=self.config.output_dir
        )
        
        # Start monitoring if enabled
        if self.monitor:
            self.monitor.start_monitoring()
        
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
        
        # Stop monitoring if enabled
        if self.monitor:
            self.monitor.stop_monitoring()
        
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
    
    def execute_batch(self, batch_data: Sequence[T]) -> PipelineExecutionResult:
        """Execute the pipeline on a batch of input data.
        
        This method processes multiple inputs, leveraging parallel processing
        capabilities when available and configured.
        
        Args:
            batch_data: Sequence of input data items
            
        Returns:
            PipelineExecutionResult with comprehensive execution details
        """
        start_time = time.time()
        self.logger.info(f"Starting batch execution of pipeline: {self.name}")
        self.logger.info(f"Batch size: {len(batch_data)}")
        
        # Create execution context
        context = PipelineExecutionContext(
            pipeline_name=self.name,
            execution_mode=self.config.execution_mode,
            batch_size=len(batch_data),
            enable_monitoring=self.config.enable_monitoring,
            enable_alerts=bool(self.alert_manager)
        )
        
        # Initialize stats
        self.stats.total_documents = len(batch_data)
        
        # Start monitoring if enabled
        if self.monitor:
            self.monitor.start_monitoring()
        
        # Process batch based on execution mode
        if self.config.execution_mode == PipelineExecutionMode.PARALLEL:
            results = self._execute_batch_parallel(batch_data)
        else:
            results = self._execute_batch_sequential(batch_data)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        self.stats.processing_times["total"] = execution_time
        
        # Stop monitoring if enabled
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Determine overall success
        successful_results = [r for r in results if r.status == PipelineStageStatus.COMPLETED]
        success = len(successful_results) == len(results)
        
        # Create error summary if needed
        error_summary = None
        if not success:
            failed_count = len(results) - len(successful_results)
            error_summary = f"{failed_count}/{len(results)} items failed processing"
        
        # Log completion
        self.logger.info(f"Batch execution completed in {execution_time:.2f}s")
        self.logger.info(f"Processed {len(successful_results)}/{len(results)} items successfully")
        
        return PipelineExecutionResult(
            pipeline_name=self.name,
            execution_context=context,
            total_execution_time=execution_time,
            stage_results=[result.to_dict() for result in results],
            batch_stats=self.stats,
            success=success,
            error_summary=error_summary
        )
    
    def _execute_batch_sequential(self, batch_data: Sequence[T]) -> List[PipelineStageResult]:
        """Execute batch using sequential processing."""
        results = []
        for i, item in enumerate(batch_data):
            self.logger.info(f"Processing item {i+1}/{len(batch_data)}")
            result = self.execute(item)
            results.append(result)
            
            # Update stats for successful processing
            if result.status == PipelineStageStatus.COMPLETED and result.data:
                self._update_stats_from_result(result)
        
        return results
    
    def _execute_batch_parallel(self, batch_data: Sequence[T]) -> List[PipelineStageResult]:
        """Execute batch using parallel processing."""
        if not self.worker_pool:
            raise RuntimeError("Parallel processing requested but worker pool not initialized")
        
        # Submit all items to worker pool
        futures = []
        for item in batch_data:
            future = self.worker_pool.submit(self.execute, item)
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                
                # Update stats for successful processing
                if result.status == PipelineStageStatus.COMPLETED and result.data:
                    self._update_stats_from_result(result)
                    
                self.logger.info(f"Completed parallel item {i+1}/{len(batch_data)}")
                
            except Exception as e:
                # Create error result
                error_result = PipelineStageResult(
                    stage_name=self.name,
                    status=PipelineStageStatus.FAILED,
                    error=PipelineStageError(
                        self.name,
                        f"Parallel execution failed: {str(e)}",
                        original_error=e
                    )
                )
                results.append(error_result)
                self.logger.error(f"Failed parallel item {i+1}/{len(batch_data)}: {str(e)}")
        
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
    
    def _update_stats_from_result(self, result: PipelineStageResult):
        """Update statistics from a successful result."""
        try:
            # Check if the data is a DocumentSchema or has the necessary attributes
            if hasattr(result.data, 'total_chunks'):
                self.stats.add_document_stats(result.data)
            else:
                # Just increment the count for non-document objects
                self.stats.documents_processed += 1
        except Exception as e:
            self.logger.warning(f"Failed to update stats: {str(e)}")
    
    def cleanup(self):
        """Clean up pipeline resources."""
        if self.worker_pool:
            self.worker_pool.shutdown()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        self.logger.info(f"Pipeline {self.name} resources cleaned up")