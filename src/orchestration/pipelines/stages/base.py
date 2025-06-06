"""
Base classes for pipeline stages in the unified orchestration system.

This module consolidates the stage-based architecture from src/pipeline/stages/base.py
with the orchestration system's parallel processing capabilities.
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union, TypeVar, Generic

from src.orchestration.core.parallel_worker import WorkerPool
from src.orchestration.core.queue.queue_manager import QueueManager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for stage input
U = TypeVar('U')  # Type variable for stage output


class PipelineStageStatus(str, Enum):
    """Status values for pipeline stage execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStageError(Exception):
    """Exception raised for errors in pipeline stages.
    
    Attributes:
        stage_name: Name of the stage where the error occurred
        message: Explanation of the error
        original_error: The original exception that was caught
        context: Additional context information about the error
    """
    
    def __init__(
        self, 
        stage_name: str, 
        message: str, 
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize PipelineStageError.
        
        Args:
            stage_name: Name of the stage where the error occurred
            message: Explanation of the error
            original_error: The original exception that was caught
            context: Additional context information about the error
        """
        self.stage_name = stage_name
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
        # Format detailed error message
        detailed_message = f"Error in stage '{stage_name}': {message}"
        if original_error:
            detailed_message += f"\nOriginal error: {str(original_error)}"
            
        super().__init__(detailed_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation.
        
        Returns:
            Dictionary containing error details
        """
        error_dict = {
            "stage_name": self.stage_name,
            "message": str(self),
            "timestamp": self.timestamp,
            "context": self.context
        }
        
        if self.original_error:
            error_dict["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error),
                "traceback": traceback.format_exc()
            }
            
        return error_dict


class PipelineStageResult(Generic[T]):
    """Result of a pipeline stage execution.
    
    This class encapsulates the result of executing a pipeline stage,
    including status, execution time, and any errors that occurred.
    
    Attributes:
        stage_name: Name of the stage that produced this result
        status: Execution status of the stage
        data: Output data from the stage (if successful)
        error: Error information (if failed)
        start_time: When the stage execution started
        end_time: When the stage execution ended
        execution_time: Total execution time in seconds
        metadata: Additional metadata about the execution
    """
    
    def __init__(
        self,
        stage_name: str,
        status: PipelineStageStatus,
        data: Optional[T] = None,
        error: Optional[PipelineStageError] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize PipelineStageResult.
        
        Args:
            stage_name: Name of the stage that produced this result
            status: Execution status of the stage
            data: Output data from the stage (if successful)
            error: Error information (if failed)
            start_time: When the stage execution started
            end_time: When the stage execution ended
            metadata: Additional metadata about the execution
        """
        self.stage_name = stage_name
        self.status = status
        self.data = data
        self.error = error
        self.start_time = start_time
        self.end_time = end_time
        self.execution_time = (end_time - start_time) if (start_time and end_time) else None
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation.
        
        Returns:
            Dictionary containing result details
        """
        result_dict = {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }
        
        if self.error:
            result_dict["error"] = self.error.to_dict()
            
        return result_dict
    
    def __bool__(self) -> bool:
        """Boolean representation of result success.
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        return self.status == PipelineStageStatus.COMPLETED


class PipelineStage(ABC, Generic[T, U]):
    """Abstract base class for all pipeline stages in the unified system.
    
    This class combines the stage-based processing from the original pipeline
    with the parallel processing capabilities of the orchestration system.
    
    Each stage is responsible for a specific aspect of document processing,
    such as document parsing, chunking, embedding generation, or storage.
    
    Stages can operate in both sequential and parallel modes depending on
    the pipeline configuration.
    """
    
    def __init__(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False,
        worker_pool: Optional[WorkerPool] = None,
        queue_manager: Optional[QueueManager] = None
    ):
        """Initialize pipeline stage.
        
        Args:
            name: Unique name for this pipeline stage
            config: Configuration dictionary for this stage
            enable_parallel: Whether to enable parallel processing
            worker_pool: Worker pool for parallel execution
            queue_manager: Queue manager for task distribution
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.enable_parallel = enable_parallel
        self.worker_pool = worker_pool
        self.queue_manager = queue_manager
    
    @abstractmethod
    def run(self, input_data: T) -> U:
        """Execute the pipeline stage on input data.
        
        This method must be implemented by all concrete stages to perform
        the actual processing work.
        
        Args:
            input_data: Input data for this stage
            
        Returns:
            Processed output data
            
        Raises:
            PipelineStageError: If an error occurs during processing
        """
        pass
    
    @abstractmethod
    def validate(self, data: Union[T, U]) -> Tuple[bool, List[str]]:
        """Validate data structure for this stage.
        
        This method must be implemented by all concrete stages to validate
        either input or output data for the stage.
        
        Args:
            data: Data structure to validate (either input or output)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    def run_parallel(self, input_batch: List[T]) -> List[U]:
        """Execute the stage on a batch of input data in parallel.
        
        This method leverages the orchestration system's parallel processing
        capabilities to process multiple inputs concurrently.
        
        Args:
            input_batch: List of input data items
            
        Returns:
            List of processed output data items
            
        Raises:
            PipelineStageError: If parallel processing is not available
        """
        if not self.enable_parallel or not self.worker_pool:
            # Fall back to sequential processing
            return [self.run(item) for item in input_batch]
        
        # Submit tasks to worker pool
        futures = []
        for item in input_batch:
            future = self.worker_pool.submit(self.run, item)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise PipelineStageError(
                    self.name,
                    f"Parallel execution failed: {str(e)}",
                    original_error=e
                )
        
        return results
    
    def execute(self, input_data: T) -> PipelineStageResult[U]:
        """Execute the stage with timing, validation, and error handling.
        
        This is the main entry point for running a pipeline stage. It:
        1. Validates the input data
        2. Executes the stage's run method
        3. Validates the output data
        4. Captures timing and error information
        
        Args:
            input_data: Input data for this stage
            
        Returns:
            PipelineStageResult containing execution status and results
        """
        start_time = time.time()
        metadata = {"stage_name": self.name}
        
        try:
            # Set status to running
            self.logger.info(f"Starting stage: {self.name}")
            status = PipelineStageStatus.RUNNING
            
            # Validate input data
            is_valid, error_messages = self.validate(input_data)
            if not is_valid:
                error_str = "; ".join(error_messages)
                raise PipelineStageError(self.name, f"Input validation failed: {error_str}")
            
            # Execute the stage
            result = self.run(input_data)
            
            # Validate output data
            is_valid, error_messages = self.validate(result)
            if not is_valid:
                error_str = "; ".join(error_messages)
                raise PipelineStageError(self.name, f"Output validation failed: {error_str}")
            
            # Set status to completed
            status = PipelineStageStatus.COMPLETED
            end_time = time.time()
            
            # Add execution time to metadata
            execution_time = end_time - start_time
            metadata["execution_time"] = execution_time
            self.logger.info(f"Completed stage: {self.name} in {execution_time:.2f}s")
            
            return PipelineStageResult(
                stage_name=self.name,
                status=status,
                data=result,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
            
        except Exception as e:
            # Log the error
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.error(f"Error in stage {self.name}: {str(e)}", exc_info=True)
            
            # Wrap in PipelineStageError if it's not already
            if not isinstance(e, PipelineStageError):
                error = PipelineStageError(
                    stage_name=self.name,
                    message=f"Failed to execute stage: {str(e)}",
                    original_error=e,
                    context={"execution_time": execution_time}
                )
            else:
                error = e
            
            # Return error result
            return PipelineStageResult(
                stage_name=self.name,
                status=PipelineStageStatus.FAILED,
                error=error,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata
            )
    
    def execute_batch(self, input_batch: List[T]) -> List[PipelineStageResult[U]]:
        """Execute the stage on a batch of input data.
        
        This method processes multiple inputs, optionally using parallel
        processing if enabled.
        
        Args:
            input_batch: List of input data items
            
        Returns:
            List of PipelineStageResult objects
        """
        if self.enable_parallel and len(input_batch) > 1:
            # Use parallel execution for better performance
            try:
                results = self.run_parallel(input_batch)
                return [
                    PipelineStageResult(
                        stage_name=self.name,
                        status=PipelineStageStatus.COMPLETED,
                        data=result
                    )
                    for result in results
                ]
            except Exception as e:
                # Fall back to sequential processing
                self.logger.warning(f"Parallel execution failed, falling back to sequential: {str(e)}")
        
        # Sequential execution
        return [self.execute(item) for item in input_batch]