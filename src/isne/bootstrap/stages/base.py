"""
Base class for ISNE Bootstrap Pipeline Stages

Provides common interface and functionality for all pipeline stages.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseBootstrapStage(ABC):
    """Base class for all bootstrap pipeline stages."""
    
    def __init__(self, stage_name: str):
        """Initialize base stage."""
        self.stage_name = stage_name
        self.logger = logging.getLogger(f"{__name__}.{stage_name}")
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the stage with given inputs.
        
        This method should be implemented by each stage to perform
        its specific processing logic.
        
        Returns:
            Stage-specific result object
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> List[str]:
        """
        Validate inputs for the stage.
        
        Returns:
            List of validation error messages (empty list if valid)
        """
        pass
    
    @abstractmethod
    def get_expected_outputs(self) -> List[str]:
        """
        Get list of expected output keys/attributes.
        
        Returns:
            List of expected output keys
        """
        pass
    
    @abstractmethod
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Size metric for the input (files, documents, etc.)
            
        Returns:
            Estimated duration in seconds
        """
        pass
    
    @abstractmethod
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements for the stage.
        
        Args:
            input_size: Size metric for the input
            
        Returns:
            Dictionary with resource estimates (memory_mb, cpu_cores, etc.)
        """
        pass
    
    def pre_execute_checks(self, *args, **kwargs) -> List[str]:
        """
        Perform pre-execution checks.
        
        This method can be overridden by stages that need additional
        pre-execution validation beyond input validation.
        
        Returns:
            List of check failure messages
        """
        return []
    
    def post_execute_cleanup(self, result: Any) -> None:
        """
        Perform post-execution cleanup.
        
        This method can be overridden by stages that need to clean up
        resources after execution.
        
        Args:
            result: The result returned by execute()
        """
        pass
    
    def get_stage_info(self) -> Dict[str, Any]:
        """
        Get general information about the stage.
        
        Returns:
            Dictionary with stage information
        """
        return {
            "stage_name": self.stage_name,
            "class_name": self.__class__.__name__,
            "expected_outputs": self.get_expected_outputs()
        }