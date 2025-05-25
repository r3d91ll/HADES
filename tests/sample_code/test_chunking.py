"""
Test file with a variety of Python constructs to verify AST-based code chunking.

This file intentionally includes:
- Module level imports
- Classes with methods of varying sizes
- Nested classes
- Standalone functions
- Classes with docstrings
- Long comments
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field


# A long comment block to test chunking behavior with non-code content
# This comment should be included with nearby code or in its own chunk
# depending on the chunker configuration.
# The goal is to ensure that comments are handled properly
# and don't get split in awkward places.


@dataclass
class Configuration:
    """
    A sample configuration class with various attributes.
    
    This class includes various data types and nested structures
    to test how the chunker handles complex class definitions.
    """
    name: str
    value: int = 42
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the configuration settings."""
        if not self.name:
            return False
        
        if self.value < 0:
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "enabled": self.enabled,
            "options": self.options
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Configuration':
        """Create a configuration from a dictionary."""
        return cls(
            name=data.get("name", ""),
            value=data.get("value", 42),
            enabled=data.get("enabled", True),
            options=data.get("options", {})
        )
    
    class Validator:
        """Nested validator class to test nested class chunking."""
        
        def __init__(self, config: 'Configuration'):
            self.config = config
            
        def check_rules(self) -> List[str]:
            """Check if configuration meets all rules."""
            errors = []
            
            if not self.config.name:
                errors.append("Name cannot be empty")
                
            if self.config.value < 0:
                errors.append("Value must be non-negative")
                
            for key, value in self.config.options.items():
                if not isinstance(key, str):
                    errors.append(f"Option key must be string, got {type(key)}")
                    
                if isinstance(value, dict):
                    # Recursively check nested dictionaries
                    for k, v in value.items():
                        if not isinstance(k, str):
                            errors.append(f"Nested option key must be string, got {type(k)}")
            
            return errors


def process_configuration(config: Configuration) -> Tuple[bool, List[str]]:
    """
    Process a configuration object.
    
    This standalone function tests how functions are chunked
    and how they reference other symbols.
    
    Args:
        config: Configuration object to process
        
    Returns:
        Tuple of (success, error_messages)
    """
    validator = Configuration.Validator(config)
    errors = validator.check_rules()
    
    if errors:
        return False, errors
        
    # Process the configuration
    result = {
        "processed": True,
        "name": config.name,
        "value": config.value * 2,  # Double the value
        "options_count": len(config.options)
    }
    
    # Log the result
    logging.info(f"Processed configuration: {json.dumps(result)}")
    
    return True, []


def very_long_function_with_many_lines_to_test_chunking_limits():
    """
    A very long function to test how the chunker handles functions that exceed token limits.
    
    This function is intentionally verbose and includes repeated code patterns
    to ensure it exceeds typical chunk size limits.
    """
    # Initialize a large data structure
    data = []
    for i in range(100):
        item = {
            "id": i,
            "name": f"Item {i}",
            "values": [j * j for j in range(20)],
            "metadata": {
                "created": "2025-04-30",
                "priority": i % 5,
                "tags": [f"tag{t}" for t in range(5)],
                "nested": {
                    "level1": {
                        "level2": {
                            "level3": [1, 2, 3, 4, 5]
                        }
                    }
                }
            }
        }
        data.append(item)
    
    # Perform various operations on the data
    result = []
    for item in data:
        # Extract values
        item_id = item["id"]
        item_name = item["name"]
        item_values = item["values"]
        
        # Compute statistics
        avg_value = sum(item_values) / len(item_values) if item_values else 0
        max_value = max(item_values) if item_values else 0
        min_value = min(item_values) if item_values else 0
        
        # Create summary
        summary = {
            "id": item_id,
            "name": item_name,
            "stats": {
                "avg": avg_value,
                "max": max_value,
                "min": min_value,
                "range": max_value - min_value,
                "count": len(item_values)
            }
        }
        
        # Add metadata
        for key, value in item["metadata"].items():
            if key == "nested":
                # Flatten nested structure
                flat = {}
                for k1, v1 in value.items():
                    for k2, v2 in v1.items():
                        for k3, v3 in v2.items():
                            flat[f"{k1}.{k2}.{k3}"] = v3
                summary["flattened"] = flat
            else:
                summary[key] = value
        
        # Apply transformations based on priority
        priority = item["metadata"]["priority"]
        if priority == 0:
            summary["status"] = "critical"
            summary["action"] = "immediate"
        elif priority == 1:
            summary["status"] = "high"
            summary["action"] = "urgent"
        elif priority == 2:
            summary["status"] = "medium"
            summary["action"] = "normal"
        elif priority == 3:
            summary["status"] = "low"
            summary["action"] = "delayed"
        else:
            summary["status"] = "negligible"
            summary["action"] = "optional"
            
        result.append(summary)
    
    # Aggregate results
    aggregated = {
        "total_items": len(result),
        "by_status": {},
        "by_action": {}
    }
    
    for item in result:
        status = item["status"]
        action = item["action"]
        
        # Count by status
        if status not in aggregated["by_status"]:
            aggregated["by_status"][status] = 0
        aggregated["by_status"][status] += 1
        
        # Count by action
        if action not in aggregated["by_action"]:
            aggregated["by_action"][action] = 0
        aggregated["by_action"][action] += 1
    
    return aggregated


if __name__ == "__main__":
    # Sample usage
    config = Configuration(
        name="Test Config",
        value=100,
        options={
            "debug": True,
            "timeout": 30,
            "nested": {
                "setting1": "value1",
                "setting2": "value2"
            }
        }
    )
    
    success, errors = process_configuration(config)
    print(f"Processing {'succeeded' if success else 'failed'}")
    if errors:
        print(f"Errors: {errors}")
        
    result = very_long_function_with_many_lines_to_test_chunking_limits()
    print(f"Processed {result['total_items']} items")
