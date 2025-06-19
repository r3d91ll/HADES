"""
Production Pipeline Configuration Loader

Loads and validates configuration for production pipeline stages.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProductionConfigLoader:
    """Loads configuration for production pipeline stages."""
    
    def __init__(self, config_base_path: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_base_path: Base path for config files (defaults to src/config)
        """
        if config_base_path is None:
            # Default to src/config from project root
            self.config_base_path = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_base_path = Path(config_base_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Relative path to config file from config base
            
        Returns:
            Loaded configuration dictionary
        """
        full_path = self.config_base_path / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")
            
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded config from {full_path}")
        return config
    
    def load_production_config(self) -> Dict[str, Any]:
        """Load main production pipeline configuration."""
        return self.load_config("pipelines/production.yaml")
    
    def load_bootstrap_config(self) -> Dict[str, Any]:
        """Load bootstrap stage configuration.""" 
        return self.load_config("pipelines/production/bootstrap.yaml")
    
    def load_training_config(self) -> Dict[str, Any]:
        """Load training stage configuration."""
        return self.load_config("pipelines/production/training.yaml")
    
    def load_application_config(self) -> Dict[str, Any]:
        """Load application stage configuration."""
        return self.load_config("pipelines/production/application.yaml")
    
    def load_collections_config(self) -> Dict[str, Any]:
        """Load collections stage configuration."""
        return self.load_config("pipelines/production/collections.yaml")
    
    def merge_config_with_overrides(self, base_config: Dict[str, Any], 
                                   overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with runtime overrides.
        
        Args:
            base_config: Base configuration from file
            overrides: Runtime override values
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        def _merge_dict(base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    _merge_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        _merge_dict(merged, overrides)
        return merged
    
    def validate_config(self, config: Dict[str, Any], stage: str) -> bool:
        """
        Validate configuration for a specific stage.
        
        Args:
            config: Configuration to validate
            stage: Stage name (bootstrap, training, application, collections)
            
        Returns:
            True if valid, raises exception if not
        """
        required_sections = {
            "bootstrap": ["stage", "input", "database", "processing"],
            "training": ["stage", "database", "model", "training"],
            "application": ["stage", "model", "database", "processing"],
            "collections": ["stage", "database", "collections"]
        }
        
        if stage not in required_sections:
            raise ValueError(f"Unknown stage: {stage}")
            
        missing_sections = []
        for section in required_sections[stage]:
            if section not in config:
                missing_sections.append(section)
                
        if missing_sections:
            raise ValueError(f"Missing required config sections for {stage}: {missing_sections}")
            
        logger.info(f"Configuration validation passed for stage: {stage}")
        return True

# Global config loader instance
config_loader = ProductionConfigLoader()