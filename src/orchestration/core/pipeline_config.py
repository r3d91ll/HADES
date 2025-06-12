"""
Pipeline Configuration System with Component Selection

This module provides a configuration system that supports dynamic component
selection and swapping for the HADES pipeline architecture.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
import yaml
from dataclasses import dataclass, field

from src.config.config_loader import load_config

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a single pipeline component."""
    implementation: str  # e.g., "docling", "chonky", "vllm"
    config: Dict[str, Any] = field(default_factory=dict)
    fallback: Optional[str] = None
    enabled: bool = True


@dataclass
class PipelineComponentConfig:
    """Complete component configuration for a pipeline."""
    document_processor: ComponentConfig
    chunker: ComponentConfig
    embedder: ComponentConfig
    graph_enhancer: ComponentConfig
    storage: ComponentConfig


class PipelineConfigLoader:
    """
    Loads and manages pipeline configurations with component selection support.
    
    This loader supports:
    - Dynamic component selection via configuration
    - Component-specific configuration loading
    - Fallback component specification
    - Configuration validation
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the pipeline configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._component_configs: Dict[str, ComponentConfig] = {}
        
    def load_pipeline_config(self, 
                           config_name: Optional[str] = None,
                           config_dict: Optional[Dict[str, Any]] = None) -> PipelineComponentConfig:
        """
        Load pipeline configuration with component selections.
        
        Args:
            config_name: Name of configuration file (without .yaml)
            config_dict: Direct configuration dictionary (overrides file)
            
        Returns:
            PipelineComponentConfig with all component configurations
        """
        # Load base configuration
        if config_dict:
            config = config_dict
        elif config_name:
            config = load_config(config_name)
        elif self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Load default data ingestion config
            config = load_config('data_ingestion_config')
            
        # Extract component selections
        components = config.get('components', {})
        
        # Create component configurations
        return PipelineComponentConfig(
            document_processor=self._create_component_config(
                'document_processor',
                components.get('document_processor', {}),
                config.get('docproc', {})
            ),
            chunker=self._create_component_config(
                'chunker',
                components.get('chunker', {}),
                config.get('chunking', {})
            ),
            embedder=self._create_component_config(
                'embedder',
                components.get('embedder', {}),
                config.get('embedding', {})
            ),
            graph_enhancer=self._create_component_config(
                'graph_enhancer',
                components.get('graph_enhancer', {}),
                config.get('isne', {})
            ),
            storage=self._create_component_config(
                'storage',
                components.get('storage', {}),
                config.get('storage', {})
            )
        )
    
    def _create_component_config(self,
                               component_type: str,
                               component_selection: Dict[str, Any],
                               stage_config: Dict[str, Any]) -> ComponentConfig:
        """
        Create configuration for a specific component.
        
        Args:
            component_type: Type of component (e.g., 'document_processor')
            component_selection: Component selection from config
            stage_config: Stage-specific configuration
            
        Returns:
            ComponentConfig for the component
        """
        # Default implementations if not specified
        default_implementations = {
            'document_processor': 'docling',
            'chunker': 'cpu',
            'embedder': 'modernbert',
            'graph_enhancer': 'isne',
            'storage': 'arangodb'
        }
        
        # Get implementation choice
        implementation = component_selection.get(
            'implementation',
            default_implementations.get(component_type, 'default')
        )
        
        # Get component-specific config
        # First check for implementation-specific config
        impl_config = component_selection.get('config', {})
        
        # Merge with stage config
        merged_config = {**stage_config, **impl_config}
        
        # Create component config
        return ComponentConfig(
            implementation=implementation,
            config=merged_config,
            fallback=component_selection.get('fallback'),
            enabled=component_selection.get('enabled', True)
        )
    
    def get_component_config(self, component_type: str) -> Optional[ComponentConfig]:
        """Get configuration for a specific component type."""
        return self._component_configs.get(component_type)
    
    def validate_config(self, config: PipelineComponentConfig) -> List[str]:
        """
        Validate pipeline configuration.
        
        Args:
            config: Pipeline component configuration
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required components are enabled
        required_components = [
            ('document_processor', config.document_processor),
            ('chunker', config.chunker),
            ('storage', config.storage)
        ]
        
        for name, component in required_components:
            if not component.enabled:
                errors.append(f"Required component '{name}' is disabled")
        
        # Validate implementation choices
        valid_implementations = {
            'document_processor': ['docling', 'markdown', 'python', 'json', 'yaml'],
            'chunker': ['cpu', 'gpu', 'chonky', 'ast'],
            'embedder': ['cpu', 'vllm', 'modernbert', 'ollama'],
            'graph_enhancer': ['isne', 'none', 'sagegraph'],
            'storage': ['arangodb', 'memory', 'networkx']
        }
        
        components = [
            ('document_processor', config.document_processor),
            ('chunker', config.chunker),
            ('embedder', config.embedder),
            ('graph_enhancer', config.graph_enhancer),
            ('storage', config.storage)
        ]
        
        for comp_type, component in components:
            valid_impls = valid_implementations.get(comp_type, [])
            if component.implementation not in valid_impls:
                errors.append(
                    f"Invalid implementation '{component.implementation}' "
                    f"for {comp_type}. Valid options: {valid_impls}"
                )
        
        return errors
    
    def create_example_config(self) -> Dict[str, Any]:
        """Create an example configuration with component selection."""
        return {
            'pipeline': {
                'name': 'data_ingestion_modular',
                'version': '2.0.0',
                'description': 'Modular data ingestion with component selection'
            },
            'components': {
                'document_processor': {
                    'implementation': 'docling',
                    'fallback': 'markdown',
                    'enabled': True,
                    'config': {
                        'extract_tables': True,
                        'ocr_enabled': True
                    }
                },
                'chunker': {
                    'implementation': 'cpu',
                    'fallback': None,
                    'enabled': True,
                    'config': {
                        'strategy': 'paragraph',
                        'max_chunk_size': 1000,
                        'overlap': 100
                    }
                },
                'embedder': {
                    'implementation': 'modernbert',
                    'fallback': 'cpu',
                    'enabled': True,
                    'config': {
                        'model_name': 'answerdotai/ModernBERT-base',
                        'batch_size': 32
                    }
                },
                'graph_enhancer': {
                    'implementation': 'isne',
                    'fallback': 'none',
                    'enabled': True,
                    'config': {
                        'model_path': './models/isne_model.pt',
                        'output_dim': 64
                    }
                },
                'storage': {
                    'implementation': 'arangodb',
                    'fallback': 'memory',
                    'enabled': True,
                    'config': {
                        'database_name': 'hades_pathrag',
                        'mode': 'create'
                    }
                }
            },
            # Include existing stage configs for backward compatibility
            'docproc': {},
            'chunking': {},
            'embedding': {},
            'isne': {},
            'storage': {}
        }


def load_pipeline_config(config_source: Optional[Any] = None) -> PipelineComponentConfig:
    """
    Convenience function to load pipeline configuration.
    
    Args:
        config_source: Can be:
            - None: Load default config
            - str: Config name to load
            - Path: Path to config file
            - dict: Direct configuration dictionary
            
    Returns:
        Loaded pipeline component configuration
    """
    loader = PipelineConfigLoader()
    
    if config_source is None:
        return loader.load_pipeline_config()
    elif isinstance(config_source, str):
        return loader.load_pipeline_config(config_name=config_source)
    elif isinstance(config_source, Path):
        loader.config_path = config_source
        return loader.load_pipeline_config()
    elif isinstance(config_source, dict):
        return loader.load_pipeline_config(config_dict=config_source)
    else:
        raise ValueError(f"Invalid config source type: {type(config_source)}")