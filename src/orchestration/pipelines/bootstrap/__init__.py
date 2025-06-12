"""
Bootstrap Pipeline Package

This package contains bootstrap pipelines for initializing HADES components
from scratch, particularly for graph enhancement models like ISNE.
"""

from .modular_pipeline import ModularBootstrapPipeline, run_modular_bootstrap
from .adaptive_training import AdaptiveISNETrainer, ScheduledISNETrainer

__all__ = [
    'ModularBootstrapPipeline',
    'run_modular_bootstrap',
    'AdaptiveISNETrainer',
    'ScheduledISNETrainer'
]