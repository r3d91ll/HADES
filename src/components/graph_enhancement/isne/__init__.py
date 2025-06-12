"""
ISNE Components

This module provides all ISNE (Inductive Shallow Node Embedding) components:
- CoreISNE: General ISNE graph enhancement using existing HADES pipeline
- ISNETrainingEnhancer: Training-focused ISNE component for model training
- ISNEInferenceEnhancer: Inference-focused ISNE component for fast inference
"""

from .core import CoreISNE
from .training import ISNETrainingEnhancer
from .inference import ISNEInferenceEnhancer

__all__ = [
    "CoreISNE",
    "ISNETrainingEnhancer", 
    "ISNEInferenceEnhancer"
]