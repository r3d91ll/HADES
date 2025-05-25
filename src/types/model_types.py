"""
Model type definitions for HADES-PathRAG.

This module provides type definitions for model configuration and interaction.
"""

from typing import Dict, List, Any, Optional, Union, Literal

# Model mode enum for clarity
ModelMode = Literal["inference", "ingestion"]

# Server configuration
ServerConfigType = Dict[str, Any]

# Model backend configuration
ModelBackendConfigType = Dict[str, Any]

# Overall model configuration 
ModelConfigType = Dict[str, Any]
