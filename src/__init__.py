"""HADES-PathRAG Integration Package

This package integrates PathRAG with HADES, implementing
XnX notation for knowledge graph relationship management.
"""

__version__ = "0.1.0"

# PathRAG (LLM-inference stack) is heavy and pulls in Torch/CUDA. For
# lightweight utilities (e.g., data-ingestion pipelines) we keep it optional.
# Set env `HADES_ENABLE_PATHRAG=1` to auto-import, or import explicitly.

import os

if os.getenv("HADES_ENABLE_PATHRAG", "0") == "1":
    try:
        from . import pathrag  # noqa: F401
    except Exception as _exc:  # pragma: no cover
        # Log but do not fail – allows docproc & friends to be used standalone.
        import warnings

        warnings.warn(
            "Failed to import PathRAG sub-module; set HADES_ENABLE_PATHRAG=0 or"
            " ensure its dependencies are available.\n"
            f"Original error: {_exc!r}"
        )
# XnX and MCP modules have been moved to dead-code