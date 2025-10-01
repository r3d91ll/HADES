#!/usr/bin/env bash
#
# Install PyTorch Geometric Optional Accelerated Libraries
# =========================================================
#
# These packages provide optimized operations for GraphSAGE:
# - pyg-lib: Compiled sampling and aggregation ops
# - torch-scatter: Scatter/gather operations
# - torch-sparse: Sparse matrix operations
#
# They must be installed from PyG's custom wheel repository based on your
# CUDA version and PyTorch version.
#
# Usage:
#   bash scripts/install_pyg_extras.sh
#
# Note: If this fails, GraphSAGE will still work but may be slower.
#       The base torch-geometric package works without these.

set -e

echo "=========================================="
echo "Installing PyG Optional Extensions"
echo "=========================================="

# Detect PyTorch and CUDA versions
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "cpu")

if [ -z "$TORCH_VERSION" ]; then
    echo "ERROR: PyTorch not found. Please install PyTorch first."
    echo "  See: https://pytorch.org/get-started/locally/"
    exit 1
fi

echo "Detected PyTorch: $TORCH_VERSION"
echo "Detected CUDA: $CUDA_VERSION"

# Determine wheel URL based on versions
TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1,2 | tr -d .)
if [ "$CUDA_VERSION" == "cpu" ] || [ -z "$CUDA_VERSION" ]; then
    CUDA_SUFFIX="cpu"
else
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d .)
    CUDA_SUFFIX="cu${CUDA_MAJOR}"
fi

WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_MAJOR}+${CUDA_SUFFIX}.html"

echo "Using wheel repository: $WHEEL_URL"
echo ""

# Install extensions
echo "Installing pyg-lib..."
pip install pyg-lib -f $WHEEL_URL || echo "WARNING: pyg-lib install failed (optional)"

echo "Installing torch-scatter..."
pip install torch-scatter -f $WHEEL_URL || echo "WARNING: torch-scatter install failed (optional)"

echo "Installing torch-sparse..."
pip install torch-sparse -f $WHEEL_URL || echo "WARNING: torch-sparse install failed (optional)"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  python -c 'import torch_geometric; print(torch_geometric.__version__)'"
echo "  python -c 'import pyg_lib; print(\"pyg_lib:\", pyg_lib.__version__)'"
echo ""
echo "If optional packages failed, GraphSAGE will still work but may be slower."
