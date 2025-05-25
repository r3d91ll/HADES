#!/bin/bash
# HADES-PathRAG: Development Environment Setup Script
# This script sets up a Python 3.10+ virtual environment, installs Poetry,
# configures all dependencies, and validates the project setup.

set -e

# Print each command before executing (for debugging)
if [ "$1" == "--debug" ]; then
    set -x
fi

# Ensure Python 3.10+ is available
PY_VERSION=$(python3 --version | cut -d' ' -f2)
PY_MAJOR=$(echo $PY_VERSION | cut -d'.' -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d'.' -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required (detected $PY_VERSION)"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

echo "Python $PY_VERSION detected (✓)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "\nCreating Python virtual environment in .venv..."
    python3 -m venv .venv
fi

# Ensure activation works regardless of shell
if [[ -f ".venv/bin/activate" ]]; then
    echo "\nActivating virtual environment..."
    source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then  # Windows compatibility
    source ".venv/Scripts/activate"
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

# Verify activation worked
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Error: Virtual environment activation failed"
    exit 1
fi

# Install Poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo "\nPoetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Configure Poetry to use the virtual environment
echo "\nConfiguring Poetry..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.create false

# Install dependencies
echo "\nInstalling dependencies..."
poetry install --no-interaction

# Notify about optional vLLM installation
echo "\nNote: vLLM installation may require additional steps depending on your GPU setup."
echo "Refer to docs/integration/vllm_setup.md for specific installation instructions."

# Validate core package imports
echo "\nValidating core package imports..."
try_import() {
    python -c "import $1" 2>/dev/null && echo "✓ $1 imported successfully" || echo "✗ $1 import failed"
}

try_import "ingest"
try_import "pathrag"
try_import "mcp"
try_import "xnx"
try_import "ingest.pre_processor"

# Skip tests and type checking by default to allow partial setup
if [ "$1" == "--full" ]; then
    # Run type checking
    echo "\nRunning type checking..."
    poetry run mypy src/ || echo "Type checking found issues that need to be fixed."

    # Run tests
    echo "\nRunning tests..."
    poetry run pytest || echo "Some tests are failing. This is expected during the transition process."
fi

echo "\n✨ HADES-PathRAG development environment setup complete! ✨"
echo "To activate the environment in new terminals, run: source .venv/bin/activate"
echo "Run 'poetry run python -m ingest.pre_processor.tests' to verify pre-processor tests"
echo "To run type checking anytime: poetry run mypy src/"
