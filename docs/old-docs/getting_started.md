# Getting Started with HADES-PathRAG

This guide will help you quickly get started with HADES-PathRAG using Ollama as the local language model provider. Follow these steps to set up and run your first knowledge graph with LLM-powered path retrieval.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Using with Ollama](#using-with-ollama)
4. [Running the MCP Server](#running-the-mcp-server)
5. [Basic Usage Examples](#basic-usage-examples)
6. [Docker Deployment](#docker-deployment)
7. [Testing Your Setup](#testing-your-setup)
8. [Next Steps](#next-steps)

## Installation

### Prerequisites

- Python 3.8+ installed
- Git installed
- For local LLM inference: Ollama installed (see [Ollama Setup](./integration/ollama_setup.md))

### Install HADES-PathRAG

```bash
# Clone the repository
git clone https://github.com/r3d91ll/HADES-PathRAG.git
cd HADES-PathRAG

# Install dependencies
pip install -e .
```

## Configuration

HADES-PathRAG uses a configuration system based on environment variables. The simplest way to get started is to copy the template:

```bash
cp .env.template .env
```

Then edit `.env` to configure your settings. For Ollama integration, the key settings are:

```
# LLM Provider
LLM_PROVIDER=ollama

# Ollama Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Using with Ollama

### 1. Use Your Local Ollama Installation

If you already have Ollama installed and running as a system service (recommended approach), you can skip directly to downloading the required models.

To check if Ollama is running as a system service:

```bash
systemctl status ollama
```

If you need to install Ollama:

```bash
# Install Ollama (installs as a systemd service on Linux)
curl -fsSL https://ollama.com/install.sh | sh
```

For complete installation instructions, see [Ollama Setup](./integration/ollama_setup.md).

### 2. Verify Ollama is Running

The Ollama service should be running at http://localhost:11434 by default. You can verify with:

```bash
# Check status using curl
curl http://localhost:11434/api/tags
```

### 3. Download Required Models

```bash
# For text generation
ollama pull llama3

# For embeddings
ollama pull nomic-embed-text
```

## Running the MCP Server

The Model Context Protocol (MCP) Server provides an API for interacting with HADES-PathRAG:

```bash
# Start the MCP server
python -m src.mcp.server
```

The server will be available at:
- WebSocket: `ws://localhost:8765`
- HTTP API: `http://localhost:8123`

## Basic Usage Examples

### Direct Python Usage

```python
from pathrag import PathRAG
from pathrag.llm import ollama_model_complete

# Initialize PathRAG with Ollama
pathrag = PathRAG(
    working_dir="./PathRAG_cache",
    llm_model_func=lambda prompt, **kwargs: ollama_model_complete(
        prompt=prompt,
        hashing_kv={"global_config": {"llm_model_name": "llama3"}},
        host="http://localhost:11434"
    )
)

# Add knowledge
knowledge = [
    "HADES is a recursive AI system that can improve itself.",
    "PathRAG uses path-based retrieval in knowledge graphs.",
    "XnX notation provides weighted relationships in knowledge graphs."
]
pathrag.insert(knowledge)

# Query the knowledge
result = await pathrag.query("How does HADES improve itself?")
print(result)
```

### Using the Example Script

We've included a complete example in `examples/ollama_example.py`:

```bash
python examples/ollama_example.py
```

This demonstrates:
- Text generation with Ollama
- Embedding generation
- PathRAG integration

## Alternative: Docker Deployment

While using a locally installed Ollama service is recommended, we also provide Docker deployment as an alternative option for specific use cases:

```bash
# Start everything with Docker Compose (if you prefer containerization)
docker-compose up -d
```

This container-based approach is useful for:
- Testing in isolated environments
- Deployment in cloud environments
- Users who prefer containerized solutions

For most development and testing scenarios, the locally installed Ollama service provides better performance and simpler setup.

For details on Docker deployment, see [Docker Deployment](./integration/docker_deployment.md).

## Testing Your Setup

Verify your setup using the included test script:

```bash
# Run Ollama integration tests
python tests/test_ollama_integration.py
```

This tests:
- Ollama connectivity
- Text generation
- Embedding generation
- PathRAG integration

## Next Steps

Once you have the basic setup working, explore these advanced features:

### XnX-Enhanced PathRAG

Use the XnX notation system for enhanced relationship representation:

```python
from src.xnx.xnx_pathrag import XnXPathRAG
from src.xnx.xnx_params import XnXParams

# Initialize XnX-PathRAG
xnx_pathrag = XnXPathRAG(
    db_url="http://localhost:8529",
    db_name="hades_knowledge",
    username="root",
    password=""
)

# Configure XnX parameters
xnx_params = XnXParams(
    llm_provider="ollama",
    llm_model="llama3",
    embed_model="nomic-embed-text"
)
xnx_pathrag.set_parameters(xnx_params)
```

### Using Different Models

Ollama supports many different models. To use a different model:

1. Pull the model:
   ```bash
   ollama pull mistral
   ```

2. Update your `.env` file:
   ```
   OLLAMA_MODEL=mistral
   ```

3. Or specify directly in code:
   ```python
   llm_model_func=lambda prompt, **kwargs: ollama_model_complete(
       prompt=prompt,
       hashing_kv={"global_config": {"llm_model_name": "mistral"}},
       host="http://localhost:11434"
   )
   ```

### Integration with Applications

See the [API Reference](./api/mcp_server.md) for details on how to integrate HADES-PathRAG with your applications using the MCP Server API.

### Customizing PathRAG

Explore the various configuration options in `src/config/` to customize PathRAG behavior for your specific use cases.

## Need Help?

- Check the documentation in the [docs/](.) directory
- For Ollama-specific issues, see [Ollama Setup](./integration/ollama_setup.md)
- For Docker deployment issues, see [Docker Deployment](./integration/docker_deployment.md)
