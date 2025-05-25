# Setting Up Ollama with HADES-PathRAG

This guide explains how to set up and use Ollama as the LLM provider for HADES-PathRAG, enabling local inference with various open-source models. HADES-PathRAG is designed to work optimally with a locally installed Ollama service running on your system.

## What is Ollama?

[Ollama](https://ollama.com/) is an open-source framework that lets you run large language models (LLMs) locally on your machine. It provides optimized runtimes for various open-source models and a simple API for integration.

## Prerequisites

- A Linux, macOS, or Windows system with sufficient resources to run the desired model
- At least 8GB RAM (16GB+ recommended for larger models)
- Sufficient disk space for model storage (5-30GB depending on models)

## Installation

### 1. Install Ollama as a System Service (Recommended)

**Linux (systemd-based distributions):**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Ollama will be installed as a systemd service and started automatically
```

To check if Ollama is running as a system service:
```bash
systemctl status ollama
```

If needed, you can manually manage the service:
```bash
# Start the service
systemctl start ollama

# Enable at boot
systemctl enable ollama
```

**macOS:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run Ollama from Applications or command line
open -a Ollama
```

**Windows:**
Download and install from https://ollama.com/download

### 2. Verify Ollama is Running

The Ollama service should be running on http://localhost:11434 by default. Verify with:

```bash
# Check server status
curl http://localhost:11434/api/tags
```

### 3. Pull Required Models

```bash
# Pull the LLM model for text generation
ollama pull llama3    # Or another model of your choice

# Pull the embedding model
ollama pull nomic-embed-text
```

## Configuration with HADES-PathRAG

HADES-PathRAG comes with Ollama integration out of the box. The system will use Ollama as the default model engine if configured properly.

### 1. Configuration Files

Create a `.env` file in the project root (you can copy from `.env.template`):

```bash
cp .env.template .env
```

Edit the `.env` file to set Ollama as your provider and configure model settings:

```
# LLM Provider
LLM_PROVIDER=ollama

# Ollama Settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=60
```

### 2. Verify Ollama Integration

You can test the Ollama integration using the provided example script:

```bash
python examples/ollama_example.py
```

This will test:
- Basic text generation with Ollama
- Embedding generation
- Integration with PathRAG

## Customizing the Ollama Integration

### Using Different Models

To change the models used for inference:

1. Pull the desired model:
   ```bash
   ollama pull mistral    # Or any other supported model
   ```

2. Update your `.env` file:
   ```
   OLLAMA_MODEL=mistral
   ```

### Model Parameters

You can customize model parameters in the `.env` file:

```
OLLAMA_TEMPERATURE=0.7   # Control randomness (0.0-1.0)
OLLAMA_TOP_P=0.9         # Nucleus sampling
OLLAMA_TOP_K=40          # Limit vocabulary options
OLLAMA_MAX_TOKENS=2048   # Maximum response length
```

### Advanced: Custom Model Configuration

For more advanced model configuration, you can create custom Ollama models using Modelfiles. See the [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for details.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure the Ollama server is running (`ollama serve`).

2. **Model Not Found**: If you see errors about models not found, pull the models first:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. **Out of Memory**: If you encounter out-of-memory errors, try a smaller model or increase your system's swap space.

4. **Slow Response Times**: First inference can be slow as the model loads into memory. Subsequent calls will be faster.

### Checking Ollama Status

```bash
# List available models
ollama list

# Check server status
curl http://localhost:11434/api/tags
```

## Performance Considerations

- **Memory Usage**: Larger models require more RAM. Models like Llama 3 8B need at least 8GB RAM.
- **GPU Acceleration**: Ollama will automatically use GPU if available, significantly improving performance.
- **First Inference**: The first call to a model will be slower as it loads into memory.

## Working with Locally Installed Ollama vs. Docker

While Docker deployment is available as an option (see [Docker Deployment](./docker_deployment.md)), using a locally installed Ollama service is the **recommended approach** for most users because:

1. **Performance**: Direct access to system resources without Docker overhead
2. **Simplicity**: No need to manage Docker containers or volumes
3. **Integration**: Seamless integration with other local development tools
4. **Persistence**: System service ensures Ollama is always available

For development and testing, the locally installed Ollama service provides the best experience and is the primary recommended approach for HADES-PathRAG integration.

## Additional Resources

- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Available Models](https://ollama.com/library)
- [Ollama systemd service documentation](https://github.com/ollama/ollama/blob/main/docs/linux.md)
