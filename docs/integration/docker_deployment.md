# Docker Deployment for HADES-PathRAG

This guide explains how to deploy HADES-PathRAG using Docker, providing a containerized solution that includes Ollama for local LLM inference and ArangoDB for knowledge graph storage.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 16GB RAM recommended for running the full stack
- At least 20GB free disk space for Docker images and models

## Quick Start

The easiest way to get started is with Docker Compose, which will set up the complete system including Ollama and ArangoDB:

```bash
# Clone the repository
git clone https://github.com/r3d91ll/HADES-PathRAG.git
cd HADES-PathRAG

# Optional: Set ArangoDB password in environment
export ARANGO_PASSWORD="your-secure-password"

# Start the services
docker-compose up -d

# Check the logs
docker-compose logs -f
```

This will:
1. Start an ArangoDB instance for graph storage
2. Build and start HADES-PathRAG with Ollama integrated
3. Download the required LLM models automatically
4. Expose all necessary ports for the services

## Accessing the Services

Once the containers are running, you can access:

- **MCP Server API**: http://localhost:8123
- **ArangoDB Web Interface**: http://localhost:8529
- **Ollama API**: http://localhost:11434

## Using the Docker Image Separately

If you prefer to run only the HADES-PathRAG container without Docker Compose:

```bash
# Build the Docker image
docker build -t hades-pathrag .

# Run the container
docker run -p 8123:8123 -p 8765:8765 -p 11434:11434 \
  -e ARANGO_HOST=your-arango-host \
  -e ARANGO_PASSWORD=your-arango-password \
  -v $(pwd)/PathRAG_cache:/app/PathRAG_cache \
  hades-pathrag
```

## Container Architecture

The containerized setup includes:

```
┌─────────────────────────────────────────┐
│                Docker Host               │
│                                         │
│  ┌───────────────┐    ┌───────────────┐ │
│  │  HADES-PathRAG │    │   ArangoDB    │ │
│  │  Container    │    │   Container   │ │
│  │               │    │               │ │
│  │ ┌───────────┐ │    │ ┌───────────┐ │ │
│  │ │ MCP Server │ │    │ │  Database │ │ │
│  │ └───────────┘ │    │ └───────────┘ │ │
│  │               │    │               │ │
│  │ ┌───────────┐ │    │               │ │
│  │ │   Ollama   │ │    │               │ │
│  │ └───────────┘ │    │               │ │
│  │               │    │               │ │
│  └───────────────┘    └───────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

## Environment Variables

You can customize the deployment by setting environment variables:

```yaml
# Docker Compose environment variables
ARANGO_PASSWORD: Password for ArangoDB root user

# HADES-PathRAG container environment variables
ARANGO_HOST: Hostname for ArangoDB (default: arango)
ARANGO_PORT: Port for ArangoDB (default: 8529)
ARANGO_DB: Database name in ArangoDB (default: hades_knowledge)
ARANGO_USER: Username for ArangoDB (default: root)
LLM_PROVIDER: LLM provider to use (default: ollama)
OLLAMA_MODEL: Ollama model to use (default: llama3)
OLLAMA_EMBED_MODEL: Embedding model (default: nomic-embed-text)
```

## Persistent Storage

The Docker Compose setup includes two volume mounts for persistent storage:

1. `arango_data`: Stores the ArangoDB database files
2. `ollama_models`: Stores the downloaded Ollama models
3. `./PathRAG_cache`: Local directory mounted into the container for caching PathRAG data

## Managing Ollama Models

You can manage Ollama models from inside the container:

```bash
# List available models
docker exec -it hades-pathrag_hades-pathrag_1 ollama list

# Pull additional models
docker exec -it hades-pathrag_hades-pathrag_1 ollama pull mistral
```

## Performance Considerations

- **Memory Usage**: Running the full stack requires at least 16GB RAM
- **CPU Usage**: More CPU cores will improve performance of both Ollama and ArangoDB
- **GPU Acceleration**: If you have a GPU, you can modify the Docker setup to use it for Ollama
- **Storage**: Downloaded models will consume significant disk space

## Troubleshooting

### Common Issues

1. **Container won't start**: Check if you have enough memory and disk space
   ```bash
   docker-compose logs hades-pathrag
   ```

2. **Cannot connect to ArangoDB**: Ensure the database is running and the credentials are correct
   ```bash
   docker-compose ps
   docker-compose logs arango
   ```

3. **Models fail to download**: Check Ollama logs for network issues
   ```bash
   docker exec -it hades-pathrag_hades-pathrag_1 curl http://localhost:11434/api/tags
   ```

4. **Out of memory errors**: Reduce the model size or increase container memory limits
   ```bash
   # Edit docker-compose.yml to add memory limits
   services:
     hades-pathrag:
       deploy:
         resources:
           limits:
             memory: 16G
   ```

## Advanced Configuration

For advanced users who want to customize the Docker deployment:

1. **Custom Ollama models**: Create a Modelfile and build custom models
2. **ArangoDB clustering**: Set up a clustered ArangoDB deployment for high availability
3. **Docker networking**: Configure custom Docker networks for security and isolation
4. **Load balancing**: Set up load balancing for multiple HADES-PathRAG instances

For more details on these advanced topics, refer to the Docker and ArangoDB documentation.
