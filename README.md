# HADES-PathRAG

Enhanced implementation of **PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths** with ArangoDB integration and **vLLM support** for high-performance local LLM inference.

## Features

- 🔍 **PathRAG Implementation**: Efficient graph-based RAG that prunes retrieval paths
- 🧠 **vLLM Integration**: High-performance inference engine for local model serving
- 📊 **ArangoDB Support**: Scalable graph database backend for enterprise use
- 🔄 **ISNE Embedding**: Inductive Shallow Node Embedding for semantic understanding
- 📝 **Text Processing**: Comprehensive document handling with customizable chunking
- 🚀 **FastAPI Interface**: Simple, lightweight API for system interaction
- 🔧 **Type-Safe Implementation**: Fully type-annotated codebase for reliability

## Components

- **Document Processing**: Advanced document parsing and normalization pipeline
- **Chunking**: Smart document chunking respecting semantic boundaries
- **Embedding**: High-dimensional vector embeddings using ModernBERT
- **ISNE**: Graph structure-aware embedding enhancement
- **Storage**: ArangoDB integration with graph and vector capabilities
- **Retrieval**: Path-based semantic search with graph traversal
- **Generation**: Context-aware response generation with LLMs

## Install

```bash
# Clone the repository
git clone https://github.com/r3d91ll/HADES-PathRAG.git
cd HADES-PathRAG

# Install dependencies
pip install -e .

# Install PyTorch Geometric (required for ISNE)
pip install torch-geometric

# Set up vLLM for local inference (optional)
pip install vllm

# (Optional) Install ArangoDB if not already installed
# Follow instructions at https://www.arangodb.com/download/
```

## Quick Start

### Document Processing and ISNE Training

```bash
# Process and embed documents with ISNE enhancement
python -m src.pipelines.ingest.orchestrator.isne_training_text \
    --input_dir /path/to/pdf/directory \
    --output_dir ./test-output/isne-training \
    --model_dir ./models/isne
```

### Using the PathRAG System

```bash
# Query your knowledge graph
python -m src.query.pathrag_query \
    --question "Your question here" \
    --document_collection documents \
    --chunk_collection chunks
```

### API Server (Optional)

```bash
# Start the FastAPI server
python -m src.api.server
```

- Use the following Python snippet in the "v1_text.py" file to initialize PathRAG and perform queries.
  
```python
import os
from pathrag import PathRAG, QueryParam

# For using vLLM (no API keys needed)
from pathrag.llm import vllm_model_complete

# For using OpenAI (if preferred)
# from pathrag.llm import openai_complete
# os.environ["OPENAI_API_KEY"] = "your_api_key"

WORKING_DIR = "./your_working_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize with vLLM as the LLM provider
rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lambda prompt, **kwargs: vllm_model_complete(
        prompt=prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
)

data_file="./text.txt"
question="your_question"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
```

## Parameter modification

You can adjust the relevant parameters in the `base.py` and `operate.py` files.

## Batch Insert

```python
import os
folder_path = "your_folder_path"  

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

## Cite

Please cite our paper if you use this code in your own work:

```python
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```

## HADES: Enhanced PathRAG Implementation

This project contains an advanced implementation of PathRAG that integrates ArangoDB and vLLM for high-performance knowledge graph retrieval. For detailed documentation, see our [Ingestion System](./docs/ingestion_system.md), [Chunking Strategy](./docs/chunking.md), and [API Reference](./docs/api.md).

## Documentation Structure

The `/docs` directory contains comprehensive documentation organized as follows:

### Core Components

- [Ingestion System](./docs/ingestion_system.md) - Complete ingestion pipeline and incremental update process
- [Chunking Strategy](./docs/chunking.md) - Hybrid chunking with semantic and code-aware approaches
- [API Reference](./docs/api.md) - FastAPI interface for the PathRAG system

### Integration Guides

- [ArangoDB Setup](./docs/integration/arango_setup.md) - Setting up ArangoDB for HADES
- [Docker Deployment](./docs/integration/docker_deployment.md) - Containerized deployment
- [GitHub Ingestion](./docs/integration/github_ingestion.md) - Ingesting code from GitHub repositories

### Original Research

- [Academic Papers](./docs/original_paper/) - Research foundation for PathRAG

### Experimental Features

- [XnX Notation](./docs/xnx/) - Documentation for experimental relationship notation

### What is HADES?

HADES represents our approach to building a powerful, type-safe knowledge retrieval system:

- **H**euristic approach to knowledge representation and retrieval
- **A**daptive graph traversal with relationship-aware path ranking
- **D**ata-centric with efficient ArangoDB storage and incremental updates
- **E**xtrapolation capabilities for models using retrieved context
- **S**ystem designed as a network of interconnected processes

### Key Enhancements

- **Incremental Knowledge Graph**: Support for updating individual nodes and relationships
- **ArangoDB Integration**: Optimized graph storage with combined vector and graph operations
- **FastAPI Interface**: Simple REST API for interaction with the system
- **vLLM Integration**: High-performance inference for improved throughput
- **ISNE Embedding**: Advanced embedding approach that captures relationship information

Run `python -m src.api.cli` to launch the API server interface.

## Development Roadmap

### Completed

- ✅ Basic PathRAG implementation with ArangoDB support
- ✅ Type-safe base interfaces for embeddings, storage, and graph operations
- ✅ Python pre-processor with symbol table support
- ✅ Ingestion pipeline documentation and design
- ✅ FastAPI server implementation

### In Progress

- 🔄 Docling pre-processor improvements
- 🔄 Hybrid chunking implementation (Chonky + code-aware)
- 🔄 ISNE embedding integration
- 🔄 Incremental update support

### Planned

- 📅 Integration tests for full pipeline
- 📅 Performance optimization (parallelism)
- 📅 vLLM integration for inference
- 📅 Web interface development
- 📅 XnX notation support (experimental)
