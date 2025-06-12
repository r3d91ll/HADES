# API Module

The API module provides comprehensive REST API and command-line interfaces for interacting with the HADES system. It enables programmatic access to all system functionality including document processing, querying, training management, and system administration.

## 📋 Overview

This module implements:

- **REST API server** for HTTP-based system interaction
- **Command-line interface (CLI)** for terminal-based operations
- **Request/response models** with Pydantic validation
- **Core API logic** shared across interface types
- **Authentication and authorization** for secure access

## 🏗️ Architecture

### Interface Types

```text
HTTP Requests → REST API Server → Core API Logic → System Components
CLI Commands → CLI Interface → Core API Logic → System Components
```

### Component Organization

- **`server.py`** - FastAPI-based REST API server
- **`cli.py`** - Click-based command-line interface  
- **`core.py`** - Shared API logic and business rules
- **`models.py`** - Pydantic request/response schemas

## 📁 Module Contents

### Core Files

- **`server.py`** - FastAPI REST API implementation with endpoints
- **`cli.py`** - Click CLI with commands for all system operations
- **`core.py`** - Business logic shared between server and CLI
- **`models.py`** - API data models and validation schemas
- **`__init__.py`** - Module exports and convenience imports

## 🚀 Key Features

### REST API Server

**Start the server**:

```bash
# Development server
python -m src.api.server

# Production server with uvicorn
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

**API Documentation**:

- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

### Command-Line Interface

**Available commands**:

```bash
# Document processing
python -m src.api.cli ingest --input-dir ./docs --output-dir ./processed

# Query system
python -m src.api.cli query "How does ISNE training work?"

# Training management
python -m src.api.cli train --mode bootstrap --corpus-dir ./corpus

# System administration
python -m src.api.cli admin status
python -m src.api.cli admin health-check
```

## 🔌 API Endpoints

### Document Management

**Process Documents**:

```http
POST /api/v1/documents/process
Content-Type: application/json

{
  "input_paths": ["./docs/paper1.pdf", "./docs/code.py"],
  "output_dir": "./processed",
  "processing_config": {
    "chunking_strategy": "adaptive",
    "embedding_model": "modernbert"
  }
}
```

**Get Document Status**:

```http
GET /api/v1/documents/{document_id}/status
```

### Query Interface

**Semantic Query**:

```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "How does the chunking system handle Python code?",
  "mode": "hybrid",
  "max_results": 10,
  "include_sources": true
}
```

**Response**:

```json
{
  "answer": "The chunking system uses AST-based analysis...",
  "sources": [
    {
      "chunk_id": "chunk_123",
      "text": "AST chunker implementation...",
      "score": 0.92,
      "source_document": "chunking/ast_chunker.py"
    }
  ],
  "query_metadata": {
    "processing_time": 0.245,
    "total_chunks_searched": 15420
  }
}
```

### Training Management

**Start Training**:

```http
POST /api/v1/training/start
Content-Type: application/json

{
  "training_type": "incremental",
  "config_override": {
    "learning_rate": 0.001,
    "epochs": 50
  }
}
```

**Get Training Status**:

```http
GET /api/v1/training/status
```

**Training History**:

```http
GET /api/v1/training/history?days=7&training_type=weekly
```

### System Administration

**Health Check**:

```http
GET /api/v1/admin/health
```

**Response**:

```json
{
  "status": "healthy",
  "components": {
    "database": {"status": "connected", "response_time": 0.023},
    "embedding_service": {"status": "available", "model_loaded": true},
    "storage": {"status": "available", "free_space_gb": 45.2}
  },
  "last_training": {
    "type": "daily_incremental", 
    "completed": "2024-01-15T02:15:30Z",
    "status": "success"
  }
}
```

**System Metrics**:

```http
GET /api/v1/admin/metrics?time_range=24h
```

## 🔧 Configuration

### API Server Configuration

Set environment variables or use configuration files:

```bash
# Server settings
export HADES_API_HOST="0.0.0.0"
export HADES_API_PORT="8000"
export HADES_API_WORKERS="4"

# Security settings
export HADES_API_KEY="your-api-key"
export HADES_CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"

# System integration
export HADES_DATABASE_URL="http://localhost:8529"
export HADES_MODEL_PATH="./models/current_isne_model.pt"
```

### CLI Configuration

Configure CLI defaults:

```yaml
# ~/.hades/cli_config.yaml
api:
  default_host: "localhost:8000"
  timeout: 30
  
output:
  format: "json"  # json, table, or plain
  verbose: false
  
paths:
  default_corpus_dir: "./corpus"
  default_output_dir: "./output"
```

## 📊 Request/Response Models

### Core Data Models

**Document Processing Request**:

```python
from src.api.models import DocumentProcessingRequest

request = DocumentProcessingRequest(
    input_paths=["./docs/paper.pdf"],
    output_dir="./processed",
    processing_config={
        "chunking_strategy": "adaptive",
        "preserve_code_structure": True,
        "embedding_model": "modernbert"
    },
    metadata={
        "project_name": "research_analysis",
        "processing_priority": "high"
    }
)
```

**Query Request**:

```python
from src.api.models import QueryRequest

query = QueryRequest(
    question="Explain the ISNE bootstrap process",
    mode="hybrid",  # naive, local, global, hybrid
    max_results=10,
    filters={
        "source_type": ["pdf", "python"],
        "date_range": {"start": "2024-01-01", "end": "2024-01-31"}
    },
    include_sources=True,
    include_metadata=True
)
```

**Training Request**:

```python
from src.api.models import TrainingRequest

training = TrainingRequest(
    training_type="weekly_full_retrain",
    scope="weekly",
    config_override={
        "learning_rate": 0.0005,
        "batch_size": 32,
        "epochs": 75
    },
    force=False,
    dry_run=False
)
```

### Response Models

**Standardized API Response**:

```python
{
    "success": true,
    "data": {...},          # Actual response data
    "message": "Operation completed successfully",
    "metadata": {
        "request_id": "req_123456",
        "processing_time": 0.245,
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "errors": []            # Any warnings or errors
}
```

## 🔐 Authentication and Security

### API Key Authentication

**Server-side setup**:

```python
# Set API key in environment
export HADES_API_KEY="your-secure-api-key"
```

**Client requests**:

```http
GET /api/v1/query
Authorization: Bearer your-secure-api-key
```

```bash
# CLI with API key
python -m src.api.cli --api-key your-secure-api-key query "question"
```

### CORS Configuration

Configure allowed origins for web clients:

```python
# In server configuration
CORS_ORIGINS = [
    "http://localhost:3000",      # Development frontend
    "https://yourapp.com",        # Production frontend
    "https://api.yourapp.com"     # API gateway
]
```

### Rate Limiting

Built-in rate limiting for API protection:

```python
# Default limits
rate_limits = {
    "query": "100/minute",        # Query requests
    "processing": "10/minute",    # Document processing
    "training": "5/hour",         # Training operations
    "admin": "50/minute"          # Admin operations
}
```

## 📱 Usage Examples

### Python Client

```python
import requests
from src.api.models import QueryRequest

# Initialize client
api_base = "http://localhost:8000/api/v1"
headers = {"Authorization": "Bearer your-api-key"}

# Submit query
query_data = QueryRequest(
    question="How does chunking work in HADES?",
    mode="hybrid",
    max_results=5
)

response = requests.post(
    f"{api_base}/query",
    json=query_data.dict(),
    headers=headers
)

result = response.json()
print(f"Answer: {result['data']['answer']}")
for source in result['data']['sources']:
    print(f"- {source['source_document']}: {source['score']:.3f}")
```

### JavaScript Client

```javascript
const hadesClient = {
    baseURL: 'http://localhost:8000/api/v1',
    apiKey: 'your-api-key',
    
    async query(question, options = {}) {
        const response = await fetch(`${this.baseURL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                question,
                mode: options.mode || 'hybrid',
                max_results: options.maxResults || 10,
                include_sources: true
            })
        });
        return response.json();
    }
};

// Usage
const result = await hadesClient.query(
    "Explain ISNE training process",
    { mode: "hybrid", maxResults: 5 }
);
console.log(result.data.answer);
```

### cURL Examples

**Query system**:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does document processing work?",
    "mode": "hybrid",
    "max_results": 5
  }'
```

**Check system health**:

```bash
curl -X GET "http://localhost:8000/api/v1/admin/health" \
  -H "Authorization: Bearer your-api-key"
```

**Start training**:

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "training_type": "incremental",
    "config_override": {"learning_rate": 0.001}
  }'
```

## 🛠️ Development and Extension

### Adding New Endpoints

**1. Define the endpoint in `server.py`**:

```python
@app.post("/api/v1/custom/analyze", response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    try:
        result = await core_api.analyze_document(request)
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, errors=[str(e)])
```

**2. Add business logic in `core.py`**:

```python
async def analyze_document(request: AnalysisRequest) -> AnalysisResult:
    # Implement analysis logic
    return AnalysisResult(...)
```

**3. Define models in `models.py`**:

```python
class AnalysisRequest(BaseModel):
    document_path: str
    analysis_type: str
    options: Dict[str, Any] = {}

class AnalysisResult(BaseModel):
    document_id: str
    analysis_results: Dict[str, Any]
    processing_time: float
```

**4. Add CLI command in `cli.py`**:

```python
@cli.command()
@click.argument('document_path')
@click.option('--analysis-type', default='full')
def analyze(document_path, analysis_type):
    """Analyze a document for specific patterns."""
    request = AnalysisRequest(
        document_path=document_path,
        analysis_type=analysis_type
    )
    result = core_api.analyze_document(request)
    click.echo(f"Analysis complete: {result.document_id}")
```

### Error Handling

Standardized error responses:

```python
class APIError(Exception):
    def __init__(self, message: str, code: str = "GENERIC_ERROR", details: Dict = None):
        self.message = message
        self.code = code
        self.details = details or {}

# Usage in endpoints
try:
    result = process_document(path)
except ValidationError as e:
    raise APIError("Invalid document format", "VALIDATION_ERROR", {"errors": e.errors()})
except FileNotFoundError:
    raise APIError("Document not found", "FILE_NOT_FOUND", {"path": path})
```

## 🔍 Monitoring and Logging

### Request Logging

All API requests are logged with:

- Request ID for tracing
- Endpoint and method
- Processing time
- Response status
- User identification (if authenticated)

### Metrics Collection

Built-in metrics for:

- Request rate and latency
- Error rates by endpoint
- Training job success/failure rates
- System resource usage

### Health Monitoring

Continuous health checks for:

- Database connectivity
- Model availability
- Storage accessibility
- Memory and CPU usage

## 📚 Related Documentation

- **Core Logic**: See individual module READMEs for business logic details
- **Models and Schemas**: See `types/api/` for detailed type definitions
- **Configuration**: See `config/README.md` for configuration options
- **Authentication**: See deployment documentation for security setup
- **Client Libraries**: Check `examples/` for client implementation samples

## 🎯 Best Practices

1. **Use proper HTTP status codes** - Follow REST conventions
2. **Validate all inputs** - Use Pydantic models for type safety
3. **Handle errors gracefully** - Provide meaningful error messages
4. **Log important operations** - Aid debugging and monitoring
5. **Version your APIs** - Maintain backward compatibility
6. **Secure sensitive endpoints** - Require authentication for admin operations
7. **Rate limit appropriately** - Protect against abuse while allowing normal usage
