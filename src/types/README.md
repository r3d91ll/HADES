# HADES Type System

This directory contains the type definitions for all HADES components and subsystems, organized by functional area.

## Directory Structure

The type system is organized into component-specific and system-wide types:

```
src/types/
├── components/            # Component-specific types
│   ├── contracts.py       # Pydantic contracts for all component types
│   ├── protocols.py       # Protocol interfaces for component implementations
│   ├── chunking/          # Chunking component types
│   ├── docproc/           # Document processing component types
│   ├── embedding/         # Embedding component types
│   ├── isne/              # ISNE component types
│   ├── model_engine/      # Model engine component types
│   ├── storage/           # Storage component types
│   ├── database/          # Database component types
│   └── schemas/           # Schema component types
├── components_impl/       # Component implementation-specific types
│   ├── chunking/          # Chunking component implementation types
│   │   ├── chonky/        # GPU-accelerated chunking types
│   │   └── text/          # Text chunking types
│   └── model_engine/      # Model engine implementation types
│       ├── core/          # Core model engine types
│       ├── vllm/          # vLLM engine types
│       └── haystack/      # Haystack engine types
├── api/                   # API-specific types
├── alerts/                # Alert system types
├── config/                # Configuration types
├── documents/             # Document structure types
├── orchestration/         # Orchestration types
├── pipeline/              # Pipeline types
├── utils/                 # Utility types
└── validation/            # Validation types
```

## Type Categories

### Component Types (`components/`)
- **contracts.py**: Pydantic models defining input/output contracts for all component types
- **protocols.py**: Protocol interfaces that all component implementations must follow
- **chunking/**: General chunking component types and interfaces
- **docproc/**: Document processing component types
- **embedding/**: Embedding component types
- **isne/**: ISNE component types
- **model_engine/**: Model engine component types
- **storage/**: Storage component types

### Component Implementation Types (`components_impl/`)
Component-specific type definitions for different implementations:
- Configuration models
- Performance metrics
- State management
- Error handling
- Implementation-specific data structures

### System-Wide Types
General type definitions used across the system:
- **api/**: REST API request/response models
- **alerts/**: Alert and monitoring types
- **orchestration/**: Pipeline orchestration types
- **validation/**: Data validation types

## Type Design Principles

### 1. Contract Compliance
All component implementations must conform to their respective contracts defined in `components/contracts.py`.

### 2. Protocol Adherence
Component implementations must implement the protocols defined in `components/protocols.py`.

### 3. Pydantic Validation
All data structures use Pydantic v2 for:
- Runtime validation
- Serialization/deserialization
- JSON schema generation
- Type safety

### 4. Immutability
Data models are designed to be immutable where possible for thread safety.

### 5. Extensibility
Types support extension and customization through:
- Optional fields
- Union types
- Generic parameters
- Inheritance

## Usage Examples

### Component Contracts
```python
from src.types.components.contracts import (
    ChunkingInput,
    ChunkingOutput,
    EmbeddingInput,
    EmbeddingOutput
)

# Create type-safe input
chunking_input = ChunkingInput(
    text="Sample text to chunk",
    document_id="doc_1",
    chunk_size=512,
    chunk_overlap=50
)
```

### Component Implementation Types
```python
from src.types.components_impl.model_engine.vllm.types import (
    vLLMServerConfig,
    vLLMInferenceStats
)

# Configure vLLM engine
config = vLLMServerConfig(
    model_name="meta-llama/Llama-2-7b",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)
```

### Protocol Implementation
```python
from src.types.components.protocols import Chunker
from src.types.components.contracts import ChunkingInput, ChunkingOutput

class MyChunker(Chunker):
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        # Implementation follows contract
        pass
```

## Type Validation

All types include comprehensive validation:

```python
from src.types.components.contracts import ChunkingInput
from pydantic import ValidationError

try:
    # This will raise ValidationError
    invalid_input = ChunkingInput(
        text="",  # Empty text not allowed
        document_id="",  # Empty ID not allowed
        chunk_size=-1,  # Negative size not allowed
        chunk_overlap=1000  # Overlap > chunk_size not allowed
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Adding New Types

### For New Component Implementations
1. Create directory: `components_impl/new_component/implementation_name/`
2. Add `types.py` with Pydantic models
3. Add `__init__.py` with exports
4. Follow existing patterns for configuration, metrics, and state types

### For New System Types
1. Create directory: `new_subsystem/`
2. Add type definitions following Pydantic patterns
3. Update `__init__.py` exports
4. Add documentation and examples

## Type Safety

The HADES type system provides:
- **Compile-time safety**: MyPy static analysis
- **Runtime validation**: Pydantic model validation
- **IDE support**: Full autocompletion and error detection
- **API documentation**: Automatic OpenAPI schema generation