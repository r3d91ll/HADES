# Chunker Configuration Guide

This document provides detailed information about the chunking system in HADES-PathRAG, focusing on the configuration options for both AST-based code chunking and Chonky semantic text chunking.

## Overview

HADES-PathRAG uses two primary chunking systems:

1. **AST-based chunking** for code files (respects code structure)
2. **Chonky semantic chunking** for natural language content

The chunking configuration is managed through YAML files and can be loaded and validated using the `chunker_config.py` module.

## Configuration File

The default configuration file is located at `src/config/chunker_config.yaml`. You can also provide a custom configuration file path when loading the configuration.

## Language Mapping

The `chunker_mapping` section in the configuration file maps file types to the appropriate chunker:

```yaml
chunker_mapping:
  python: 'ast'
  javascript: 'ast'
  java: 'ast'
  cpp: 'ast'
  markdown: 'chonky'
  text: 'chonky'
  html: 'chonky'
  pdf: 'chonky'
  default: 'chonky'  # Fallback chunker for unknown file types
```

## AST-based Code Chunker Configuration

The `ast` section in the configuration file controls the behavior of the AST-based code chunker:

```yaml
ast:
  max_tokens: 2048        # Maximum tokens per chunk
  use_class_boundaries: true    # Respect class boundaries
  use_function_boundaries: true # Respect function boundaries
  extract_imports: true         # Create separate chunk for imports
  preserve_docstrings: true     # Keep docstrings with their symbol
```

### AST Chunker Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tokens` | int | 2048 | Maximum number of tokens per chunk |
| `use_class_boundaries` | bool | true | Whether to respect class boundaries when chunking |
| `use_function_boundaries` | bool | true | Whether to respect function boundaries when chunking |
| `extract_imports` | bool | true | Whether to create a separate chunk for imports |
| `preserve_docstrings` | bool | true | Whether to keep docstrings with their associated symbol |

## Chonky Semantic Text Chunker Configuration

The `chonky` section in the configuration file controls the behavior of the Chonky semantic text chunker:

```yaml
chonky:
  # Model settings
  model_id: "mirth/chonky_modernbert_large_1"  # Modern BERT model for improved chunking
  model_engine: "haystack"                    # Which model engine to use (haystack, huggingface, vllm)
  device: "cuda:0"                           # Device to load model on (cuda:0, cpu, etc.)
  
  # Chunking parameters
  max_tokens: 2048                           # Maximum tokens per chunk
  min_tokens: 64                             # Minimum tokens per chunk
  overlap_tokens: 200                        # Token overlap between chunks
  semantic_chunking: true                    # Use semantic chunking vs. static
  preserve_structure: true                   # Attempt to preserve document structure
  batch_size: 8                              # Batch size for processing multiple documents
  
  # Overlap context settings
  overlap_context:
    enabled: true                           # Whether to use the new overlap context structure
    store_pre_context: true                 # Store text before the chunk
    store_post_context: true                # Store text after the chunk
    max_pre_context_chars: 1000             # Maximum characters for pre-context
    max_post_context_chars: 1000            # Maximum characters for post-context
    store_position_info: true               # Store position information for precise reconstruction
  
  # Cache settings
  cache_with_device: true                   # Whether to include device in cache key
  cache_size: 4                             # Number of models to keep in cache
  
  # Model engine settings
  early_availability_check: true            # Check model engine availability at module initialization
  auto_start_engine: true                   # Automatically try to start the engine if not running
  max_startup_retries: 3                    # Maximum number of retries when starting the engine
  
  # Advanced settings
  tokenizer_name: null                      # Optional custom tokenizer (defaults to model_id)
  force_reload: false                       # Whether to force reload model even if cached
  trust_remote_code: true                   # Whether to trust remote code when loading model
  timeout: 30                               # Timeout in seconds for model operations
```

### Chonky Chunker Options

#### Model Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_id` | string | "mirth/chonky_modernbert_large_1" | The model ID to use for semantic chunking |
| `model_engine` | string | "haystack" | Which model engine to use (haystack, huggingface, vllm) |
| `device` | string | "cuda:0" | Device to load model on (cuda:0, cpu, etc.) |

#### Chunking Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tokens` | int | 2048 | Maximum number of tokens per chunk |
| `min_tokens` | int | 64 | Minimum number of tokens per chunk |
| `overlap_tokens` | int | 200 | Number of tokens to overlap between chunks |
| `semantic_chunking` | bool | true | Whether to use semantic chunking vs. static |
| `preserve_structure` | bool | true | Whether to attempt to preserve document structure |
| `batch_size` | int | 8 | Batch size for processing multiple documents |

#### Overlap Context Settings

The overlap context settings control how context is stored for each chunk, which is important for maintaining context across chunk boundaries:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Whether to use the new overlap context structure |
| `store_pre_context` | bool | true | Whether to store text before the chunk |
| `store_post_context` | bool | true | Whether to store text after the chunk |
| `max_pre_context_chars` | int | 1000 | Maximum characters for pre-context |
| `max_post_context_chars` | int | 1000 | Maximum characters for post-context |
| `store_position_info` | bool | true | Whether to store position information for precise reconstruction |

#### Cache Settings

The cache settings control how models are cached to improve performance:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cache_with_device` | bool | true | Whether to include device in cache key (useful for multi-GPU setups) |
| `cache_size` | int | 4 | Number of models to keep in cache |

#### Model Engine Settings

The model engine settings control how the model engine is initialized and managed:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `early_availability_check` | bool | true | Whether to check model engine availability at module initialization |
| `auto_start_engine` | bool | true | Whether to automatically try to start the engine if not running |
| `max_startup_retries` | int | 3 | Maximum number of retries when starting the engine |

#### Advanced Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tokenizer_name` | string | null | Optional custom tokenizer (defaults to model_id) |
| `force_reload` | bool | false | Whether to force reload model even if cached |
| `trust_remote_code` | bool | true | Whether to trust remote code when loading model |
| `timeout` | int | 30 | Timeout in seconds for model operations |

## Loading Configuration in Code

You can load the chunker configuration in your code using the `load_config` function from the `chunker_config` module:

```python
from src.config.chunker_config import load_config, get_chunker_for_language, get_chunker_config

# Load the default configuration
config = load_config()

# Or load a custom configuration
# config = load_config('/path/to/custom/config.yaml')

# Get the appropriate chunker type for a language
chunker_type = get_chunker_for_language('python', config)  # Returns 'ast'

# Get configuration for a specific chunker
ast_config = get_chunker_config('ast', config)
chonky_config = get_chunker_config('chonky', config)
```

## Chunk Output Format

### AST Chunker Output

The AST chunker produces chunks with the following structure:

```python
{
    "id": "chunk_id",
    "parent": "parent_id",
    "path": "/path/to/file.py",
    "type": "python",
    "content": "def example_function():\n    pass",
    "symbol_type": "function",
    "name": "example_function",
    "line_start": 10,
    "line_end": 11,
    "token_count": 42,
    "content_hash": "hash_value",
    "embedding": None  # Placeholder for future embedding
}
```

### Chonky Chunker Output

The Chonky chunker produces chunks with the following structure when overlap context is enabled:

```python
{
    "id": "chunk_id",
    "parent": "parent_id",
    "path": "/path/to/file.md",
    "type": "markdown",
    "content": "This is the main content of the chunk.",
    "overlap_context": {
        "pre_context": "Text that comes before the chunk.",
        "post_context": "Text that comes after the chunk.",
        "pre_context_start": 0,
        "pre_context_end": 32,
        "post_context_start": 70,
        "post_context_end": 100
    },
    "symbol_type": "paragraph",
    "name": "paragraph_0",
    "line_start": 0,
    "line_end": 0,
    "token_count": 42,
    "content_hash": "hash_value",
    "embedding": None  # Placeholder for future embedding
}
```

For backward compatibility, when overlap context is disabled, the output will have the following structure:

```python
{
    "id": "chunk_id",
    "parent": "parent_id",
    "path": "/path/to/file.md",
    "type": "markdown",
    "content": "This is the main content of the chunk.",
    "content_with_overlap": "Text that comes before the chunk. This is the main content of the chunk. Text that comes after the chunk.",
    "content_offset": 32,
    "content_length": 38,
    "symbol_type": "paragraph",
    "name": "paragraph_0",
    "line_start": 0,
    "line_end": 0,
    "token_count": 42,
    "content_hash": "hash_value",
    "embedding": None  # Placeholder for future embedding
}
```

## Best Practices

### Memory Optimization

- For large documents, consider reducing `max_pre_context_chars` and `max_post_context_chars` to save memory.
- Set `store_position_info` to `false` if you don't need precise reconstruction of the original document.

### Performance Optimization

- Enable `cache_with_device` when using multiple GPUs to avoid model loading conflicts.
- Increase `batch_size` for faster processing of multiple documents.
- Set `early_availability_check` to `false` in production environments where the model engine is already running.

### Quality Optimization

- Increase `overlap_tokens` for better context preservation across chunk boundaries.
- Set `min_tokens` to a higher value to avoid very small chunks.
- Enable `semantic_chunking` for better chunk boundaries based on content meaning.

## Troubleshooting

### Model Engine Issues

If you encounter issues with the model engine not starting:

1. Check that the model engine service is properly installed and configured.
2. Increase `max_startup_retries` to give the engine more time to start.
3. Set `auto_start_engine` to `false` and manually start the engine before running the chunker.

### Memory Issues

If you encounter memory issues when processing large documents:

1. Reduce `max_pre_context_chars` and `max_post_context_chars`.
2. Reduce `batch_size` to process fewer documents at once.
3. Use a smaller model by changing `model_id`.

### Chunking Quality Issues

If you're not satisfied with the chunking quality:

1. Adjust `max_tokens` and `min_tokens` to get chunks of appropriate size.
2. Increase `overlap_tokens` for better context preservation.
3. Try a different model by changing `model_id`.
