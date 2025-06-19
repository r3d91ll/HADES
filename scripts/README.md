# HADES Scripts Organization

This directory contains organized scripts for the HADES project. Scripts have been categorized for better maintainability and clarity.

## Directory Structure

### Production Pipelines (Moved to `src/pipelines/production/`)

**Note: Production scripts have been moved to their proper architectural location:**

- `src/pipelines/production/bootstrap_full_isne_testdata.py` - Bootstrap ISNE test dataset into ArangoDB
- `src/pipelines/production/train_isne_memory_efficient.py` - Memory-efficient ISNE model training  
- `src/pipelines/production/apply_efficient_isne_model.py` - Apply trained ISNE model to create production database
- `src/pipelines/production/build_semantic_collections.py` - Build semantic collections for production use
- `src/pipelines/production/create_arango_graph.py` - Create ArangoDB graph visualizations

These are now properly integrated as pipeline components rather than standalone scripts.

### `/bootstrap/`
**Scripts for data ingestion and initial setup:**

- `bootstrap_full_dataset.py` - Bootstrap complete dataset
- `bootstrap_10percent_sample.py` - Bootstrap sample dataset for testing
- `bootstrap_micro_validation.py` - Ultra-fast validation (30 files, 2-3 minutes)
- `bootstrap_tiny_validation.py` - Minimal validation test

### `/experiments/`
**Experimental and testing scripts:**

- `analyze_full_dataset.py` - Analyze dataset contents before processing
- `analyze_graph_connectivity.py` - Analyze graph structure and connectivity
- `analyze_isne_data_readiness.py` - Validate data readiness for ISNE training
- `debug_gpu_config.py` - Debug and optimize GPU configurations
- `inspect_arango_databases.py` - Database inspection and validation
- `quick_isne_check.py` - Quick ISNE model validation
- `quick_model_sanity_check.py` - Model sanity checking utilities
- `quick_test_isne_training.py` - Fast ISNE training test (1 epoch)
- `validate_configs.py` - Configuration file validation
- `validate_isne_embeddings.py` - Embedding quality validation

### `/archive/`
**Deprecated and one-time use scripts:**

- `extract_graph_from_results.py` - Graph extraction utilities
- `extract_pdf_text.py` - PDF text extraction tools
- `query_bootstrap_results.py` - Query bootstrap results from ArangoDB
- `reconstruct_graph_data.py` - Graph data reconstruction utilities

### `/deprecated/`
**Obsolete scripts moved during cleanup:**

- Contains 41+ obsolete experimental and training scripts
- Historical development iterations no longer needed
- Duplicate functionality superseded by production API

## API Integration

The production scripts have been integrated into the HADES FastAPI server as endpoints:

### Production Pipeline API (`/api/v1/production/`)

- `POST /bootstrap` - Bootstrap ISNE dataset
- `POST /train` - Train ISNE model
- `POST /apply-model` - Apply trained model
- `POST /build-collections` - Build semantic collections
- `POST /run-complete-pipeline` - Run entire pipeline end-to-end
- `GET /status/{operation_id}` - Check operation status
- `GET /status` - List all operations

### MCP Tools

Production operations are also available as MCP tools for external integration:

- `mcp_bootstrap_isne_dataset()`
- `mcp_train_isne_model()`
- `mcp_apply_isne_model()`
- `mcp_build_semantic_collections()`
- `mcp_run_complete_pipeline()`
- `mcp_get_operation_status()`
- `mcp_wait_for_completion()`

## Usage Patterns

### 1. Development and Testing
Use scripts in `/experiments/` and `/bootstrap/` for development work:

```bash
# Run experimental training variations
python experiments/train_isne_gpu_optimized.py

# Test bootstrap processes
python bootstrap/bootstrap_micro_validation.py
```

### 2. Production Operations
Use the FastAPI endpoints or MCP tools for production operations:

```bash
# Start the API server
python -m src.api.server

# Use curl or HTTP client to trigger operations
curl -X POST "http://localhost:8595/api/v1/production/run-complete-pipeline" \
  -d "input_dir=/path/to/data&database_prefix=prod"
```

### 3. Direct Pipeline Execution
The production pipelines can still be run directly:

```bash
# Run complete pipeline manually
cd src/pipelines/production
python bootstrap_full_isne_testdata.py
python train_isne_memory_efficient.py  
python apply_efficient_isne_model.py
python build_semantic_collections.py
```

## Scripts Cleanup Summary

**Before Cleanup:** 70+ scripts scattered in the root `/scripts/` directory  
**After Organization:** 23 active scripts in 4 logical categories + 50+ moved to deprecated  
**Final Result:** Clean, organized workspace with API integration

### Benefits:
1. **Clear separation** between production-ready and experimental code
2. **API endpoints** for external integration
3. **MCP tools** for programmatic access  
4. **Background execution** with status tracking
5. **Reduced clutter** in the workspace
6. **Better maintainability** and code organization

## Next Steps

1. **Archive cleanup**: Remove truly obsolete scripts from `/archive/`
2. **Documentation**: Add individual README files for complex experiments
3. **Testing**: Add integration tests for the API endpoints
4. **Monitoring**: Add metrics and logging to production endpoints
5. **Deployment**: Create Docker containers for production pipeline operations