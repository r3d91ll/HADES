# HADES MCP Tools Testing Checklist

## Overview

This document serves as a comprehensive testing checklist for all MCP HADES tools. The objective is to systematically test each tool, document all deficiencies and failures for further analysis, and fix as little as possible in place during testing.

**Testing Philosophy**: Document issues, don't fix in place. The goal is comprehensive issue identification for batch resolution.

## Test Environment Setup

- [ ] Verify HADES server is running on port 8595
- [ ] Confirm MCP integration is active
- [ ] Test data directory prepared: `/path/to/test/data`
- [ ] ArangoDB connection verified
- [ ] Test databases available: `test_training_db`, `test_production_db`

## Available MCP Tools

### ISNE Training Pipeline (`/isne/*`)

- `mcp__hades__start_training_isne_train_post`
- `mcp__hades__get_job_status_isne_jobs__job_id__get`  
- `mcp__hades__cancel_job_isne_jobs__job_id__delete`
- `mcp__hades__list_jobs_isne_jobs_get`

### Production Pipeline (`/prod/*`)

- `mcp__hades__bootstrap_data_prod_bootstrap_post`
- `mcp__hades__train_isne_model_prod_train_post`
- `mcp__hades__apply_isne_model_prod_apply_model_post`
- `mcp__hades__build_collections_prod_build_collections_post`
- `mcp__hades__get_pipeline_status_prod_status__operation_id__get`
- `mcp__hades__list_operations_prod_status_get`
- `mcp__hades__run_pipeline_prod_run_pipeline_post`

---

## Test Set 1: ISNE Training Pipeline Tools

### Test 1.1: Start Training with Default Parameters

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__start_training_isne_train_post`
- [ ] **Parameters**: `{}`
- [ ] **Expected**: Job starts with auto-generated ID
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 1.2: Start Training with Custom Parameters

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__start_training_isne_train_post`
- [ ] **Parameters**:

```json
{
    "epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 64,
    "data_percentage": 0.1,
    "enable_evaluation": true,
    "job_id": "test_training_job"
}
```

- [ ] **Expected**: Job starts with specified parameters
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 1.3: List All Training Jobs

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__list_jobs_isne_jobs_get`
- [ ] **Parameters**: `{}`
- [ ] **Expected**: Returns list of all active jobs
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 1.4: Get Job Status

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__get_job_status_isne_jobs__job_id__get`
- [ ] **Parameters**: `{"job_id": "test_training_job"}`
- [ ] **Expected**: Returns detailed job status and progress
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 1.5: Cancel/Remove Job

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__cancel_job_isne_jobs__job_id__delete`
- [ ] **Parameters**: `{"job_id": "test_training_job"}`
- [ ] **Expected**: Job removed from tracking
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

---

## Test Set 2: Production Pipeline Tools

### Test 2.1: Bootstrap Data

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__bootstrap_data_prod_bootstrap_post`
- [ ] **Parameters**:

```json
{
    "input_dir": "/path/to/test/data",
    "debug_logging": true,
    "similarity_threshold": 0.8
}
```

- [ ] **Expected**: Data bootstrap operation starts
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.2: Train ISNE Model

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__train_isne_model_prod_train_post`
- [ ] **Parameters**:

```json
{
    "database_name": "test_training_db",
    "epochs": 25,
    "batch_size": 256,
    "learning_rate": 0.001
}
```

- [ ] **Expected**: Production training starts
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.3: Apply ISNE Model

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__apply_isne_model_prod_apply_model_post`
- [ ] **Parameters**:

```json
{
    "model_path": "/path/to/trained/model.pth",
    "source_db": "test_training_db",
    "target_db": "test_production_db",
    "similarity_threshold": 0.85
}
```

- [ ] **Expected**: Model application starts
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.4: Build Collections

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__build_collections_prod_build_collections_post`
- [ ] **Parameters**: `{"database_name": "test_production_db"}`
- [ ] **Expected**: Semantic collections built
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.5: List Operations

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__list_operations_prod_status_get`
- [ ] **Parameters**: `{}`
- [ ] **Expected**: Returns all pipeline operations
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.6: Get Pipeline Status

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__get_pipeline_status_prod_status__operation_id__get`
- [ ] **Parameters**: `{"operation_id": "bootstrap_20241220_120000"}`
- [ ] **Expected**: Returns specific operation status
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 2.7: Run Complete Pipeline

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__run_pipeline_prod_run_pipeline_post`
- [ ] **Parameters**:

```json
{
    "input_dir": "/path/to/test/data",
    "database_prefix": "full_test"
}
```

- [ ] **Expected**: End-to-end pipeline execution
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

---

## Test Set 3: Error Handling & Edge Cases

### Test 3.1: Invalid Learning Rate

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__start_training_isne_train_post`
- [ ] **Parameters**: `{"learning_rate": 1.5}`
- [ ] **Expected**: Error response for invalid parameter
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 3.2: Invalid Data Percentage

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__start_training_isne_train_post`
- [ ] **Parameters**: `{"data_percentage": 2.0}`
- [ ] **Expected**: Error response for out-of-range value
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 3.3: Non-existent Job ID

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__get_job_status_isne_jobs__job_id__get`
- [ ] **Parameters**: `{"job_id": "nonexistent_job"}`
- [ ] **Expected**: 404 error response
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 3.4: Non-existent Operation ID

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__get_pipeline_status_prod_status__operation_id__get`
- [ ] **Parameters**: `{"operation_id": "fake_operation_id"}`
- [ ] **Expected**: 404 error response
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 3.5: Missing Required Parameters - Bootstrap

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__bootstrap_data_prod_bootstrap_post`
- [ ] **Parameters**: `{}`
- [ ] **Expected**: Error for missing input_dir
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 3.6: Missing Required Parameters - Apply Model

- [ ] **Test Status**: Not Started
- [ ] **Tool**: `mcp__hades__apply_isne_model_prod_apply_model_post`
- [ ] **Parameters**: `{}`
- [ ] **Expected**: Error for missing model_path
- [ ] **Actual Result**:
- [ ] **Issues Found**:
- [ ] **Notes**:

---

## Test Set 4: Integration Workflows

### Test 4.1: End-to-End ISNE Training Workflow

- [ ] **Test Status**: Not Started
- [ ] **Description**: Complete training workflow with monitoring
- [ ] **Steps**:
  1. [ ] Start training job with custom parameters
  2. [ ] Monitor job status until completion
  3. [ ] Verify training results
  4. [ ] Clean up job
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 4.2: Full Production Pipeline Workflow

- [ ] **Test Status**: Not Started
- [ ] **Description**: Complete production pipeline from data to collections
- [ ] **Steps**:
  1. [ ] Bootstrap test data
  2. [ ] Monitor bootstrap status
  3. [ ] Train ISNE model
  4. [ ] Monitor training status
  5. [ ] Apply trained model
  6. [ ] Build semantic collections
  7. [ ] Verify final production database
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 4.3: Concurrent Operations

- [ ] **Test Status**: Not Started
- [ ] **Description**: Multiple operations running simultaneously
- [ ] **Steps**:
  1. [ ] Start multiple training jobs simultaneously
  2. [ ] Start production pipeline while training is running
  3. [ ] Monitor all operations
  4. [ ] Verify resource handling
- [ ] **Issues Found**:
- [ ] **Notes**:

---

## Test Set 5: Performance & Monitoring

### Test 5.1: Resource Usage During Training

- [ ] **Test Status**: Not Started
- [ ] **Description**: Monitor system resources during ISNE training
- [ ] **Metrics to Check**:
  - [ ] Memory usage
  - [ ] GPU utilization (if available)
  - [ ] Training checkpoints
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 5.2: Large Dataset Processing

- [ ] **Test Status**: Not Started
- [ ] **Description**: Test with full dataset
- [ ] **Parameters**: `{"data_percentage": 1.0}`
- [ ] **Metrics to Check**:
  - [ ] Processing time
  - [ ] Resource usage
  - [ ] Scalability
- [ ] **Issues Found**:
- [ ] **Notes**:

### Test 5.3: Pipeline Recovery

- [ ] **Test Status**: Not Started
- [ ] **Description**: Test error recovery mechanisms
- [ ] **Steps**:
  - [ ] Test resuming interrupted operations
  - [ ] Verify error recovery mechanisms
  - [ ] Check data consistency after failures
- [ ] **Issues Found**:
- [ ] **Notes**:

---

## Issue Summary

### Critical Issues

*To be filled during testing*

### Major Issues  

*To be filled during testing*

### Minor Issues

*To be filled during testing*

### Enhancement Suggestions

*To be filled during testing*

---

## Testing Progress

- **Total Tests**: 24
- **Completed**: 0
- **Failed**: 0
- **Passed**: 0
- **Skipped**: 0

**Last Updated**: [Date]  
**Tested By**: [Name]  
**HADES Version**: [Version]  
**Environment**: [Environment Details]

---

## Notes and Observations

*Document any general observations, patterns, or insights discovered during testing*
