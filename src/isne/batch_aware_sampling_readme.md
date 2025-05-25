# Batch-Aware Sampling for ISNE Training

## Overview

Batch-aware sampling is an optimization technique implemented in the RandomWalkSampler to significantly improve ISNE training efficiency. By ensuring that positive and negative pairs are sampled only from nodes within the current mini-batch, this approach eliminates or greatly reduces the filtering of out-of-bounds pairs during training.

## Implementation Details

### Core Components

1. **RandomWalkSampler Extensions**:
   - `sample_positive_pairs_within_batch`: Samples positive pairs only from nodes present in the current batch
   - `sample_negative_pairs_within_batch`: Generates negative pairs constrained to the current batch with fallback mechanism

2. **ISNETrainer Integration**:
   - Auto-detection of batch-aware sampling capability
   - Configuration-based enabling/disabling
   - Enhanced metrics for tracking filtering rates

3. **Configuration**:
   - Enable via `use_batch_aware_sampling: True` in sampler configuration

## Benefits

1. **Reduced Filtering Rate**: Guarantees that sampled pairs contain nodes present in the current batch, eliminating the need to filter out pairs containing out-of-bounds nodes.

2. **Improved Training Efficiency**: By reducing filtering operations, more of the sampled pairs can be used for actual training, making each epoch more efficient.

3. **Better Convergence**: Consistent availability of training pairs leads to more stable gradient updates, potentially improving model convergence.

## Usage

To enable batch-aware sampling in the ISNE training pipeline:

```python
sampler_config = {
    "sampler_class": RandomWalkSampler,
    "sampler_params": {
        # Standard parameters
        "walk_length": 6,
        "context_size": 4,
        "walks_per_node": 10,
        "p": 1.0,
        "q": 0.7,
        "num_negative_samples": 1,
        # Enable batch-aware sampling
        "use_batch_aware_sampling": True
    }
}

# Pass to trainer or orchestrator
trainer = ISNETrainer(..., sampler_config=sampler_config)
```

## Performance Metrics

During integration testing, the batch-aware sampling implementation demonstrated:

- 0% filtering rate (compared to variable rates with standard sampling)
- Use of fallback mechanism when needed to ensure sufficient pairs
- Fast training completion (50 epochs in <2 seconds on test dataset)

## Limitations and Future Work

1. **Fallback Pairs**: When insufficient pairs exist within a batch, the implementation falls back to adding some pairs that may contain out-of-bounds nodes. These are still subject to filtering during loss computation.

2. **Small Batches**: Very small batches may struggle to generate enough pairs, leading to more reliance on the fallback mechanism.

3. **Potential Improvements**:

   - Adaptive batch sizing based on graph connectivity
   - Improved fallback strategies for difficult batches
   - Caching common pair patterns for faster generation

## Test Coverage

Unit and integration tests have been implemented to verify:
- Correct pair generation within batch boundaries
- Proper fallback behavior when needed
- Integration with the ISNE training pipeline
- Performance characteristics under realistic conditions
