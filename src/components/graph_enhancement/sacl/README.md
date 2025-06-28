# SACL-Inspired Graph Enhancement

This module implements semantic augmentation techniques inspired by the SACL paper to enhance our PathRAG implementation.

## Key Features

1. **Semantic Description Generation**
   - Generates natural language descriptions for code chunks
   - Creates cross-modal edges between code and descriptions
   - Mitigates bias towards well-documented code

2. **Dual Representation Storage**
   - Maintains both original and normalized code representations
   - Enables retrieval across different documentation levels
   - Reduces textual bias in retrieval

3. **Enhanced Edge Weighting**
   - Combines semantic and structural similarities
   - Uses configurable α parameter for weighting
   - Integrates with supra-weight calculations

## Integration with PathRAG

This enhancement works alongside our existing ISNE and supra-weight components to provide:
- Better semantic understanding of code
- Reduced bias in retrieval
- Improved cross-modal connections
- Enhanced repository-level understanding

## Configuration

```yaml
sacl_enhancement:
  enabled: true
  semantic_weight: 0.7  # α parameter from SACL paper
  generate_descriptions: true
  normalize_identifiers: true
  description_model: "llama-3.1-8b"
```

## Benefits for HADES

1. **Improved Code Retrieval**: Addresses the textual bias issue
2. **Better Cross-Modal Understanding**: Strengthens code-documentation relationships
3. **Enhanced ANT/STS Bridges**: Provides richer semantic connections
4. **Reduced Bias**: Normalizes the advantage of well-documented code
```