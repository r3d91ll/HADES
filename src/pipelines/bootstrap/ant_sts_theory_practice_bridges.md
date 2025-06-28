# ANT/STS Theory-Practice Bridges in HADES

## Overview

The enhanced supra-weight bootstrap pipeline now incorporates Actor-Network Theory (ANT) and Science & Technology Studies (STS) principles to create semantic bridges between:

- **Research Papers** (Theory)
- **Documentation** (Intermediaries)
- **Code Implementation** (Practice)

These bridges enable ISNE to learn relationships that connect abstract theoretical concepts to concrete implementations.

## Bridge Types

### 1. Theory-to-Implementation Bridges
Direct connections between research papers and code that implements their concepts.
- **Example**: ANT paper on "actor-networks" ↔ `NetworkXStorage` class
- **Weight**: 0.9 (high confidence)
- **Detection**: Shared technical terms, concept overlap

### 2. Documentation Bridges
Documentation acts as an intermediary, translating theory into practice.
- **Example**: README explaining graph theory ↔ graph algorithm implementation
- **Weight**: 0.85 (high confidence)
- **Role**: Mediators in ANT terms, actively transforming knowledge

### 3. Concept Translation
Abstract concepts becoming concrete data structures or algorithms.
- **Example**: "Knowledge representation" theory ↔ embedding data structures
- **Weight**: Variable (0.4-0.95) based on semantic similarity
- **Categories**:
  - High-confidence (>0.8): Direct implementation
  - Cross-domain (0.6-0.8): Related concepts
  - Novel discovery (0.4-0.6): Unexpected connections

### 4. Actor-Network Formation
How heterogeneous actors (code, docs, papers) form stable networks.
- **Example**: Import relationships + documentation + theory citations
- **Weight**: 0.8
- **Validates**: ANT principle of heterogeneous networks

### 5. Black-Box Inscription
Complex theory simplified into usable interfaces.
- **Example**: Complex graph theory → Simple PathRAG API
- **Weight**: 0.8
- **STS Focus**: How technology stabilizes and becomes taken-for-granted

### 6. Sociotechnical Assembly
Human-machine interfaces and their social-technical nature.
- **Example**: API design patterns ↔ user interaction theories
- **Weight**: 0.4-0.6 (novel bridges)
- **Insight**: Technology and society co-construct each other

### 7. Knowledge Inscription
How knowledge becomes inscribed in technical artifacts.
- **Example**: Theoretical models → Database schemas
- **Weight**: 0.85
- **Process**: Translation and materialization of concepts

## Implementation Details

### Theory-Practice Detector

The `TheoryPracticeDetector` class identifies bridges by:

1. **Node Classification**
   - Theory: Papers with academic language, citations
   - Practice: Code with implementation patterns
   - Documentation: READMEs, guides, tutorials

2. **Bridge Detection**
   - Content analysis for shared concepts
   - Semantic similarity using embeddings
   - Structural patterns (imports, references)

3. **ANT/STS Validation**
   - Heterogeneous network verification
   - Translation process tracking
   - Black-boxing identification

### Configuration

The `config_ant_sts.yaml` provides:
- Relaxed thresholds for novel bridge discovery
- Higher weights for reference and semantic relationships
- Support for mixed content types
- ANT/STS specific tracking

### Analysis Tools

The `ant_sts_bridge_analyzer.py` validates:
- Bridge distribution across categories
- Actor-network heterogeneity
- ANT/STS principle adherence
- Bridge quality metrics

## Expected Outcomes

### Quantitative
- 100+ theory-practice bridges
- 20+ high-confidence direct implementations
- 50+ cross-domain conceptual bridges
- 30+ novel discovery connections

### Qualitative
- **Validated ANT Principles**: Heterogeneous networks of human/non-human actors
- **STS Coherence**: Evidence of technology-society co-construction
- **Translation Accuracy**: Clear paths from theory to practice
- **Network Stabilization**: How implementations crystallize theory

## Usage Example

```python
from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline
import yaml

# Load ANT/STS configuration
with open('config_ant_sts.yaml') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = SupraWeightBootstrapPipeline(config)

# Process mixed content (papers + code + docs)
results = pipeline.run("/path/to/mixed/content")

# Analyze bridges
from ant_sts_bridge_analyzer import ANTSTSBridgeAnalyzer
analyzer = ANTSTSBridgeAnalyzer(pipeline.storage)
analysis = analyzer.analyze_bridges()
```

## Benefits for ISNE Training

1. **Richer Semantic Space**: Bridges create connections across domains
2. **Contextual Understanding**: ISNE learns theory-practice relationships
3. **Knowledge Transfer**: Enables reasoning from abstract to concrete
4. **Validated Framework**: ANT/STS principles ensure meaningful connections

The theory-practice bridges transform ISNE from a simple embedding model into a knowledge-aware system that understands how abstract concepts materialize in code.