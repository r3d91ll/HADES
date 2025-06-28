# HADES Research Bibliography

## Core Papers

### PathRAG: Path Retrieval Augmented Generation
- **Authors**: Cheng et al.
- **Date**: 2024
- **Key Concepts**: Multi-hop retrieval, path-based reasoning, graph traversal for RAG
- **Relevance**: Core algorithm for HADES implementation

### ISNE: Inductive Shallow Node Embedding
- **Authors**: [To be added]
- **Date**: 2024
- **Key Concepts**: Graph-aware embeddings, inductive learning, shallow architectures
- **Relevance**: Enhancement layer for PathRAG embeddings

## Recent Papers (Week of 2025-01-27)

### 1. MultiFinRAG: Multi-modal Financial RAG System
- **Date**: January 2025
- **Key Concepts**: Multi-modal integration, financial domain adaptation, regulatory compliance
- **Implementation Ideas**: 
  - Multi-modal edge types for code-documentation-config connections
  - Domain-specific adaptation strategies for our pipelines

### 2. Dynamic Context-Aware Prompt Recommendation for RAG
- **Date**: January 2025
- **Key Concepts**: Dynamic prompt optimization, context-aware retrieval, adaptive prompting
- **Implementation Ideas**:
  - Dynamic prompt generation based on graph context
  - Context-aware edge weighting adjustments

### 3. Engineering Agentic RAG Systems with Tool-Calling
- **Date**: January 2025
- **Key Concepts**: Tool integration, agentic workflows, MCP-style architectures
- **Implementation Ideas**:
  - Enhanced MCP integration patterns
  - Tool-aware graph construction

### 4. ComRAG: Compositional Retrieval-Augmented Generation
- **Date**: January 2025
- **Key Concepts**: Hierarchical retrieval, compositional reasoning, centroid-based memory
- **Implementation Ideas**:
  - Centroid-based dynamic updates for real-time code integration
  - Hierarchical graph structures for compositional reasoning

### 5. SACL: Semantic-Augmented Code Retrieval and Localization
- **Date**: January 2025 (arXiv: 2506.20081v2)
- **Key Concepts**: Textual bias in code retrieval, semantic augmentation, cross-modal reranking
- **Implementation Ideas**:
  - Semantic description generation for code chunks
  - Bias mitigation through normalized representations
  - Enhanced cross-modal edge creation

## Implementation Connections

### Current Implementation Status
- **PathRAG**: Core implementation in `src/pathrag/`
- **ISNE**: Training pipeline in `src/isne/`
- **Supra-weight Bootstrap**: Multi-dimensional edge weighting in `src/pipelines/bootstrap/supra_weight/`

### Research-to-Code Mappings
```yaml
research_mappings:
  PathRAG:
    implementation: "src/pathrag/PathRAG.py"
    enhancements:
      - ComRAG dynamic updates
      - SACL semantic augmentation
  
  ISNE:
    implementation: "src/isne/models/isne_model.py"
    enhancements:
      - Multi-modal embeddings from MultiFinRAG
      - Sequential processing improvements
  
  EdgeWeighting:
    implementation: "src/pipelines/bootstrap/supra_weight/core/supra_weight_calculator.py"
    enhancements:
      - SACL bias mitigation
      - Dynamic context-aware adjustments
```

## Self-Improving System Architecture

### Research Integration Pipeline
1. **Discovery**: Monitor arXiv, conferences, journals for relevant papers
2. **Ingestion**: Convert papers to structured format using docling
3. **Analysis**: Extract key concepts, algorithms, and implementation ideas
4. **Mapping**: Connect research concepts to existing code modules
5. **Generation**: Create improvement proposals and implementations
6. **Validation**: Test improvements against benchmarks

### Graph Structure for Research-Code Connections
```
Research Paper Node
├── Concepts (extracted entities)
├── Algorithms (implementation patterns)
├── Results (performance metrics)
└── Citations (related work)
    ↓
Theory-Practice Bridge Edges
    ↓
Code Module Node
├── Current Implementation
├── Performance Metrics
├── Test Coverage
└── Improvement Opportunities
```

### Automated Improvement Workflow
```python
class ResearchDrivenDevelopment:
    def discover_research(self):
        # Monitor sources for new papers
        pass
    
    def analyze_paper(self, paper):
        # Extract concepts and map to code
        pass
    
    def generate_improvements(self, mappings):
        # Create code improvements based on research
        pass
    
    def validate_improvements(self, code_changes):
        # Test and benchmark improvements
        pass
```

## Next Steps

1. **Implement Research Ingestion Pipeline**: 
   - Automated paper discovery
   - Structured extraction of concepts
   - Graph construction for papers

2. **Create Research-Code Bridge Detection**:
   - Identify which papers relate to which code modules
   - Strength scoring for connections
   - Impact assessment for potential improvements

3. **Build Improvement Generation System**:
   - Template-based code generation from research insights
   - A/B testing framework for improvements
   - Automated benchmarking and validation

## Vision

The ultimate goal is a system that:
- Continuously learns from the latest research
- Automatically identifies improvement opportunities
- Generates and tests enhancements
- Self-improves through research-driven development

This creates a virtuous cycle where:
```
Research → Understanding → Implementation → Validation → Better System → Better Research Integration
```