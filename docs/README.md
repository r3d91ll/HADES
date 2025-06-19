# HADES Documentation

Comprehensive documentation for the HADES (Heuristic Adaptive Data Extrapolation System) project.

## 📚 Documentation Organization

### 🏗️ **Core Architecture**

- [**HADES Service Architecture**](./HADES_SERVICE_ARCHITECTURE.md) - System design and component overview
- [**Ingestion System**](./ingestion_system.md) - Document processing and pipeline architecture
- [**Streaming Pipeline Architecture**](./streaming_pipeline_architecture.md) - Stream processing design

### 📊 **Database & Storage**

- [**Unified Database Schema**](./schemas/hades_unified_database_schema.md) - Current production schema design
- [**ArangoDB Setup**](./integration/arango_setup.md) - Database configuration and setup

### 🔬 **Analysis & Reports**

- [**ISNE Production Success Report**](./analysis/ISNE_PRODUCTION_SUCCESS_REPORT.md) - ✅ **Current production achievements**
- [**Enhanced E2E Test Guide**](./enhanced_e2e_test_guide.md) - Testing methodology and results

### 🛠️ **Development & Integration**

- [**Incremental ISNE Architecture**](./development/ArangoDB_Incremental_ISNE_Architecture.md) - Future incremental processing design
- [**Hybrid Pipeline Implementation**](./hybrid_pipeline_implementation_plan.md) - Implementation strategy
- [**MCP Server Ideas**](./MCP_SERVER_IDEAS.md) - Model Context Protocol integration

### 🔌 **Integration Guides**

- [**ArangoDB Setup**](./integration/arango_setup.md) - Database configuration
- [**Docker Deployment**](./integration/docker_deployment.md) - Containerized deployment
- [**GitHub Ingestion**](./integration/github_ingestion.md) - Code repository processing
- [**HADES Integration**](./integration/hades_integration.md) - System integration patterns
- [**vLLM Embeddings**](./integration/vllm_embeddings.md) - High-performance embedding generation
- [**Chunker Configuration**](./integration/chunker_configuration.md) - Document chunking strategies
- [**JSON Schema Analysis**](./integration/json_schema_analysis.md) - Data structure analysis

### 📖 **Legacy Documentation**

- [**Original Database Schema**](./legacy/hades_database_schema.md) - Historical schema design
- [**ISNE Data Analysis Report**](./legacy/ISNE_DATA_ANALYSIS_REPORT.md) - Previous analysis (superseded)

### 📄 **Research Papers**

- [**PathRAG Paper**](./PathRAG_paper.pdf) - Original PathRAG research
- [**ISNE Paper**](./Unsupervised%20Graph%20Representation%20Learning%20with%20Inductive%20Shallow%20Node%20Embedding.pdf) - ISNE methodology
- [**ISNE Paper Extract**](./isne_paper_extracted.txt) - Text extraction of ISNE research

## 🎯 **Quick Navigation**

### For New Users
1. Start with [HADES Service Architecture](./HADES_SERVICE_ARCHITECTURE.md)
2. Review [ISNE Production Success Report](./analysis/ISNE_PRODUCTION_SUCCESS_REPORT.md)
3. Follow [ArangoDB Setup](./integration/arango_setup.md)

### For Developers
1. Review [Hybrid Pipeline Implementation](./hybrid_pipeline_implementation_plan.md)
2. Study [Incremental ISNE Architecture](./development/ArangoDB_Incremental_ISNE_Architecture.md)
3. Follow [Enhanced E2E Test Guide](./enhanced_e2e_test_guide.md)

### For Integration
1. Check [Integration Guides](./integration/) directory
2. Review [MCP Server Ideas](./MCP_SERVER_IDEAS.md)
3. Study [Unified Database Schema](./schemas/hades_unified_database_schema.md)

## 📈 **Current Status**

- ✅ **Production Ready**: Complete ISNE pipeline successfully deployed
- ✅ **API Integration**: REST endpoints and MCP tools available
- ✅ **91,165 Enhanced Embeddings**: Graph-aware representations created
- ✅ **Semantic Collections**: Code and documentation analysis complete
- ✅ **Professional Architecture**: Organized codebase with 85%+ test coverage

## 🔄 **Documentation Maintenance**

This documentation is organized into categories based on relevance and usage:

- **Current/Active**: Reflects the production-ready system
- **Development**: Future enhancements and architectural evolution
- **Integration**: How to connect and deploy HADES
- **Legacy**: Historical documents for reference
- **Analysis**: Performance reports and system evaluation

## 📝 **Contributing to Documentation**

When adding new documentation:

1. **Choose the appropriate directory** based on content type
2. **Update this README** to include new documents
3. **Link related documents** for easy navigation
4. **Follow the established naming conventions**
5. **Include status indicators** (✅ current, 🔄 in progress, 📚 legacy)

---

*Last Updated: December 18, 2025*  
*HADES Version: Production v1.0*