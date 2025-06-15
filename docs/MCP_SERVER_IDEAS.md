# MCP Server Ideas for HADES

## Document Processing MCP Server

**Discovered Need**: During ISNE evaluation, we needed to read the research paper PDF but Claude couldn't read binary files. We used HADES' own document processing components to extract the text - this would make an excellent MCP server!

### Core Concept
Transform HADES document processing capabilities into MCP tools that AI assistants can use to read and analyze documents.

### Proposed Tools

#### Basic Document Reading
```yaml
tools:
  - name: read_pdf
    description: Extract text content from PDF files
    input:
      file_path: string
    output:
      content: string
      metadata: object
      
  - name: read_document
    description: Read any supported document format
    input:
      file_path: string
      format: string (optional)
    output:
      content: string
      metadata: object
      format_detected: string
```

#### Advanced Extraction
```yaml
tools:
  - name: extract_sections
    description: Extract hierarchical document structure
    input:
      file_path: string
    output:
      sections: array of {title, content, level}
      
  - name: extract_tables
    description: Extract tables as structured data
    input:
      file_path: string
    output:
      tables: array of {headers, rows, caption}
      
  - name: extract_code_blocks
    description: Extract code snippets with language detection
    input:
      file_path: string
    output:
      code_blocks: array of {language, content, line_numbers}
```

#### Research-Specific Tools
```yaml
tools:
  - name: extract_citations
    description: Extract bibliography and references
    input:
      file_path: string
    output:
      citations: array of {authors, title, year, venue}
      
  - name: extract_equations
    description: Extract mathematical equations
    input:
      file_path: string
    output:
      equations: array of {latex, context, equation_number}
      
  - name: extract_figures
    description: Extract figure captions and descriptions
    input:
      file_path: string
    output:
      figures: array of {caption, description, figure_number}
```

### Implementation Approach

1. **Wrapper around existing components**:
   ```python
   from src.components.docproc.factory import create_docproc_component
   
   class DocumentMCPServer:
       def __init__(self):
           self.processors = {
               'docling': create_docproc_component('docling'),
               'core': create_docproc_component('core')
           }
   ```

2. **Smart format detection**:
   - Use file extensions and magic bytes
   - Route to appropriate processor
   - Handle errors gracefully

3. **Caching layer**:
   - Cache processed documents
   - Return cached results for repeated requests
   - Clear cache on demand

### Use Cases

1. **Research Assistant**:
   - Read and summarize research papers
   - Extract key findings and methodologies
   - Compare multiple papers

2. **Documentation Analysis**:
   - Process technical documentation
   - Extract API references
   - Find code examples

3. **Legal/Contract Review**:
   - Extract key terms and conditions
   - Find specific clauses
   - Compare document versions

4. **Knowledge Base Building**:
   - Process entire documentation sets
   - Build searchable indexes
   - Create knowledge graphs

### Integration Benefits

- **Leverages existing HADES infrastructure**: No new document processing code needed
- **Proven technology**: Already used in production for ISNE training
- **Extensible**: Easy to add new document types and extraction features
- **Performance**: Can use GPU-accelerated processing where applicable

### Example Usage in Claude

```python
# Claude could use it like this:
result = mcp.read_pdf("/path/to/research_paper.pdf")
print(f"Paper title: {result.metadata.title}")
print(f"Abstract: {result.content[:1000]}")

# Extract specific information
tables = mcp.extract_tables("/path/to/data_report.pdf")
for table in tables:
    print(f"Table: {table.caption}")
    # Analyze table data...

# Process code documentation
code_blocks = mcp.extract_code_blocks("/path/to/tutorial.pdf")
for block in code_blocks:
    if block.language == "python":
        # Analyze or execute code...
```

### Development Priority

**After chat interface is operational**, this would be a high-value addition because:
1. Solves a real problem we encountered
2. Uses existing, tested components
3. Provides immediate value to AI assistants
4. Relatively simple to implement
5. Can be extended incrementally

### Related Ideas

- **ISNE Model MCP Server**: Expose ISNE embeddings via MCP
- **PathRAG Query MCP Server**: Allow AI assistants to query the RAG system
- **Graph Analysis MCP Server**: Tools for analyzing document relationships

---

**Note**: This idea emerged organically from our work on ISNE evaluation when we needed to read the research paper. Classic example of "eating our own dog food" leading to useful tool ideas!