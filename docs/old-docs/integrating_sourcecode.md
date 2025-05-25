# Integrating Source Code with Symbol Table Metadata

## Purpose

This document outlines a plan for tightly coupling source code chunks with their corresponding symbol table metadata for downstream data ingestion, embedding, and retrieval-augmented generation (RAG) workflows.

---

## Goals

- Ensure that every code chunk (function/class/module) is paired with its metadata (name, docstring, relationships, etc.) and its exact source code.
- Enable robust, traceable, and semantically rich embeddings and search.
- Support both programmatic and LLM/agent-driven ingestion workflows.

---

## Motivation

- **Consistency:** Prevents drift between code and metadata.
- **Traceability:** Enables linking embeddings and search results back to original code.
- **Rich Context:** Supports advanced search, navigation, and LLM tasks.

---

## Proposed Workflow

1. **Parsing:**
    - Use the MCP server to generate a symbol table and relationships for each file.
2. **Chunk Extraction:**
    - For each function/class/module, extract the exact source code from the file using the line numbers provided by the symbol table.
3. **Data Structure:**
    - Output a unified object for each code chunk, e.g.:

      ```json
      {
        "name": "foo",
        "source_code": "def foo(x): return x + 1",
        "docstring": "Adds one to x.",
        "lineno": 1,
        "relationships": [...],
        ...
      }
      ```

4. **Ingestion:**
    - Ingest these unified objects into the embedding or RAG pipeline.
5. **(Optional) LLM/Agent Post-Processing:**
    - Use LLMs/agents for summarization, advanced chunking, or enrichment as needed.

---

## Implementation Options

- **Extend MCP Server:**
  - Modify the MCP server to include `source_code` in its output for each symbol.
- **Post-Processing Script:**
  - Write a script to combine the MCP output with extracted code from files.

---

## Open Questions

- Should code comments be included in the `source_code` field?
- What chunking granularity is optimal (function, class, module, etc.)?
- How should changes in code be tracked and updated in the ingestion pipeline?

---

## Next Steps

1. Decide on implementation approach (server extension vs. post-processing).
2. Prototype extraction and pairing logic.
3. Test with real codebases and validate downstream embedding quality.
4. Iterate based on feedback from RAG/embedding tasks.

---

## Integration with HADES-PathRAG

After validating the MCP server's core capabilities in isolation, the next step is to integrate this server into the HADES-PathRAG project. This move will:

- Allow further development and testing in the actual context where the symbol/code coupling will be used for RAG workflows.
- Enable us to answer open questions (e.g., chunking granularity, inclusion of comments, update tracking) in the context of real downstream requirements.
- Ensure enhancements (like source code extraction, chunking, and metadata enrichment) are directly aligned with the needs of the RAG pipeline.

**Rationale:**

- The MCP server is already well-suited for IDE/agentic workflows, but RAG integration requires context-aware design decisions.
- Working within HADES-PathRAG will let us iterate quickly and validate that outputs are compatible with retrieval, embedding, and serving logic.

**Next Phase:**

- Migrate the MCP server codebase into HADES-PathRAG.
- Begin prototyping the unified code chunk + metadata extraction and ingestion logic within the new context.
- Use feedback from real RAG tasks to refine the pipeline.

*Document created: 2025-04-18. Updated: 2025-04-19.*
