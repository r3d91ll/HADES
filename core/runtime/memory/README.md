# core/runtime/memory/ — Experiential Memory System

Parent: `../../README.md`

## Overview

Local Python library implementing experiential memory for agents. Stores observations, reflections, entities, and relationships in an ArangoDB knowledge graph.

**Prompt engineering patterns inspired by:**
https://github.com/doobidoo/mcp-memory-service

**Infrastructure powered by HADES stack:**
- ArangoDB graph database (Unix socket connections)
- PathRAG retrieval with graph traversal (H≤2, F≤3, B≤8)
- Jina v4 embeddings (32k context, 2048 dimensions)

**All operations run in-process - no network services, no MCP protocol.**

## Memory Types

- **observations**: Raw experiences witnessed by the agent (bug fixes, decisions, events)
- **reflections**: Higher-order insights synthesized from multiple observations
- **entities**: People, components, concepts, projects extracted from experiences
- **relationships**: Graph edges connecting entities and memories (uses, causes, relates_to)

## Core Modules

- pathrag.py — PathRAG retrieval over memory graph (caps, reasons)
- plan_executor.py — query plan contract normalization
- memory_store.py — Arango persistence for memories, entities, relationships
- model_engine.py — Qwen3 model loader for reflection generation
- operations.py — Local library API (store, retrieve, search, list, archive)
- consolidation.py — Reflection generation from observations
- templates/ — System prompts for autonomous memory management

## Related Docs
- Docs/Program_Plan.md (Phases C–E)
- Docs/PRD_PathRAG_Integration.md
- Docs/SCHEMA_Experiential_Memory.md (collection specifications)

## Inspiration
- Prompt engineering patterns: https://github.com/doobidoo/mcp-memory-service
- Memory types and operation naming inspired by their approach
- Implementation completely different (ArangoDB + PathRAG vs SQLite-vec)