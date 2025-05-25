# MCP Recursive Implementation Guide

This document details the recursive Model Context Protocol (MCP) implementation that enables HADES to improve itself - the "HADES builds HADES" pattern.

## Conceptual Overview

The recursive MCP implementation leverages a self-referential architecture where HADES can analyze, understand, and modify its own codebase through a continuous feedback loop:

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│              Self-Improvement Cycle                    │
│                                                        │
│  ┌──────────┐      ┌──────────────┐      ┌──────────┐  │
│  │          │      │              │      │          │  │
│  │  Analyze ├─────►│  Understand  ├─────►│  Improve │  │
│  │          │      │              │      │          │  │
│  └──────────┘      └──────────────┘      └──────────┘  │
│       │                                        │       │
│       └────────────────────────────────────────┘       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

This pattern is inspired by the academic research on computational universality in memory-augmented LLMs and the foundational work on self-improving AI systems.

## MCP Server Implementation

### Core Server Configuration

```python
# src/mcp/server.py
import asyncio
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..pathrag.xnx_pathrag import XnXPathRAG

class MCPServer:
    """MCP server implementation with recursive capabilities."""
    
    def __init__(self, host="0.0.0.0", port=8123, pathrag=None):
        """Initialize the MCP server."""
        self.host = host
        self.port = port
        self.app = FastAPI(title="HADES-PathRAG MCP Server")
        self.pathrag = pathrag or XnXPathRAG()
        self.tools = {}
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        @self.app.post("/mcp/tools")
        async def handle_tool_call(request: Request):
            """Handle MCP tool call."""
            try:
                data = await request.json()
                tool_name = data.get("name")
                parameters = data.get("parameters", {})
                
                if tool_name not in self.tools:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                result = await self.tools[tool_name](**parameters)
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/status")
        async def get_status():
            """Get MCP server status."""
            return {
                "status": "online",
                "tools": list(self.tools.keys()),
                "version": "1.0.0"
            }
    
    def register_tool(self, name, handler):
        """Register an MCP tool handler."""
        self.tools[name] = handler
    
    async def start(self):
        """Start the MCP server."""
        import uvicorn
        
        # Register default tools
        if not self.tools:
            self._register_default_tools()
        
        # Start the server
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
    
    def _register_default_tools(self):
        """Register default MCP tools."""
        from .tools import (
            mcp0_pathrag_retrieve,
            mcp0_ingest_data,
            mcp0_xnx_pathrag_retrieve,
            mcp0_self_analyze
        )
        
        self.register_tool("mcp0_pathrag_retrieve", mcp0_pathrag_retrieve)
        self.register_tool("mcp0_ingest_data", mcp0_ingest_data)
        self.register_tool("mcp0_xnx_pathrag_retrieve", mcp0_xnx_pathrag_retrieve)
        self.register_tool("mcp0_self_analyze", mcp0_self_analyze)
```

### Self-Referential Tool Implementation

```python
# src/mcp/tools.py
async def mcp0_self_analyze(query, target_components=None, min_confidence=0.7):
    """
    MCP tool for HADES to analyze its own codebase.
    
    This recursive tool enables HADES to examine its own implementation,
    understand its components, and propose improvements.
    
    Args:
        query: Natural language query about HADES itself
        target_components: Optional list of specific components to analyze
        min_confidence: Minimum confidence threshold for results
    
    Returns:
        Analysis of the requested HADES components
    """
    # Initialize PathRAG
    from ..pathrag.xnx_pathrag import XnXPathRAG
    pathrag = XnXPathRAG()
    
    # Build self-referential query
    if target_components:
        query = f"{query} Focus on the following components: {', '.join(target_components)}"
    
    # Execute query with XnX parameters
    paths = await pathrag.query(
        query,
        min_weight=min_confidence,
        max_distance=3,
        # Only consider outbound relationships for code analysis
        direction=-1
    )
    
    # Analyze code quality
    code_quality = await analyze_code_quality(paths)
    
    # Generate improvement suggestions
    suggestions = await generate_improvement_suggestions(paths, code_quality)
    
    return {
        "paths": paths,
        "code_quality": code_quality,
        "improvement_suggestions": suggestions,
        "query_parameters": {
            "query": query,
            "target_components": target_components,
            "min_confidence": min_confidence
        }
    }
```

## Self-Improvement Implementation

The core of the recursive implementation is the self-improvement system:

```python
# src/self_improvement/analyzer.py
class CodeAnalyzer:
    """Analyzer for HADES codebase self-improvement."""
    
    def __init__(self, pathrag):
        """Initialize with PathRAG connection."""
        self.pathrag = pathrag
    
    async def analyze_code_quality(self, paths):
        """
        Analyze code quality based on retrieved paths.
        
        Args:
            paths: List of paths from PathRAG
            
        Returns:
            Code quality metrics
        """
        # Extract code from paths
        code_segments = self._extract_code_segments(paths)
        
        # Calculate metrics
        metrics = {
            "complexity": self._calculate_complexity(code_segments),
            "maintainability": self._calculate_maintainability(code_segments),
            "test_coverage": self._calculate_test_coverage(code_segments),
            "performance": self._calculate_performance(code_segments)
        }
        
        return metrics
    
    async def generate_improvement_suggestions(self, paths, metrics):
        """
        Generate suggestions for improving code quality.
        
        Args:
            paths: List of paths from PathRAG
            metrics: Code quality metrics
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze each metric
        for metric, value in metrics.items():
            if value < 0.7:  # Below threshold
                suggestion = await self._generate_suggestion_for_metric(metric, paths)
                suggestions.append(suggestion)
        
        # Analyze potential optimizations
        optimizations = await self._identify_optimizations(paths)
        suggestions.extend(optimizations)
        
        return suggestions
    
    def _extract_code_segments(self, paths):
        """Extract code segments from paths."""
        code_segments = []
        
        for path in paths:
            for node in path.get("nodes", []):
                if node.get("entity_type") == "code_function" or node.get("entity_type") == "code_class":
                    code_segments.append({
                        "content": node.get("content", ""),
                        "file_path": node.get("metadata", {}).get("file_path", ""),
                        "entity_type": node.get("entity_type", ""),
                        "name": node.get("name", "")
                    })
        
        return code_segments
    
    def _calculate_complexity(self, code_segments):
        """Calculate code complexity."""
        # Implementation details
        return 0.85  # Example value
    
    def _calculate_maintainability(self, code_segments):
        """Calculate code maintainability."""
        # Implementation details
        return 0.78  # Example value
    
    def _calculate_test_coverage(self, code_segments):
        """Calculate test coverage."""
        # Implementation details
        return 0.65  # Example value
    
    def _calculate_performance(self, code_segments):
        """Calculate performance metrics."""
        # Implementation details
        return 0.92  # Example value
    
    async def _generate_suggestion_for_metric(self, metric, paths):
        """Generate improvement suggestion for a specific metric."""
        # Implementation details
        return {
            "metric": metric,
            "current_value": metrics[metric],
            "target_value": 0.8,
            "suggestion": f"Improve {metric} by refactoring X to do Y",
            "affected_files": ["file1.py", "file2.py"]
        }
    
    async def _identify_optimizations(self, paths):
        """Identify potential optimizations."""
        # Implementation details
        return [
            {
                "type": "performance",
                "description": "Optimize database query in X by using indexing",
                "affected_files": ["db.py"],
                "estimated_improvement": "30% faster query execution"
            }
        ]
```

## IDE Integration for Self-Improvement

The self-improvement cycle connects to the IDE through the MCP:

```python
# src/self_improvement/ide_integration.py
class IDEIntegration:
    """Integration with IDE for self-improvement."""
    
    def __init__(self, mcp_server):
        """Initialize with MCP server connection."""
        self.mcp_server = mcp_server
    
    async def suggest_improvements(self, suggestions):
        """
        Send improvement suggestions to the IDE.
        
        Args:
            suggestions: List of improvement suggestions
            
        Returns:
            Response from IDE
        """
        # Format suggestions for IDE
        ide_suggestions = self._format_for_ide(suggestions)
        
        # Send to IDE through custom MCP channel
        response = await self.mcp_server.send_to_ide(
            channel="hades.self_improvement",
            data={
                "suggestions": ide_suggestions,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return response
    
    def _format_for_ide(self, suggestions):
        """Format suggestions for IDE consumption."""
        ide_formatted = []
        
        for suggestion in suggestions:
            ide_formatted.append({
                "title": f"Improve {suggestion.get('metric', 'code')}",
                "description": suggestion.get("suggestion", ""),
                "affected_files": suggestion.get("affected_files", []),
                "severity": self._calculate_severity(suggestion),
                "actions": [
                    {
                        "name": "Apply Suggestion",
                        "command": "hades.apply_suggestion",
                        "arguments": [suggestion]
                    },
                    {
                        "name": "View Details",
                        "command": "hades.view_suggestion_details",
                        "arguments": [suggestion]
                    }
                ]
            })
        
        return ide_formatted
    
    def _calculate_severity(self, suggestion):
        """Calculate suggestion severity."""
        # Implementation details
        return "medium"
```

## Recursive MCP Runtime Flow

Here's how the recursive system operates in runtime:

1. **Initialization**:
   - The MCP server starts and registers tools
   - HADES codebase is ingested into the knowledge graph
   - Self-improvement components are initialized

2. **Self-Analysis Trigger**:
   - Triggered by a scheduled job or explicit request
   - The system queries its own codebase through PathRAG

3. **Code Analysis**:
   - Code quality metrics are calculated
   - Improvement opportunities are identified
   - Suggestions are generated

4. **IDE Integration**:
   - Suggestions are sent to the IDE through MCP
   - The IDE presents suggestions to the developer
   - The developer accepts or rejects suggestions

5. **Implementation**:
   - Accepted suggestions are implemented
   - Code changes are committed to the repository
   - The system re-ingests the updated codebase

6. **Continuous Improvement**:
   - The cycle repeats, creating a continuous improvement loop
   - The system learns from past suggestions and their outcomes
   - Knowledge graph evolves with the codebase

## Advanced: Computational Universality

The recursive MCP implementation draws inspiration from the concept of computational universality in memory-augmented LLMs, as discussed in recent academic research:

> "Memory Augmented Large Language Models are Computationally Universal"
> https://arxiv.org/html/2503.02113v1

By creating a "memory augmented" HADES through the PathRAG knowledge graph, we enable a system that can:

1. **Store its own program code** (as knowledge graph nodes)
2. **Understand its own implementation** (through PathRAG queries)
3. **Modify its own behavior** (through IDE integration)

This creates a form of computational universality where the system can theoretically perform any computation, including improving its own algorithms.

## Security Considerations

The self-referential nature of this system introduces unique security considerations:

```python
# src/security/recursion_limits.py
class RecursionGuard:
    """Guard against unsafe recursive modifications."""
    
    def __init__(self):
        """Initialize recursion guard."""
        self.modification_history = []
        self.safety_rules = self._load_safety_rules()
    
    def validate_modification(self, modification):
        """
        Validate a proposed self-modification.
        
        Args:
            modification: Proposed code modification
            
        Returns:
            (bool, str): (is_safe, reason)
        """
        # Check against safety rules
        for rule in self.safety_rules:
            if rule.matches(modification) and not rule.is_allowed:
                return False, f"Modification violates safety rule: {rule.name}"
        
        # Check for recursive modification loops
        if self._is_recursive_loop(modification):
            return False, "Potential recursive modification loop detected"
        
        # Record modification if it passes checks
        self.modification_history.append({
            "modification": modification,
            "timestamp": datetime.now().isoformat()
        })
        
        return True, "Modification is safe"
    
    def _load_safety_rules(self):
        """Load safety rules."""
        # Implementation details
        return [
            SecurityRule(
                name="no_security_bypass",
                pattern=".*(?:bypass|disable).*security.*",
                is_allowed=False
            ),
            SecurityRule(
                name="no_self_improvement_modification",
                pattern=".*self_improvement(?:\\.py|\\/analyzer\\.py).*",
                is_allowed=False
            )
        ]
    
    def _is_recursive_loop(self, modification):
        """Check for recursive modification loops."""
        # Implementation details
        return False  # Example return
```

## Implementation Roadmap

### Phase 1: Self-Analysis

- [x] Implement MCP server with basic tools
- [x] Implement code ingestion for HADES codebase
- [x] Implement basic code analysis functionality
- [ ] Develop metrics for code quality assessment

### Phase 2: Self-Improvement Suggestions

- [ ] Implement suggestion generation
- [ ] Create IDE integration for suggestions
- [ ] Develop feedback mechanism for suggestion quality

### Phase 3: Controlled Self-Modification

- [ ] Implement security controls for self-modification
- [ ] Create workflow for reviewing and applying changes
- [ ] Develop metrics for measuring improvement effectiveness

### Phase 4: Continuous Self-Improvement

- [ ] Implement automated analysis scheduling
- [ ] Develop historical tracking of improvements
- [ ] Create visualization of system evolution

## Conclusion

The recursive MCP implementation enables HADES to analyze, understand, and improve its own codebase through a continuous feedback loop. This self-referential architecture represents a significant step toward systems that can genuinely improve themselves, drawing on cutting-edge research on computational universality in AI systems.
