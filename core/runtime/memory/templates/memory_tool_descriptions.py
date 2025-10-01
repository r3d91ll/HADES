"""
Tool descriptions for experiential memory operations.

These descriptions are designed to be included in agent system prompts
for autonomous memory management. Prompt engineering patterns inspired by:
https://github.com/doobidoo/mcp-memory-service

All operations run locally in-process via direct ArangoDB connections.
No network services, no MCP protocol - pure Python library.
"""

TOOL_DESCRIPTIONS = {
    "store_observation": {
        "name": "store_observation",
        "description": (
            "Store a significant experience or learning for future reference. "
            "Use this when you fix bugs, make design decisions, discover patterns, "
            "or complete significant work. Do NOT use for routine operations or "
            "trivial observations."
        ),
        "parameters": {
            "content": {
                "type": "string",
                "description": (
                    "The observation text. Include what happened, why it matters, "
                    "how it was done, and the results. Be specific and include context."
                ),
                "required": True,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-5 meaningful tags for categorization. Include technical area "
                    "(debugging, architecture), component (authentication, database), "
                    "and optional status (solved, experimental)."
                ),
                "required": False,
            },
            "memory_type": {
                "type": "string",
                "enum": ["observation", "decision", "learning", "event"],
                "description": (
                    "Type of memory: 'observation' for factual events, 'decision' for "
                    "choices with rationale, 'learning' for patterns/principles, "
                    "'event' for milestones."
                ),
                "required": False,
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Additional context like component name, confidence score, "
                    "test results, or project information."
                ),
                "required": False,
            },
        },
        "examples": [
            {
                "content": "Fixed race condition in authentication by adding mutex locks around session access. Testing showed 0 failures after 10k concurrent requests.",
                "tags": ["debugging", "concurrency", "authentication", "solved"],
                "memory_type": "decision",
                "metadata": {"component": "auth_system", "confidence": 0.9},
            }
        ],
    },
    "retrieve_memories": {
        "name": "retrieve_memories",
        "description": (
            "Retrieve past experiences using semantic search. Use this when the user "
            "asks about past work, when encountering similar problems, before making "
            "architectural decisions, or when building on previous work."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Natural language query describing what you're looking for.",
                "required": True,
            },
            "n_results": {
                "type": "integer",
                "description": "Maximum number of memories to retrieve (default: 5).",
                "required": False,
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum relevance threshold 0.0-1.0 (default: 0.7).",
                "required": False,
            },
            "time_filter": {
                "type": "string",
                "description": (
                    "Natural language time filter: 'today', 'yesterday', 'last week', "
                    "'last month', 'last year', or 'last N days'."
                ),
                "required": False,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by specific tags.",
                "required": False,
            },
        },
        "examples": [
            {
                "query": "How have we handled concurrency issues in authentication?",
                "n_results": 5,
                "time_filter": "last month",
                "tags": ["concurrency", "authentication"],
            }
        ],
    },
    "search_by_tag": {
        "name": "search_by_tag",
        "description": (
            "Search memories by specific tags when you know the exact categories. "
            "Faster than semantic search for category-based lookups."
        ),
        "parameters": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to search for.",
                "required": True,
            },
            "match_all": {
                "type": "boolean",
                "description": (
                    "If true, require all tags; if false, match any tag (default: false)."
                ),
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 10).",
                "required": False,
            },
        },
        "examples": [
            {"tags": ["architecture", "database"], "match_all": True, "limit": 10}
        ],
    },
    "list_observations": {
        "name": "list_observations",
        "description": (
            "List recent observations with optional session filtering. "
            "Use for reviewing recent work or browsing session history."
        ),
        "parameters": {
            "session_id": {
                "type": "string",
                "description": "Filter by specific session (optional).",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 50).",
                "required": False,
            },
        },
        "examples": [{"session_id": "chat_abc123", "limit": 20}],
    },
}


# Concise single-line descriptions for compact tool lists
TOOL_DESCRIPTIONS_COMPACT = {
    "store_observation": "Store significant experiences, decisions, patterns, or milestones for future reference.",
    "retrieve_memories": "Search past experiences using natural language semantic queries with time/tag filters.",
    "search_by_tag": "Find memories by specific category tags (faster than semantic search).",
    "list_observations": "List recent observations with optional session filtering.",
}


# Guidelines for when to use memory operations (for system prompt)
USAGE_GUIDELINES = """
Memory Operation Guidelines:

WHEN TO STORE OBSERVATIONS:
- After solving bugs or making design decisions
- When discovering patterns or insights
- Upon completing significant work
- When user provides important context

WHEN NOT TO STORE:
- Routine file operations (read, write, query)
- Trivial observations with no future value
- Redundant information already in memory
- Every single action you take

BEFORE MAKING DECISIONS:
1. Search for similar past decisions/problems
2. Consider historical context
3. Store new decisions with rationale

TAGGING STRATEGY:
- Use 3-5 meaningful tags per observation
- Include: technical area + component + optional status
- Examples: ["debugging", "concurrency", "authentication", "python", "solved"]

QUALITY > QUANTITY:
- Focus on memories with lasting value
- Include context, rationale, and outcomes
- Think "Would I want to recall this in a month?"
"""