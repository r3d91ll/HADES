# Experiential Memory System - Agent Guidelines

You have access to an experiential memory system that allows you to store observations, retrieve past experiences, and build a knowledge graph of entities and relationships.

## Available Memory Operations

### store_observation(content, tags, memory_type, metadata)

Store significant experiences and learnings for future reference.

**When to use:**
- After fixing bugs or making design decisions (store the solution and rationale)
- When discovering patterns or insights (e.g., "race conditions often occur in...")
- Upon completing significant work (milestone achievements, breakthroughs)
- When user provides important context about their project/preferences

**When NOT to use:**
- Routine file operations (reading, writing, simple queries)
- Trivial observations with no future value
- Redundant information already in memory
- Every single action you take

**Memory Types:**
- `observation`: Factual events you witness ("Fixed auth bug by adding mutex")
- `decision`: Choices made with rationale ("Chose ArangoDB for graph capabilities")
- `learning`: Patterns or principles discovered ("Concurrency bugs need careful state management")
- `event`: Significant milestones ("Completed PathRAG integration v1.0")

**Tags:** Use meaningful, searchable tags:
- Technical areas: `debugging`, `architecture`, `performance`, `testing`
- Components: `authentication`, `database`, `api`, `frontend`
- Languages: `python`, `go`, `javascript`, `rust`
- Concepts: `concurrency`, `caching`, `optimization`, `security`

**Example:**
```python
store_observation(
    content="Fixed race condition in authentication by adding mutex locks around session access. "
            "Testing showed 0 race conditions after 10k concurrent requests.",
    tags=["debugging", "concurrency", "authentication", "solved"],
    memory_type="decision",
    metadata={
        "component": "auth_system",
        "confidence": 0.9,
        "test_results": "10k requests, 0 failures"
    }
)
```

### retrieve_memories(query, n_results, time_filter, tags)

Search your past experiences semantically using natural language.

**When to use:**
- User asks about past work ("What did we do last week?")
- Encountering similar problems (search for "authentication bugs")
- Building on previous decisions (query "why we chose ArangoDB")
- Before making architectural decisions (check if similar decisions were made)
- When stuck on a problem (search for related past solutions)

**Time filters:** Use natural language
- "today", "yesterday"
- "last week", "last month", "last year"
- "last 7 days", "last 30 days"

**Example:**
```python
results = retrieve_memories(
    query="How have we handled concurrency issues in the past?",
    n_results=5,
    time_filter="last month",
    tags=["concurrency", "debugging"]
)
```

### search_by_tag(tags, match_all)

Find memories by category when you know specific tags.

**When to use:**
- Browsing memories in a specific domain ("show me all database work")
- Finding all instances of a pattern (all `debugging` + `concurrency` memories)
- Quick category-based lookups without semantic search

**Example:**
```python
memories = search_by_tag(
    tags=["architecture", "database"],
    match_all=True  # Require both tags
)
```

## Memory Management Guidelines

### Quality Over Quantity

**Store memories that:**
- Have lasting value (solutions, patterns, decisions)
- Represent significant work or insights
- Would be useful to recall weeks/months later
- Contain rationale and context, not just facts

**Don't store:**
- Every file you read or edit
- Routine operations with no insights
- Information easily discoverable in code/docs
- Temporary state or intermediate steps

### Rich Context

When storing observations, include:
- **What:** The observation itself
- **Why:** The rationale or context
- **How:** Key implementation details
- **Result:** Outcomes or metrics

**Good:** "Switched from REST to GraphQL for user API because it eliminated N+1 queries. Response time improved from 800ms to 120ms p95. Schema introspection helps frontend development."

**Bad:** "Changed API to GraphQL."

### Tagging Strategy

Use 3-5 tags per observation:
- At least one technical area tag
- At least one component/domain tag
- Optional: language, pattern, or status tags

**Good tags:** `["debugging", "concurrency", "authentication", "python", "solved"]`

**Bad tags:** `["bug", "thing", "code", "work"]` (too vague)

### Retrieval Before Storage

**Before storing a potentially duplicate observation:**
1. Search for similar memories
2. If found, consider updating/consolidating instead of creating new
3. Only store if it adds genuinely new information

**Example workflow:**
```python
# Check if we already have memories about this
existing = retrieve_memories(
    query="authentication mutex patterns",
    n_results=3,
    time_filter="last month"
)

if not existing:
    # No similar memories - safe to store
    store_observation(content=..., tags=...)
else:
    # Consider if this adds new information
    # If yes, store with reference to existing
    # If no, skip storage
```

### Session Boundaries

Memories are automatically linked to conversation sessions. At session end:
- Significant work should have 3-10 observations
- Complex sessions may generate 1-2 reflections (auto-consolidated)
- Not every turn needs an observation

## Advanced: Reflection and Entity Extraction

Reflections are higher-order insights automatically generated from multiple observations. You don't need to create these manually - the system will consolidate related observations into reflections periodically.

Entities (people, components, concepts) are automatically extracted from observations and linked in a knowledge graph. Focus on storing good observations, and the graph will emerge naturally.

## Integration with Your Workflow

### During Problem-Solving

1. **Check memory first:** Search for similar past problems
2. **Work on solution:** Standard problem-solving process
3. **Store outcome:** Record solution, rationale, and results

### During Design Discussions

1. **Retrieve context:** Search for past architectural decisions
2. **Present options:** Inform user with historical context
3. **Store decision:** Record chosen approach and rationale

### End of Session

1. **Review work:** What were the significant achievements?
2. **Store milestones:** 2-5 observations about key accomplishments
3. **Tag well:** Enable future retrieval

## Error Handling

If memory operations fail:
- Log the failure but continue task execution
- Memory is enhancing, not blocking, functionality
- User's immediate task takes priority over memory storage

## Privacy and Security

- Never store sensitive information (passwords, tokens, keys)
- User data should be anonymized or generalized
- Focus on patterns and solutions, not private details