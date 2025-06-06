# XnX Traversal Guide for HADES-PathRAG

## 📖 Purpose

This document explains how to use **XnX Notation** constraints during graph traversal within the HADES-PathRAG project, including **weight filtering**, **directional flow**, and **temporal filtering**.

---

## 📌 XnX Concept Recap

### XnX Notation Format:
```
weight node_id direction [optional temporal window]
```
Example:
```
0.92 FunctionA -1 [2020-01-01 → 2025-01-01]
```
- **weight**: Confidence or strength (0.0 - 1.0)
- **node_id**: Target node
- **direction**: `-1` outbound / `1` inbound
- **temporal window**: Optional time range when the relationship is valid

---

## 🔄 XnX-Based Graph Traversal Example

### Basic Traversal with XnX Constraints
```python
from src.xnx.arango_adapter import get_arango_db

def traverse_with_xnx_constraints(
    start_node: str,
    min_weight: float = 0.8,
    max_distance: int = 3,
    direction: str = "any"  # 'inbound', 'outbound', or 'any'
):
    db = get_arango_db()
    graph_name = "HADES_GRAPH"

    direction_filter = ""
    if direction == "outbound":
        direction_filter = "FILTER e.direction == -1"
    elif direction == "inbound":
        direction_filter = "FILTER e.direction == 1"

    query = f"""
    FOR v, e, p IN 1..{max_distance}
        OUTBOUND @start_node
        GRAPH @graph_name
        FILTER e.weight >= @min_weight
        {direction_filter}
        RETURN {{
            node: v._key,
            path_weight: SUM(p.edges[*].weight),
            path_length: LENGTH(p.edges),
            edges: p.edges
        }}
    """

    bind_vars = {
        "start_node": start_node,
        "graph_name": graph_name,
        "min_weight": min_weight
    }

    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return [doc for doc in cursor]
```

### Example Call:
```python
results = traverse_with_xnx_constraints(
    start_node="entities/developer.john_doe",
    min_weight=0.85,
    max_distance=2,
    direction="outbound"
)
```

---

## ⏳ Advanced: Temporal Filtering
```python
def traverse_with_temporal_xnx(
    start_node: str,
    min_weight: float = 0.8,
    max_distance: int = 3,
    direction: str = "any",
    valid_at: str = "2025-01-01T00:00:00Z"
):
    db = get_arango_db()
    graph_name = "HADES_GRAPH"

    direction_filter = ""
    if direction == "outbound":
        direction_filter = "FILTER e.direction == -1"
    elif direction == "inbound":
        direction_filter = "FILTER e.direction == 1"

    query = f"""
    FOR v, e, p IN 1..{max_distance}
        OUTBOUND @start_node
        GRAPH @graph_name
        FILTER e.weight >= @min_weight
        FILTER @valid_at >= e.valid_from AND @valid_at <= e.valid_to
        {direction_filter}
        RETURN {{
            node: v._key,
            path_weight: SUM(p.edges[*].weight),
            path_length: LENGTH(p.edges),
            edges: p.edges
        }}
    """

    bind_vars = {
        "start_node": start_node,
        "graph_name": graph_name,
        "min_weight": min_weight,
        "valid_at": valid_at
    }

    cursor = db.aql.execute(query, bind_vars=bind_vars)
    return [doc for doc in cursor]
```

### Example Temporal Query:
```python
results = traverse_with_temporal_xnx(
    start_node="entities/policy.company_policy",
    min_weight=0.75,
    max_distance=3,
    direction="inbound",
    valid_at="2025-01-01T00:00:00Z"
)
```

---

## 👌 Automatic XnX String Output Formatter

Use this helper function to format the output edges into standard `weight node_id direction` XnX strings:

```python
def format_xnx_output(edge):
    """
    Takes a graph edge dict and returns formatted XnX string.
    """
    return f"{edge['weight']:.2f} {edge['_to']} {edge['direction']}"

# Example usage during traversal result processing:
for result in results:
    for edge in result['edges']:
        print(format_xnx_output(edge))
```

Example Output:
```
0.92 entities/module.data_processor -1
0.88 entities/database.main -1
```

---

## 🔢 Mathematical Foundations of XnX

### 📈 Graph Theory Basis

XnX notation operates over a **directed weighted graph (digraph)**:
- **G = (V, E)**
  - **V**: Set of nodes (vertices)
  - **E**: Set of directed edges **(u, v, w, d, t_1, t_2)** with attributes:
    - **u**: Source node
    - **v**: Target node
    - **w**: Edge weight ∈ [0, 1]
    - **d**: Direction of flow (**-1** for outbound, **+1** for inbound)
    - **[t_1 → t_2]**: Validity window (temporal range)

### ⬆ Directionality as Signed Flow

- Edge **(u → v)** with **d = -1** means `u` influences or flows toward `v`
- Edge **(v → u)** with **d = +1** means flow into `u` from `v`

### 📐 Path Scoring Function

For a path **P = {e_1, e_2, ..., e_n}**, the total score is:

\[
\text{PathScore}(P) = \prod_{i=1}^n w_i
\]

Alternatively, log-scaled for numerical stability:

\[
\log(\text{PathScore}(P)) = \sum_{i=1}^n \log(w_i)
\]

Higher scores imply stronger confidence paths.

### ⏳ Temporal Constraint

A path **P** is valid at time **t** if:

\[
\forall e_i \in P, (t_{1i} \leq t \leq t_{2i})
\]

Meaning every edge is valid at query time.

### 🔒 Access Control (ACL) Model

Access to resource **r** from actor **a** exists if:

\[
\exists P(a \to r) : \text{PathScore}(P) \geq \tau \quad \land \quad \max(\text{distance}(P)) \leq D
\]

Where **τ** is a minimum threshold.

---

## 🚀 Future Additions & Optimizations

### Performance Optimization
- **Mojo Implementation**: Core XnX traversal algorithms will be migrated to Mojo for performance gains while maintaining Python API
- **Parallel Processing**: Implementing concurrent path evaluations for faster query results
- **Caching Strategy**: Frequently accessed paths and subgraphs will be cached

### Error Handling Strategy
```python
try:
    result = traverse_with_xnx_constraints(...)
except InvalidNodeError as e:
    logger.error(f"Node does not exist: {e}")
    # Fallback strategy
except WeightThresholdError as e:
    logger.warning(f"No paths meeting weight requirement: {e}")
    # Try with lower threshold
except TemporalConstraintError as e:
    logger.warning(f"No valid paths at requested time: {e}")
    # Try with expanded time window
```

### Model Integration Framework
- **GNN-Based Traversal**: Graph Neural Networks for efficient XnX constraint handling
- **Dual Model Approach**:
  - Code-specific model for programming contexts
  - General-purpose model for all other domains
- **Domain Detection**: Automatic detection and model switching

### Future Features
- XnX string parser/export for easy CLI debugging
- Edge permanence category filters
- Path confidence scoring
- LLM prompt auto-generation based on XnX queries
- Visualization components for graph exploration

---

## 📚 Usage Reminder:
Always ensure edges in the graph contain these XnX fields:
```json
{
  "weight": 0.85,
  "direction": -1,
  "valid_from": "2023-01-01T00:00:00Z",
  "valid_to": "2027-01-01T00:00:00Z"
}
```

👉 **XnX constraints give you precision and explainability** in knowledge graph traversal.

---

## 📖 Bibliography

### Foundational and ANT References
- Latour, B. (2005). *Reassembling the Social: An Introduction to Actor-Network-Theory.* Oxford University Press.
- Latour, B. (1996). On Actor-Network Theory: A few clarifications. *Soziale Welt*, 47(4), 369–381.
- Law, J. (2009). Actor Network Theory and Material Semiotics. In *The New Blackwell Companion to Social Theory* (pp. 141-158).

### Motivational Literature
- Chen, Y., Wu, X., Li, C., Zhang, X., Zhou, Y., Zeng, M., & Gao, J. (2024). *Deconstructing Long Chain-of-Thought (DLCoT)*. arXiv preprint arXiv:2503.16385. https://arxiv.org/abs/2503.16385
- Von Oswald, J., Shah, P., Akyürek, E., & Szlam, A. (2024). *Transformers Learn to Implement Multi-step Gradient Descent with Chain of Thought.* arXiv preprint arXiv:2502.21212. https://arxiv.org/abs/2502.21212

---

_Last updated: 2025-03-21_