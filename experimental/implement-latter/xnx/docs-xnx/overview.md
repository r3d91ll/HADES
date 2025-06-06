# XnX Notation: Overview

## Introduction

XnX Notation is a human-readable shorthand system designed for expressing and controlling relationship characteristics in graph-based knowledge systems. In the context of HADES, XnX Notation provides a framework for weighted path tuning within our PathRAG implementation, enabling more precise control over graph traversal, improved fact validation, and enhanced knowledge retrieval.

## Core Concept

The XnX Notation follows the format:

```
w x d
```

Where:

- **w (Weight)**: A decimal value (0.0 to 1.0) representing the strength, confidence, or relevance of a relationship.
- **x (Node Identifier)**: A label identifying the target node of the relationship.
- **d (Distance/Flow)**: A signed integer indicating both the distance (or hops) and direction of data flow.
  - Positive `d` implies **inbound/ingress** flow toward the node
  - Negative `d` implies **outbound/egress** flow away from the node

### Example

```
0.92 FunctionA -1
```

This represents:
- A relationship with 92% confidence/strength
- To a node called "FunctionA"
- With a flow direction outward (egress) of 1 hop

## Temporal XnX: Tracking Relationships Over Time

### Extended Notation for Temporal Dimensions

The base XnX format can be extended to include temporal dimensions, creating a notation system that can track how relationships evolve over time:

```
w x d[t1→t2]
```

Where:
- **w, x, d**: Retain their meanings from the base notation
- **[t1→t2]**: Represents the time interval during which this relationship weight applies
  - Time intervals can be absolute dates or relative time references
  - Special timestamps can include version identifiers (e.g., Linux kernel versions)

### Permanence Categories

Temporal XnX can include permanence indicators:

```
w(p) x d
```

Where:
- **p (Permanence)**: Indicates relationship permanence category:
  - **∞**: Permanent, immutable relationships
  - **L**: Long-term relationships (stable but could theoretically change)
  - **C**: Contextual relationships (valid within specific contexts)
  - **T[date]**: Time-bound with explicit expiration date

### Examples

```
0.95(∞) SSN -1                               # Permanent relationship to SSN
0.85(T[2027-03-22]) ContractData -1          # Relationship expires on specific date
0.70(C) MaritalStatus -1                     # Contextual relationship
0.65 kernel.scheduler -1[Linux_3.1→Linux_6.4] # Cross-version relationship in kernel
```

## Integration with HADES PathRAG

### Current PathRAG Implementation

HADES currently implements PathRAG (Path-Augmented Retrieval-Augmented Generation) to enhance knowledge retrieval by considering paths within the knowledge graph. The existing implementation:

1. Stores entities and relationships in ArangoDB
2. Provides MCP tools for data ingestion and retrieval (`mcp0_ingest_data`, `mcp0_pathrag_retrieve`)
3. Uses an entity-relationship model with explicit from/to fields and relation types

### How XnX Enhances PathRAG

XnX Notation extends our PathRAG implementation by:

1. **Adding Path Weights and Direction Control**
   - Enabling filtering based on relationship confidence/strength
   - Controlling traversal direction based on flow indicators
   - Limiting traversal depth based on distance metrics

2. **Improving Retrieval Precision**
   - Prioritizing high-confidence paths
   - Excluding irrelevant relationships based on direction
   - Reducing computational overhead by limiting path exploration

3. **Enabling Fact Verification**
   - Using path consistency to validate statements
   - Comparing multiple paths to establish confidence
   - Identifying contradictory information

## Core Applications

XnX notation enables several powerful applications in the HADES system:

1. **Knowledge Graph Traversal**: Control and optimize traversal through complex knowledge graphs
2. **Fact Verification**: Validate statements by considering path weights and consistency
3. **Temporal Knowledge Tracking**: Monitor how relationships evolve over time
4. **Access Control**: Manage permissions based on relationship paths and strengths
5. **Knowledge Transfer Tracking**: Monitor how information flows through systems and organizations
6. **Version Comparison**: Compare software components across different versions

## Next Steps

For implementation details, see the [Implementation Guide](./implementation.md), which covers the technical aspects of integrating XnX notation into the HADES-PathRAG system.
