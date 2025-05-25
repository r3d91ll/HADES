# ArangoDB Setup for HADES-PathRAG

This document outlines the ArangoDB database structure for the HADES-PathRAG system, with specific focus on storing and querying document data enhanced with ISNE embeddings.

## Overview

ArangoDB is a multi-model database that combines document, graph, and key-value storage models. This flexibility makes it ideal for our PathRAG system where we need to:

1. Store document content and metadata (document model)
2. Represent relationships between chunks and documents (graph model)
3. Perform vector similarity searches on embeddings (vector indexes)

## Database Structure

### Collections

In ArangoDB, we'll use both **document collections** (for storing nodes) and **edge collections** (for relationships).

#### Document Collections

1. **Documents**
   - Purpose: Store metadata about each document
   - Structure:
  
     ```json
     {
       "_key": "pdf_4588_CG-RAG_Research...",  // Unique identifier
       "file_name": "CG-RAG_Research_Question_Answering.pdf",
       "file_size": 958034,
       "file_type": "pdf",
       "processing_metadata": {
         "worker_id": 0,
         "completed_at": "2025-05-22T07:38:15.555Z",
         "timing": { ... }
       }
     }
     ```

2. **Chunks**
   - Purpose: Store document chunks with their text content and embeddings
   - Structure:

     ```json
     {
       "_key": "pdf_c0a7a004_CG-RAG__Research_..._p0",  // Unique identifier
       "parent_id": "pdf_4588_CG-RAG_Research...",  // Reference to parent document
       "content": "## CG-RAG: Research Question Answering by...",  // Full text content
       "content_hash": "e9c5bb02c9e3793228002cf43db5f443",
       "embedding": [0.0051719, -0.0032053, ...],  // Base embedding vector
       "isne_embedding": [0.0226900, -0.0116553, ...],  // ISNE-enhanced embedding
       "metadata": {
         "type": "academic_pdf",
         "chunk_index": 0,
         "start_offset": 0,
         "end_offset": 4702,
         "line_start": 0,
         "line_end": 0,
         "token_count": 612,
         "symbol_type": "paragraph"
       }
     }
     ```

#### Edge Collections

1. **ChunkRelations**
   - Purpose: Store relationships between chunks
   - Types of relationships:
     - **sequential**: Natural ordering within a document
     - **semantic**: Content-based relationships
     - **citation**: Cross-references between chunks
   - Structure:

     ```json
     {
       "_from": "Chunks/pdf_chunk1_id",  // Source chunk
       "_to": "Chunks/pdf_chunk2_id",    // Target chunk
       "type": "sequential",             // Relationship type
       "weight": 1.0,                    // Relationship strength
       "metadata": {                     // Optional metadata
         "distance": 0.23                // e.g., cosine distance between embeddings
       }
     }
     ```

2. **DocumentRelations**
   - Purpose: Store relationships between documents
   - Structure similar to ChunkRelations but connects Documents instead of Chunks

### Graph Definition

We'll define a named graph in ArangoDB to encapsulate our collections:

```javascript
db._createGraph("PathRAG", [
  { collection: "ChunkRelations", from: ["Chunks"], to: ["Chunks"] },
  { collection: "DocumentRelations", from: ["Documents"], to: ["Documents"] }
]);
```

This defines a graph called "PathRAG" that includes our edge collections and specifies which document collections they connect.

## Indexes

To ensure efficient queries, we'll create several indexes:

### Standard Indexes

```javascript
// For faster lookups by parent document
db.Chunks.ensureIndex({ type: "persistent", fields: ["parent_id"] });

// For full-text search on content
db.Chunks.ensureIndex({ type: "fulltext", fields: ["content"] });
```

### Vector Indexes

```javascript
// For base embedding similarity search
db.Chunks.ensureIndex({ 
  type: "inverted", 
  fields: ["embedding[*]"],
  analyzer: "vector",
  vectorOptions: {
    dimensions: 768,    // Dimension of your embedding vectors
    centroidDimensions: 16,
    maxDistance: 1.0,
    minSimilarity: 0.0,
    distanceFunction: "cosine"
  }
});

// For ISNE embedding similarity search
db.Chunks.ensureIndex({ 
  type: "inverted", 
  fields: ["isne_embedding[*]"],
  analyzer: "vector",
  vectorOptions: {
    dimensions: 768,
    centroidDimensions: 16,
    maxDistance: 1.0,
    minSimilarity: 0.0,
    distanceFunction: "cosine"
  }
});
```

## Key Differences from SQL

For those familiar with SQL databases, here are some key differences:

1. **No Schema Enforcement**: Collections don't have a fixed schema like SQL tables
2. **Document-Based**: Each record is a JSON document that can have its own structure
3. **Graph Traversal**: You can directly traverse relationships without JOINs
4. **Edge Collections**: Relationships are first-class citizens with their own properties
5. **Vector Search**: Native support for embedding vector similarity search

## Database Operations

### Setting Up the Database

```javascript
// Create a new database
db._createDatabase("hades_pathrag");
db._useDatabase("hades_pathrag");

// Create document collections
db._create("Documents");
db._create("Chunks");

// Create edge collections
db._create("ChunkRelations", { type: 2 });  // type 2 means edge collection
db._create("DocumentRelations", { type: 2 });

// Create graph
db._createGraph("PathRAG", [
  { collection: "ChunkRelations", from: ["Chunks"], to: ["Chunks"] },
  { collection: "DocumentRelations", from: ["Documents"], to: ["Documents"] }
]);

// Create indexes (as shown in the Indexes section)
```

### Loading Data from JSON

To load our ISNE-enhanced JSON data into ArangoDB:

1. First, insert document metadata:

   ```javascript
   for (const doc of jsonData) {
     const documentData = {
       _key: doc.file_id,
       file_name: doc.file_name,
       file_size: doc.file_size,
       file_type: doc.file_name.split('.').pop(),
       processing_metadata: {
         worker_id: doc.worker_id,
         completed_at: doc.completed_at,
         timing: doc.timing
       }
     };
     db.Documents.insert(documentData);
   }
   ```

2. Then, insert chunks with their embeddings:

   ```javascript
   for (const doc of jsonData) {
     for (const chunk of doc.chunks) {
       const chunkData = {
         _key: chunk.id,
         parent_id: doc.file_id,
         content: chunk.content,
         content_hash: chunk.content_hash,
         embedding: chunk.embedding,
         isne_embedding: chunk.isne_embedding,
         metadata: {
           type: chunk.type,
           chunk_index: chunk.chunk_index,
           start_offset: chunk.start_offset,
           end_offset: chunk.end_offset,
           line_start: chunk.line_start,
           line_end: chunk.line_end,
           token_count: chunk.token_count,
           symbol_type: chunk.symbol_type
         }
       };
       db.Chunks.insert(chunkData);
     }
   }
   ```

3. Finally, create relationships between chunks:

   ```javascript
   // Create sequential relationships
   for (const doc of jsonData) {
     for (let i = 0; i < doc.chunks.length - 1; i++) {
       const currentChunk = doc.chunks[i];
       const nextChunk = doc.chunks[i + 1];
       
       const edge = {
         _from: `Chunks/${currentChunk.id}`,
         _to: `Chunks/${nextChunk.id}`,
         type: "sequential",
         weight: 1.0
       };
       
       db.ChunkRelations.insert(edge);
     }
   }
   
   // Create semantic relationships based on embedding similarity
   // This would typically be done by comparing embeddings and
   // creating edges between chunks with similarity above a threshold
   ```

## Query Examples

### Basic Document Retrieval

```aql
// Get a document by ID
RETURN DOCUMENT("Documents/pdf_4588_CG-RAG_Research...")

// Get all chunks for a document
FOR chunk IN Chunks
  FILTER chunk.parent_id == "pdf_4588_CG-RAG_Research..."
  RETURN chunk
```

### Graph Traversal

```aql
// Follow sequential chunk relationships (up to 3 hops)
FOR v, e IN 1..3 OUTBOUND 'Chunks/pdf_chunk1_id' ChunkRelations
  FILTER e.type == "sequential"
  RETURN { chunk: v, relation: e }

// Find paths between two chunks
FOR path IN ANY SHORTEST_PATH 'Chunks/pdf_chunk1_id' TO 'Chunks/pdf_chunk2_id' 
  ChunkRelations
  RETURN path
```

### Vector Similarity Search

```aql
// Search by base embedding
FOR c IN Chunks
  SEARCH ANALYZER(
    VECTOR_DISTANCE(c.embedding, @query_vector) < 0.2,
    "vector"
  )
  SORT VECTOR_DISTANCE(c.embedding, @query_vector)
  LIMIT 10
  RETURN c

// Search by ISNE embedding (typically provides better semantic matches)
FOR c IN Chunks
  SEARCH ANALYZER(
    VECTOR_DISTANCE(c.isne_embedding, @query_vector) < 0.2,
    "vector"
  )
  SORT VECTOR_DISTANCE(c.isne_embedding, @query_vector)
  LIMIT 10
  RETURN c
```

### Hybrid Search (Text + Vector)

```aql
// Combine full-text and vector search
FOR c IN Chunks
  SEARCH ANALYZER(PHRASE(c.content, "citation graph"), "text") AND
         ANALYZER(VECTOR_DISTANCE(c.isne_embedding, @query_vector) < 0.3, "vector")
  SORT VECTOR_DISTANCE(c.isne_embedding, @query_vector)
  LIMIT 5
  RETURN c
```

## Python Integration

Here's how we'd integrate this with our Python code:

```python
from arango import ArangoClient

# Initialize the client
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("hades_pathrag", username="root", password="")

# Insert a document
document_data = {
    "_key": doc["file_id"],
    "file_name": doc["file_name"],
    "file_size": doc["file_size"],
    # other fields...
}
db.collection("Documents").insert(document_data)

# Insert a chunk
chunk_data = {
    "_key": chunk["id"],
    "parent_id": doc["file_id"],
    "content": chunk["content"],
    "embedding": chunk["embedding"],
    "isne_embedding": chunk["isne_embedding"],
    # other fields...
}
db.collection("Chunks").insert(chunk_data)

# Create a relationship
edge_data = {
    "_from": f"Chunks/{chunk1_id}",
    "_to": f"Chunks/{chunk2_id}",
    "type": "sequential",
    "weight": 1.0
}
db.collection("ChunkRelations").insert(edge_data)

# Perform a vector search
query = """
FOR c IN Chunks
  SEARCH ANALYZER(
    VECTOR_DISTANCE(c.isne_embedding, @query_vector) < 0.2,
    "vector"
  )
  SORT VECTOR_DISTANCE(c.isne_embedding, @query_vector)
  LIMIT 10
  RETURN c
"""
cursor = db.aql.execute(query, bind_vars={"query_vector": query_vector})
results = [doc for doc in cursor]
```

## Conclusion

This setup provides a flexible, powerful foundation for the HADES-PathRAG system. ArangoDB's multi-model approach allows us to:

1. Store document content and embeddings efficiently
2. Represent complex relationships between content chunks
3. Perform fast vector similarity searches
4. Combine graph traversals with semantic searches

Both the base embeddings and ISNE-enhanced embeddings are valuable:

- Base embeddings provide a foundational semantic representation
- ISNE embeddings incorporate structural information that improves retrieval relevance
- Storing both gives us flexibility to use either depending on the specific query needs

As you develop the system further, this database design can be extended to include additional features like user feedback, specialized domain collections, or more complex relationship types.
