# Keyword Implementation Plan

## Problem Statement
We need to maintain keyword associations throughout the pipeline without losing the connection between keywords and their source content when chunking occurs.

## Proposed Solution: Dual-Level Keywords

### 1. Document-Level Keywords
- **When**: Extract BEFORE chunking
- **From**: Full document content
- **Purpose**: Capture overall themes, main concepts, document purpose
- **Storage**: At document root level in JSON

### 2. Chunk-Level Keywords  
- **When**: Extract AFTER chunking
- **From**: Individual chunk content
- **Purpose**: Precise, granular keywords for specific content
- **Storage**: Within each chunk object

## Implementation Details

### Modified Pipeline Components

#### 1. Update DocProc Stage
Add keyword extraction immediately after content extraction:

```python
# In docproc components
class EnhancedDocProcessor:
    def process(self, file_path: Path) -> Dict:
        # Existing processing
        content = self.extract_content(file_path)
        
        # NEW: Extract document-level keywords
        doc_keywords = self.keyword_extractor.extract(
            content, 
            mode="document",
            max_keywords=20
        )
        
        return {
            "document_id": str(file_path),
            "content": content,
            "document_keywords": doc_keywords,
            "metadata": {...}
        }
```

#### 2. Update Chunking Stage
Preserve document keywords and add chunk keyword extraction:

```python
# In chunking components
class EnhancedChunker:
    def chunk(self, document: Dict) -> Dict:
        # Existing chunking
        chunks = self.create_chunks(document["content"])
        
        # NEW: Extract keywords for each chunk
        for chunk in chunks:
            chunk["chunk_keywords"] = self.keyword_extractor.extract(
                chunk["content"],
                mode="chunk",
                max_keywords=10
            )
        
        # Preserve document-level keywords
        document["chunks"] = chunks
        del document["content"]  # Replace content with chunks
        
        return document
```

#### 3. Update Embedding Stage
Consider keywords when generating embeddings:

```python
# In embedding components
class KeywordAwareEmbedder:
    def embed_chunk(self, chunk: Dict, doc_keywords: List[str]) -> Dict:
        # Combine chunk content with keywords for richer embeddings
        enhanced_text = self.combine_content_and_keywords(
            chunk["content"],
            chunk["chunk_keywords"],
            doc_keywords  # Include document context
        )
        
        chunk["embedding"] = self.model.encode(enhanced_text)
        return chunk
```

## JSON Structure at Each Stage

### After DocProc + Keywords:
```json
{
    "document_id": "src/components/pathrag.py",
    "document_keywords": ["pathrag", "graph", "retrieval", "query"],
    "content": "full file content here...",
    "metadata": {
        "type": "python",
        "size": 5432,
        "path": "src/components/pathrag.py"
    }
}
```

### After Chunking + Chunk Keywords:
```json
{
    "document_id": "src/components/pathrag.py",
    "document_keywords": ["pathrag", "graph", "retrieval", "query"],
    "metadata": {...},
    "chunks": [
        {
            "chunk_id": "chunk_0",
            "content": "class PathRAG:\n    def __init__(self, ...):",
            "chunk_keywords": ["class", "pathrag", "init", "constructor"],
            "metadata": {
                "start_line": 10,
                "end_line": 45,
                "ast_type": "class_definition"
            }
        },
        {
            "chunk_id": "chunk_1", 
            "content": "def query(self, text: str) -> List[Result]:",
            "chunk_keywords": ["query", "method", "search", "results"],
            "metadata": {
                "start_line": 46,
                "end_line": 78,
                "ast_type": "method_definition"
            }
        }
    ]
}
```

## Benefits for ISNE

### 1. Multi-Level Graph Construction
- **Document edges**: Connect documents with similar document_keywords
- **Chunk edges**: Connect chunks with similar chunk_keywords
- **Cross-level edges**: Connect chunks to documents via shared keywords

### 2. Richer Training Signal
```python
# ISNE can learn from multiple relationship types
relationships = [
    ("doc_similarity", doc1, doc2, similarity_score),
    ("chunk_similarity", chunk1, chunk2, similarity_score),
    ("doc_contains_chunk", doc1, chunk1, 1.0),
    ("keyword_bridge", chunk1, chunk2, keyword_overlap_score)
]
```

### 3. Better Retrieval
- Query can match at document level (broad search)
- Then drill down to chunk level (precise retrieval)
- Keywords provide additional semantic signal

## Implementation Priority

1. **Phase 1**: Add document keyword extraction to existing pipeline
2. **Phase 2**: Add chunk keyword extraction after chunking
3. **Phase 3**: Update embedder to be keyword-aware
4. **Phase 4**: Modify ISNE to use multi-level keywords
5. **Phase 5**: Update retrieval to leverage both levels

## Configuration Example

```yaml
# keyword_extraction_config.yaml
document_keywords:
  max_keywords: 20
  min_keyword_length: 3
  include_technical_terms: true
  include_entities: true
  
chunk_keywords:
  max_keywords: 10
  min_keyword_length: 2
  focus_on_local_context: true
  inherit_document_keywords: false  # Don't duplicate
  
embedding:
  use_keywords: true
  keyword_weight: 0.3  # How much to weight keywords vs content
  concatenate_strategy: "prefix"  # How to combine with content
```

This approach maintains keyword associations while providing maximum flexibility for retrieval and graph construction.