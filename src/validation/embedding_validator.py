"""
Embedding validation utilities for data ingestion.

This module provides validation functions to ensure embedding consistency
during the data ingestion pipeline, particularly when applying ISNE embeddings.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)

def validate_embeddings_before_isne(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate documents before ISNE embedding to ensure data consistency.
    
    Args:
        documents: List of processed documents with base embeddings
        
    Returns:
        Dictionary with validation results
    """
    total_docs = len(documents)
    docs_with_chunks = sum(1 for doc in documents if "chunks" in doc and doc["chunks"])
    total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
    
    # Check for any existing ISNE embeddings (shouldn't be any at this stage)
    existing_isne = sum(1 for doc in documents 
                      for chunk in doc.get("chunks", []) 
                      if "isne_embedding" in chunk)
    
    # Count chunks with base embeddings
    chunks_with_base_embeddings = sum(1 for doc in documents 
                                   for chunk in doc.get("chunks", []) 
                                   if "embedding" in chunk and chunk["embedding"])
    
    # Track chunks without base embeddings
    missing_base_embeddings: List[str] = []
    for doc_idx, doc in enumerate(documents):
        if "chunks" not in doc:
            continue
            
        for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
            if "embedding" not in chunk or not chunk["embedding"]:
                chunk_id = f"{doc.get('file_id', f'doc_{doc_idx}')}_{chunk_idx}"
                missing_base_embeddings.append(chunk_id)
    
    # Log validation results
    logger.info(f"Pre-ISNE Validation: {total_docs} documents, {docs_with_chunks} with chunks, {total_chunks} total chunks")
    logger.info(f"Found {chunks_with_base_embeddings}/{total_chunks} chunks with base embeddings")
    
    if existing_isne > 0:
        logger.warning(f"⚠️ Found {existing_isne} chunks with existing ISNE embeddings before application!")
        
    if chunks_with_base_embeddings < total_chunks:
        logger.warning(f"⚠️ Missing base embeddings in {total_chunks - chunks_with_base_embeddings} chunks")
        if len(missing_base_embeddings) > 0:
            logger.warning(f"  First 5 chunks missing embeddings: {missing_base_embeddings[:5]}")
    
    # Store validation results
    return {
        "total_docs": total_docs,
        "docs_with_chunks": docs_with_chunks,
        "total_chunks": total_chunks,
        "chunks_with_base_embeddings": chunks_with_base_embeddings,
        "existing_isne": existing_isne,
        "missing_base_embeddings": len(missing_base_embeddings),
        "missing_base_embedding_ids": missing_base_embeddings[:10]  # Store first 10 for reference
    }

def validate_embeddings_after_isne(documents: List[Dict[str, Any]], 
                                  pre_validation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate documents after ISNE embedding to ensure data consistency.
    
    Args:
        documents: List of processed documents with ISNE embeddings
        pre_validation: Results from pre-validation for comparison
        
    Returns:
        Dictionary with validation results
    """
    # Count ISNE embeddings
    chunks_with_isne = sum(1 for doc in documents 
                        for chunk in doc.get("chunks", []) 
                        if "isne_embedding" in chunk)
    
    # Count chunks with relationship data
    chunks_with_relationships = sum(1 for doc in documents 
                                 for chunk in doc.get("chunks", []) 
                                 if "relationships" in chunk and chunk["relationships"])
    
    # Count total relationships
    total_relationships = sum(len(chunk.get("relationships", [])) 
                           for doc in documents 
                           for chunk in doc.get("chunks", []))
    
    # Check for chunks without ISNE embeddings
    chunks_missing_isne: List[str] = []
    # Check for chunks without relationships
    chunks_missing_relationships: List[str] = []
    # Check for invalid relationship structures
    chunks_with_invalid_relationships: List[str] = []
    
    for doc_idx, doc in enumerate(documents):
        if "chunks" not in doc:
            continue
            
        for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
            chunk_id = f"{doc.get('file_id', f'doc_{doc_idx}')}_{chunk_idx}"
            
            # Check for missing ISNE embeddings
            if "isne_embedding" not in chunk:
                chunks_missing_isne.append(chunk_id)
            
            # Check for missing relationships
            if "isne_embedding" in chunk and ("relationships" not in chunk or not chunk["relationships"]):
                chunks_missing_relationships.append(chunk_id)
            
            # Validate relationship structure
            if "relationships" in chunk and chunk["relationships"]:
                for rel_idx, rel in enumerate(chunk["relationships"]):
                    if not all(key in rel for key in ["source", "target", "type", "weight"]):
                        if chunk_id not in chunks_with_invalid_relationships:
                            chunks_with_invalid_relationships.append(chunk_id)
    
    # Look for any non-chunk ISNE embeddings that might be in document metadata
    doc_level_isne = sum(1 for doc in documents if "isne_embedding" in doc)
    
    # Count total ISNE embeddings (to detect duplicates)
    total_isne_count = sum(1 for doc in documents 
                         for chunk in doc.get("chunks", []) 
                         for _ in [1] if "isne_embedding" in chunk)
    
    # Check for duplicate ISNE embeddings (same chunk with multiple embeddings)
    duplicate_chunks: Set[str] = set()
    for doc_idx, doc in enumerate(documents):
        if "chunks" not in doc:
            continue
            
        for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
            # Look for isne_embedding keys with list values
            isne_keys = [k for k in chunk.keys() if k.startswith("isne_embedding")]
            if len(isne_keys) > 1:
                chunk_id = f"{doc.get('file_id', f'doc_{doc_idx}')}_{chunk_idx}"
                duplicate_chunks.add(chunk_id)
    
    # Log validation results
    logger.info(f"Post-ISNE Validation: {chunks_with_isne}/{pre_validation['total_chunks']} chunks have ISNE embeddings")
    logger.info(f"Relationship Data: {chunks_with_relationships} chunks have relationships, {total_relationships} total relationships")
    
    # Check for discrepancies
    if chunks_with_isne != pre_validation['total_chunks']:
        logger.warning(f"⚠️ Discrepancy detected: {chunks_with_isne} chunks with ISNE vs {pre_validation['total_chunks']} total chunks")
        if len(chunks_missing_isne) > 0:
            logger.warning(f"⚠️ Found {len(chunks_missing_isne)} chunks without ISNE embeddings")
            logger.warning(f"  First 5 chunks missing ISNE: {chunks_missing_isne[:5]}")
    
    if chunks_with_relationships < chunks_with_isne:
        logger.warning(f"⚠️ Discrepancy detected: {chunks_with_relationships} chunks with relationships vs {chunks_with_isne} with ISNE embeddings")
        if len(chunks_missing_relationships) > 0:
            logger.warning(f"⚠️ Found {len(chunks_missing_relationships)} chunks with ISNE but no relationships")
            logger.warning(f"  First 5 chunks missing relationships: {chunks_missing_relationships[:5]}")
    
    if chunks_with_invalid_relationships:
        logger.warning(f"⚠️ Found {len(chunks_with_invalid_relationships)} chunks with invalid relationship structures")
        logger.warning(f"  First 5 chunks with invalid relationships: {chunks_with_invalid_relationships[:5]}")
    
    if total_isne_count > chunks_with_isne:
        logger.warning(f"⚠️ Found {total_isne_count - chunks_with_isne} duplicate ISNE embeddings")
        if duplicate_chunks:
            logger.warning(f"  First 5 chunks with duplicate embeddings: {list(duplicate_chunks)[:5]}")
            
    if doc_level_isne > 0:
        logger.warning(f"⚠️ Found {doc_level_isne} document-level ISNE embeddings (outside of chunks)")
        
    if total_isne_count != pre_validation['total_chunks']:
        logger.warning(f"⚠️ Total ISNE count ({total_isne_count}) does not match chunk count ({pre_validation['total_chunks']})")
    
    # Return validation results
    return {
        "chunks_with_isne": chunks_with_isne,
        "chunks_missing_isne": len(chunks_missing_isne),
        "chunks_missing_isne_ids": chunks_missing_isne[:10],  # Store first 10 for reference
        "chunks_with_relationships": chunks_with_relationships,
        "chunks_missing_relationships": len(chunks_missing_relationships),
        "chunks_missing_relationship_ids": chunks_missing_relationships[:10],
        "chunks_with_invalid_relationships": len(chunks_with_invalid_relationships),
        "chunks_with_invalid_relationship_ids": chunks_with_invalid_relationships[:10],
        "total_relationships": total_relationships,
        "doc_level_isne": doc_level_isne,
        "total_isne_count": total_isne_count,
        "duplicate_isne": total_isne_count - chunks_with_isne if total_isne_count > chunks_with_isne else 0,
        "duplicate_chunk_ids": list(duplicate_chunks)[:10]  # Store first 10 for reference
    }

def create_validation_summary(pre_validation: Dict[str, Any], 
                             post_validation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive validation summary comparing pre and post ISNE embedding states.
    
    Args:
        pre_validation: Results from validation before ISNE embedding
        post_validation: Results from validation after ISNE embedding
        
    Returns:
        Dictionary with validation summary
    """
    return {
        "pre_validation": pre_validation,
        "post_validation": post_validation,
        "discrepancies": {
            "isne_vs_chunks": post_validation["total_isne_count"] - pre_validation["total_chunks"],
            "missing_isne": post_validation["chunks_missing_isne"],
            "doc_level_isne": post_validation["doc_level_isne"],
            "duplicate_isne": post_validation["duplicate_isne"]
        }
    }

def attach_validation_summary(documents: List[Dict[str, Any]], 
                             validation_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Attach validation summary to documents list as a special attribute.
    
    This allows the validation summary to be available for reporting
    without being serialized to JSON.
    
    Args:
        documents: List of processed documents
        validation_summary: Validation results to attach
        
    Returns:
        Documents list with attached validation_summary attribute
    """
    # Since we can't directly set attributes on a list, we'll use a custom class
    # to wrap the list and add our attribute
    class DocumentList(list):
        pass
    
    # Create a new DocumentList with the same items as documents
    doc_list = DocumentList(documents)
    
    # Set the validation_summary attribute on the new list
    setattr(doc_list, "validation_summary", validation_summary)
    
    return doc_list
