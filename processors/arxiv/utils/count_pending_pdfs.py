#!/usr/bin/env python3
"""
Count PDFs that need to be processed with enhanced extraction.
"""

import os
from arango import ArangoClient

def count_pending_pdfs():
    """Count PDFs needing processing"""
    
    password = os.environ.get('ARANGO_PASSWORD')
    if not password:
        raise ValueError("ARANGO_PASSWORD environment variable required")
    
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    db = client.db('academy_store', username='root', password=password)
    
    # Count different categories
    queries = {
        "Total documents": """
            RETURN COUNT(FOR doc IN base_arxiv RETURN 1)
        """,
        "Documents with PDFs": """
            RETURN COUNT(
                FOR doc IN base_arxiv 
                FILTER doc.pdf_local_path != null 
                RETURN 1
            )
        """,
        "PDFs needing processing": """
            RETURN COUNT(
                FOR doc IN base_arxiv
                FILTER doc.pdf_local_path != null
                FILTER doc.pdf_status IN ['downloaded', null, 'converted']
                FILTER doc.full_text == null
                RETURN 1
            )
        """,
        "Already processed (with full_text)": """
            RETURN COUNT(
                FOR doc IN base_arxiv
                FILTER doc.full_text != null
                RETURN 1
            )
        """,
        "Already embedded": """
            RETURN COUNT(
                FOR doc IN base_arxiv
                FILTER doc.embeddings != null
                RETURN 1
            )
        """
    }
    
    print("=== PDF Processing Status ===\n")
    
    for desc, query in queries.items():
        cursor = db.aql.execute(query)
        count = cursor.next()
        print(f"{desc}: {count:,}")
    
    # Estimate processing time
    cursor = db.aql.execute(queries["PDFs needing processing"])
    pending = cursor.next()
    
    # Based on test: ~10.4 seconds per PDF
    avg_time_per_pdf = 10.4
    total_time_seconds = pending * avg_time_per_pdf
    
    # Calculate time estimates for different worker counts
    print("\n=== Time Estimates ===")
    for workers in [15, 20, 30]:
        time_hours = total_time_seconds / (workers * 3600)
        print(f"With {workers} workers: ~{time_hours:.1f} hours")


if __name__ == "__main__":
    count_pending_pdfs()