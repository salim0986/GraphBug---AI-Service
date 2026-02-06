#!/usr/bin/env python3
"""
Check if a repository has been ingested (has data in Neo4j and Qdrant)
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def check_repo_ingestion(repo_id: str):
    """Check if repo has been ingested"""
    from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, QDRANT_API_KEY
    from src.graph_builder import GraphBuilder
    from src.vector_builder import VectorBuilder
    from sentence_transformers import SentenceTransformer
    
    print("=" * 70)
    print(f"Checking ingestion status for repo: {repo_id}")
    print("=" * 70)
    
    try:
        # Initialize connections
        print("\n1. Connecting to Neo4j...")
        graph_db = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
        
        print("2. Connecting to Qdrant...")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        vector_db = VectorBuilder(QDRANT_URL, embed_model, api_key=QDRANT_API_KEY)
        
        # Check Neo4j
        print(f"\n3. Checking Neo4j for repo_id: {repo_id}")
        query = """
        MATCH (f:File {repo_id: $repo_id})
        RETURN count(f) as file_count
        """
        result = graph_db.driver.execute_query(query, repo_id=repo_id)
        file_count = result.records[0]["file_count"] if result.records else 0
        
        query2 = """
        MATCH (n {repo_id: $repo_id})
        WHERE n:Function OR n:Class OR n:Method
        RETURN count(n) as node_count
        """
        result2 = graph_db.driver.execute_query(query2, repo_id=repo_id)
        node_count = result2.records[0]["node_count"] if result2.records else 0
        
        print(f"   ✓ Files: {file_count}")
        print(f"   ✓ Nodes (functions/classes): {node_count}")
        
        # Check Qdrant
        print(f"\n4. Checking Qdrant for repo_id: {repo_id}")
        try:
            # Count vectors with this repo_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            scroll_result = vector_db.client.scroll(
                collection_name="code_embeddings",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            # Get total count
            count_result = vector_db.client.count(
                collection_name="code_embeddings",
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                )
            )
            
            vector_count = count_result.count
            print(f"   ✓ Vectors: {vector_count}")
            
        except Exception as e:
            print(f"   ✗ Error checking Qdrant: {e}")
            vector_count = 0
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if file_count > 0 and node_count > 0 and vector_count > 0:
            print(f"✅ Repository IS INGESTED")
            print(f"   - {file_count} files")
            print(f"   - {node_count} code nodes")
            print(f"   - {vector_count} vectors")
            return True
        else:
            print(f"❌ Repository NOT INGESTED (or partially ingested)")
            print(f"   - Files: {file_count}")
            print(f"   - Nodes: {node_count}")
            print(f"   - Vectors: {vector_count}")
            print(f"\n💡 Solution: Run POST /ingest with this repo_id")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Check the repo from the logs
    repo_id = "6436b68e-20f5-4327-b892-0d6147affeea"
    
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    
    check_repo_ingestion(repo_id)
