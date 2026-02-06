#!/usr/bin/env python3
"""
Quick Check Script - Fast verification of ingestion data

Usage:
    python quick_check_ingestion.py <repo_id>
"""

import sys
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, QDRANT_API_KEY
from src.graph_builder import GraphBuilder
from src.vector_builder import VectorBuilder
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

def check_neo4j(repo_id):
    """Quick check of Neo4j data"""
    print(f"\n{'='*60}")
    print("🗄️  CHECKING NEO4J")
    print('='*60)
    
    try:
        graph_db = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
        
        with graph_db.driver.session() as session:
            # Count nodes
            result = session.run(
                "MATCH (n) WHERE n.repo_id = $repo_id RETURN count(n) as count",
                repo_id=repo_id
            )
            node_count = result.single()["count"]
            
            # Count relationships
            result = session.run(
                "MATCH (a)-[r]->(b) WHERE a.repo_id = $repo_id RETURN count(r) as count",
                repo_id=repo_id
            )
            rel_count = result.single()["count"]
            
            # Get sample files
            result = session.run(
                """
                MATCH (n) WHERE n.repo_id = $repo_id AND n.file IS NOT NULL
                RETURN DISTINCT n.file as file
                LIMIT 5
                """,
                repo_id=repo_id
            )
            files = [r["file"] for r in result]
        
        if node_count == 0:
            print("❌ NO DATA FOUND")
            print(f"   No nodes found for repo_id='{repo_id}'")
            print("\n💡 Action Required:")
            print("   Run ingestion via POST /ingest endpoint")
            return False
        else:
            print(f"✅ DATA EXISTS")
            print(f"   Nodes: {node_count}")
            print(f"   Relationships: {rel_count}")
            print(f"   Sample files:")
            for f in files:
                print(f"     - {f}")
            return True
            
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        return False

def check_qdrant(repo_id):
    """Quick check of Qdrant data"""
    print(f"\n{'='*60}")
    print("🔍 CHECKING QDRANT")
    print('='*60)
    
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        vector_db = VectorBuilder(QDRANT_URL, embed_model, api_key=QDRANT_API_KEY)
        
        # Check collection exists
        if not vector_db.client.collection_exists("repo_code"):
            print("❌ COLLECTION NOT FOUND")
            print("   Collection 'repo_code' does not exist")
            return False
        
        # Count vectors
        collection_info = vector_db.client.get_collection("repo_code")
        total = collection_info.points_count
        
        # Count vectors for this repo
        dummy_query = embed_model.encode("test").tolist()
        result = vector_db.client.query_points(
            collection_name="repo_code",
            query=dummy_query,
            query_filter=Filter(
                must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
            ),
            limit=10
        )
        repo_count = len(result.points)
        
        # Get sample
        samples = []
        for point in result.points[:3]:
            samples.append({
                "name": point.payload.get("name"),
                "file": point.payload.get("file")
            })
        
        if repo_count == 0:
            print("❌ NO VECTORS FOUND")
            print(f"   Total vectors in collection: {total}")
            print(f"   Vectors for repo_id='{repo_id}': 0")
            print("\n💡 Action Required:")
            print("   Run ingestion via POST /ingest endpoint")
            return False
        else:
            print(f"✅ VECTORS EXIST")
            print(f"   Total in collection: {total}")
            print(f"   For this repo: {repo_count}")
            print(f"   Sample vectors:")
            for s in samples:
                print(f"     - {s['name']} ({s['file']})")
            return True
            
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_check_ingestion.py <repo_id>")
        print("\nExample:")
        print("  python quick_check_ingestion.py my-test-repo")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    
    print("\n" + "="*60)
    print(f"🔍 QUICK INGESTION CHECK: {repo_id}")
    print("="*60)
    
    neo4j_ok = check_neo4j(repo_id)
    qdrant_ok = check_qdrant(repo_id)
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if neo4j_ok and qdrant_ok:
        print("✅ INGESTION SUCCESSFUL")
        print("   Both Neo4j and Qdrant have data")
        print("\n💡 Next Step:")
        print("   If context retrieval is still failing, run:")
        print(f"   python diagnose_context_retrieval.py {repo_id}")
    elif neo4j_ok or qdrant_ok:
        print("⚠️  PARTIAL INGESTION")
        print(f"   Neo4j: {'✅' if neo4j_ok else '❌'}")
        print(f"   Qdrant: {'✅' if qdrant_ok else '❌'}")
        print("\n💡 Action Required:")
        print("   Re-run ingestion to fix incomplete data")
    else:
        print("❌ INGESTION FAILED")
        print("   No data found in databases")
        print("\n💡 Action Required:")
        print("   Run ingestion via:")
        print("   POST http://localhost:8000/ingest")
        print("   {")
        print(f'     "repo_url": "https://github.com/owner/repo.git",')
        print(f'     "repo_id": "{repo_id}",')
        print('     "installation_id": "YOUR_INSTALLATION_ID"')
        print("   }")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
