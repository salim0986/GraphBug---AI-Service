#!/usr/bin/env python3
"""
Check for repo_id mismatches between what's stored and what's being queried.
"""
import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Load environment
from dotenv import load_dotenv
load_dotenv()

def check_repo_ids():
    """Check what repo_ids actually exist in databases."""
    print("=" * 80)
    print("Checking repo_id formats in databases")
    print("=" * 80)
    
    # Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Qdrant connection
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Connect to Qdrant
    if qdrant_api_key:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_client = QdrantClient(url=qdrant_url)
    
    # Check Neo4j
    print("\n📊 NEO4J - Checking all repo_ids:")
    print("-" * 80)
    
    query = """
    MATCH (n)
    WHERE n.repo_id IS NOT NULL
    RETURN DISTINCT n.repo_id as repo_id, 
           labels(n) as labels,
           count(*) as count
    ORDER BY repo_id
    """
    
    try:
        result = driver.execute_query(query)
        records = result.records
        
        if not records:
            print("❌ No nodes with repo_id found in Neo4j")
        else:
            print(f"✅ Found {len(records)} unique repo_id patterns:\n")
            for record in records:
                print(f"   Repo ID: {record['repo_id']}")
                print(f"   Labels:  {record['labels']}")
                print(f"   Count:   {record['count']} nodes")
                print()
    except Exception as e:
        print(f"❌ Error querying Neo4j: {e}")
    
    # Check Qdrant
    print("\n📊 QDRANT - Checking collections and points:")
    print("-" * 80)
    
    try:
        collections = qdrant_client.get_collections()
        print(f"✅ Found {len(collections.collections)} collections:\n")
        
        for coll in collections.collections:
            print(f"   Collection: {coll.name}")
            
            # Try to get a few points to see repo_id format
            try:
                points = qdrant_client.scroll(
                    collection_name=coll.name,
                    limit=5,
                    with_payload=True
                )
                
                if points and points[0]:
                    print(f"   Points: {len(points[0])} samples")
                    
                    # Show unique repo_ids from sample
                    repo_ids = set()
                    for point in points[0]:
                        if point.payload and 'repo_id' in point.payload:
                            repo_ids.add(point.payload['repo_id'])
                    
                    if repo_ids:
                        print(f"   Sample repo_ids found:")
                        for rid in sorted(repo_ids):
                            print(f"      - {rid}")
                    else:
                        print(f"   ⚠️ No repo_id in payloads")
                else:
                    print(f"   ⚠️ No points found")
                    
            except Exception as e:
                print(f"   ❌ Error reading points: {e}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error querying Qdrant: {e}")
    
    # Show what's being queried
    print("\n📊 EXPECTED QUERIES:")
    print("-" * 80)
    print("Based on PR logs, the system is querying with:")
    print("   repo_id = 'mohd-abex/ai-assesser-abex'")
    print()
    print("If the actual repo_ids in the databases are different,")
    print("that's the cause of the 'No entities found' warnings.")
    print("=" * 80)
    
    driver.close()

if __name__ == "__main__":
    check_repo_ids()
