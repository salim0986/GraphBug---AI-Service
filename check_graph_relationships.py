"""
Check Graph Relationships Status
Quick script to see what relationships exist in the Neo4j graph
"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

print("🔍 Checking Neo4j Graph Relationships...\n")

with neo4j_driver.session() as session:
    # Get all relationship types
    query = """
    CALL db.relationshipTypes()
    """
    result = session.run(query)
    rel_types = [record['relationshipType'] for record in result]
    
    print(f"📊 Relationship Types in Database: {len(rel_types)}")
    for rt in rel_types:
        print(f"   • {rt}")
    
    if not rel_types:
        print("   ⚠️  No relationship types found!")
        print("\n💡 This means relationships haven't been created yet.")
        print("   To create them, you need to run build_dependencies() after ingestion.")
    else:
        print("\n📈 Relationship Counts:")
        for rt in rel_types:
            count_query = f"MATCH ()-[r:{rt}]->() RETURN count(r) as count"
            count_result = session.run(count_query)
            count = count_result.single()['count']
            print(f"   • {rt}: {count:,}")

neo4j_driver.close()
