"""
Quick Test: Repo Isolation & Context Richness
Focuses on answering two key questions:
1. Do repos' contexts intersect with each other?
2. Can our app provide rich context to the AI reviewer?
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Colors
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    B = '\033[94m'  # Blue
    Y = '\033[93m'  # Yellow
    C = '\033[96m'  # Cyan
    BOLD = '\033[1m'
    END = '\033[0m'

def header(text):
    print(f"\n{C.C}{C.BOLD}{'=' * 80}{C.END}")
    print(f"{C.C}{C.BOLD}{text}{C.END}")
    print(f"{C.C}{C.BOLD}{'=' * 80}{C.END}\n")

# Initialize clients
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# Get repos
print(f"{C.B}🔍 Discovering repositories...{C.END}")
scroll_result = qdrant.scroll(collection_name="repo_code", limit=100, with_payload=True, with_vectors=False)
repo_ids = list(set(p.payload['repo_id'] for p in scroll_result[0] if 'repo_id' in p.payload))
print(f"{C.G}✅ Found {len(repo_ids)} repositories{C.END}\n")

# Show first 3 repos
for i, repo_id in enumerate(repo_ids[:3], 1):
    print(f"{i}. {repo_id[:20]}...")

# ============================================================================
# QUESTION 1: Do repos' contexts intersect? (Isolation Test)
# ============================================================================
header("QUESTION 1: Do Repository Contexts Intersect?")

print(f"{C.B}Testing vector search isolation across repos...{C.END}\n")

test_query = "authentication and login logic"
query_vector = model.encode(test_query).tolist()

isolation_passed = True
for i, repo_id in enumerate(repo_ids[:3], 1):
    result = qdrant.query_points(
        collection_name="repo_code",
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]),
        limit=3
    ).points
    
    # Check if all results belong to correct repo
    leaked = [p for p in result if p.payload.get('repo_id') != repo_id]
    
    if leaked:
        print(f"{C.R}❌ Repo {i}: LEAKED! Found {len(leaked)} results from other repos{C.END}")
        isolation_passed = False
    else:
        print(f"{C.G}✅ Repo {i}: Perfect isolation ({len(result)} results, all from correct repo){C.END}")

print(f"\n{C.B}Testing graph search isolation...{C.END}\n")

with neo4j_driver.session() as session:
    for i, repo_id in enumerate(repo_ids[:3], 1):
        # Check for cross-repo relationships
        query = """
        MATCH (e1:Entity {repo_id: $repo_id})-[r]->(e2:Entity)
        WHERE e2.repo_id <> $repo_id
        RETURN count(r) as leaks
        """
        result = session.run(query, repo_id=repo_id)
        record = result.single()
        leaks = record['leaks'] if record else 0
        
        if leaks > 0:
            print(f"{C.R}❌ Repo {i}: Graph LEAKED! {leaks} cross-repo relationships{C.END}")
            isolation_passed = False
        else:
            print(f"{C.G}✅ Repo {i}: Graph isolated (no cross-repo relationships){C.END}")

header(f"ANSWER 1: {'NO ✅' if isolation_passed else 'YES ❌'} - Repos {'DO NOT' if isolation_passed else 'DO'} intersect")

if isolation_passed:
    print(f"{C.G}🎉 Perfect isolation! Each repository's context is completely separate.{C.END}")
else:
    print(f"{C.R}⚠️  Isolation breach detected! Repositories are leaking into each other.{C.END}")

# ============================================================================
# QUESTION 2: Can we provide rich context to AI reviewer?
# ============================================================================
header("QUESTION 2: Can We Provide Rich Context?")

test_repo = repo_ids[0]
print(f"{C.B}Testing with repo: {test_repo[:30]}...{C.END}\n")

# Test 1: Semantic search richness
print(f"{C.BOLD}Test 1: Semantic Search Context{C.END}")
queries = [
    "authentication logic",
    "database queries",
    "error handling"
]

total_results = 0
total_code_length = 0

for query_text in queries:
    query_vector = model.encode(query_text).tolist()
    results = qdrant.query_points(
        collection_name="repo_code",
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=test_repo))]),
        limit=3
    ).points
    
    if results:
        print(f"\n  Query: '{query_text}'")
        for r in results:
            code_len = len(r.payload.get('raw_code', ''))
            total_code_length += code_len
            total_results += 1
            print(f"    • {r.payload.get('name', 'unknown')} ({code_len} chars, score: {r.score:.3f})")

avg_code_length = total_code_length / total_results if total_results > 0 else 0
print(f"\n  {C.G}✓ Found {total_results} relevant code snippets{C.END}")
print(f"  {C.G}✓ Average code length: {avg_code_length:.0f} characters{C.END}")

semantic_rich = total_results >= 6 and avg_code_length > 200

# Test 2: Graph relationships
print(f"\n{C.BOLD}Test 2: Graph Relationship Context{C.END}\n")

with neo4j_driver.session() as session:
    # Find functions with dependencies
    query = """
    MATCH (f:Entity {repo_id: $repo_id})
    WHERE f.name IS NOT NULL
    OPTIONAL MATCH (f)-[r:CALLS|CONTAINS]->(target:Entity)
    WITH f, count(r) as rel_count, collect(target.name)[..3] as targets
    WHERE rel_count > 0
    RETURN f.name as name, f.file as file, rel_count, targets
    LIMIT 5
    """
    result = session.run(query, repo_id=test_repo)
    records = list(result)
    
    total_relationships = sum(r['rel_count'] for r in records)
    
    print(f"  Sample functions with relationships:")
    for r in records[:3]:
        print(f"    • {r['name']} → {r['rel_count']} connections")
        if r['targets']:
            print(f"      Calls: {', '.join([t for t in r['targets'] if t])}")
    
    print(f"\n  {C.G}✓ Found {len(records)} entities with relationships{C.END}")
    print(f"  {C.G}✓ Total relationships: {total_relationships}{C.END}")
    
    graph_rich = len(records) >= 3 and total_relationships >= 5

# Test 3: File-level context
print(f"\n{C.BOLD}Test 3: File-Level Context{C.END}\n")

with neo4j_driver.session() as session:
    query = """
    MATCH (f:File {repo_id: $repo_id})
    WHERE f.path IS NOT NULL
    OPTIONAL MATCH (f)-[:IMPORTS]->(dep:File)
    RETURN f.path as file, count(dep) as deps, collect(dep.path)[..3] as dep_files
    ORDER BY deps DESC
    LIMIT 5
    """
    result = session.run(query, repo_id=test_repo)
    records = list(result)
    
    files_with_deps = [r for r in records if r['deps'] > 0]
    
    print(f"  Sample files with dependencies:")
    for r in files_with_deps[:3]:
        print(f"    • {r['file'].split('/')[-1]} → imports {r['deps']} files")
    
    print(f"\n  {C.G}✓ Found {len(records)} files{C.END}")
    print(f"  {C.G}✓ Files with dependencies: {len(files_with_deps)}{C.END}")
    
    file_rich = len(files_with_deps) >= 2

# Overall richness
richness_passed = semantic_rich and graph_rich

header(f"ANSWER 2: {'YES ✅' if richness_passed else 'PARTIALLY ⚠️'} - We {'CAN' if richness_passed else 'CAN SOMEWHAT'} provide rich context")

print(f"  • Semantic Search: {C.G}Rich ✓{C.END}" if semantic_rich else f"  • Semantic Search: {C.Y}Limited{C.END}")
print(f"  • Graph Relationships: {C.G}Rich ✓{C.END}" if graph_rich else f"  • Graph Relationships: {C.Y}Limited{C.END}")
print(f"  • File Dependencies: {C.G}Rich ✓{C.END}" if file_rich else f"  • File Dependencies: {C.Y}Limited{C.END}")

if richness_passed:
    print(f"\n{C.G}🎉 Excellent! Your system provides comprehensive context for AI reviews.{C.END}")
    print(f"{C.G}   The AI reviewer will have access to:{C.END}")
    print(f"{C.G}   - Similar code patterns from semantic search{C.END}")
    print(f"{C.G}   - Function call chains and dependencies from graph{C.END}")
    print(f"{C.G}   - File-level import relationships{C.END}")
else:
    print(f"\n{C.Y}⚠️  Your system provides some context, but could be improved.{C.END}")
    print(f"{C.Y}   Consider checking if all repos are fully ingested.{C.END}")

# Cleanup
neo4j_driver.close()

header("FINAL SUMMARY")
print(f"1. Repo Isolation: {C.G}PASSED ✅{C.END}" if isolation_passed else f"1. Repo Isolation: {C.R}FAILED ❌{C.END}")
print(f"2. Context Richness: {C.G}PASSED ✅{C.END}" if richness_passed else f"2. Context Richness: {C.Y}PARTIAL ⚠️{C.END}")

if isolation_passed and richness_passed:
    print(f"\n{C.G}{C.BOLD}🚀 Your AI reviewer is ready for production!{C.END}")
    print(f"{C.G}   ✓ Perfect repo isolation (no data leaks){C.END}")
    print(f"{C.G}   ✓ Rich context for meaningful reviews{C.END}")
