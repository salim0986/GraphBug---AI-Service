# Context Retrieval Debugging Guide

## Problem Statement
Code reviews fail to fetch context from the repository even after running ingestion successfully.

## Root Cause Analysis

The issue can occur in two main pipelines:

### 1. **Ingestion Pipeline** (Repository → Neo4j/Qdrant)
- Clones repository
- Parses code files
- Extracts functions, classes, etc.
- Stores entities in **Neo4j** (graph database)
- Generates embeddings and stores in **Qdrant** (vector database)

**Common Issues:**
- Repository not cloned properly (private repos, auth issues)
- Parser failing on certain files
- Neo4j connection issues
- Qdrant connection issues
- Data stored with wrong `repo_id`

### 2. **Context Retrieval Pipeline** (Neo4j/Qdrant → Review Context)
- Receives PR file changes
- Queries Neo4j for entities in changed files
- Queries Neo4j for dependencies and relationships
- Queries Qdrant for similar code
- Merges all context for review

**Common Issues:**
- `repo_id` mismatch between ingestion and retrieval
- File paths don't match (absolute vs relative, normalization)
- Empty query results not handled properly
- Temporary GraphRAG not working as fallback

## Diagnostic Tools

### 🚀 Quick Check (Run this first!)

```bash
cd ai-service
source .venv/bin/activate  # or your Python environment
python quick_check_ingestion.py <repo_id>
```

**What it checks:**
- ✅ Neo4j connection and data
- ✅ Qdrant connection and vectors
- ✅ Sample data preview

**Example output:**
```
🗄️  CHECKING NEO4J
✅ DATA EXISTS
   Nodes: 245
   Relationships: 128
   Sample files:
     - src/api.py
     - src/workflow.py

🔍 CHECKING QDRANT
✅ VECTORS EXIST
   Total in collection: 1543
   For this repo: 245
```

---

### 🔬 Full Diagnostics (If quick check passes but reviews still fail)

```bash
python diagnose_context_retrieval.py <repo_id>
```

**What it tests:**
1. ✅ Database connections (Neo4j, Qdrant)
2. ✅ Data existence verification
3. ✅ Graph query methods (`find_related_by_file`, `find_file_dependencies`)
4. ✅ Vector search methods
5. ✅ Full context builder pipeline
6. ✅ Mock PR context generation

**Example output:**
```
TEST 1: Neo4j Connection ✅ PASS
TEST 2: Qdrant Connection ✅ PASS
TEST 3: Neo4j Data Verification ✅ PASS
TEST 4: Qdrant Vector Data ✅ PASS
TEST 5: Graph Query Methods ✅ PASS
TEST 6: Vector Search Methods ✅ PASS
TEST 7: Context Builder (Full Pipeline) ❌ FAIL
   Error: No entities found for file 'src/api.py'
```

---

## Common Issues & Solutions

### Issue 1: No Data in Databases

**Symptoms:**
```
❌ NO DATA FOUND
   No nodes found for repo_id='my-repo'
```

**Solution:**
Run ingestion:
```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_url": "https://github.com/owner/repo.git",
    "repo_id": "my-repo",
    "installation_id": "YOUR_INSTALLATION_ID"
  }'
```

---

### Issue 2: repo_id Mismatch

**Symptoms:**
- Ingestion succeeds but context retrieval finds no data
- Different `repo_id` used in ingestion vs review

**Solution:**
Ensure consistent `repo_id`:
- Check frontend: `lib/ai-service.ts` - what `repo_id` is sent?
- Check backend logs during ingestion for actual `repo_id` used
- Re-run ingestion with correct `repo_id`

**Example fix:**
```typescript
// frontend/lib/ai-service.ts
const repoId = `${owner}/${repo}`;  // Must match backend
```

---

### Issue 3: File Path Mismatch

**Symptoms:**
- Data exists in Neo4j
- `find_related_by_file()` returns empty results
- File paths in PR don't match database paths

**Cause:** 
Neo4j stores: `src/api.py`
PR provides: `/src/api.py` or `./src/api.py` or `api.py`

**Solution:**
Check file path normalization in `context_builder.py`:
```python
# Normalize file paths before querying
file_path = file_path.lstrip('./').lstrip('/')
```

---

### Issue 4: Private Repository Access

**Symptoms:**
```
❌ Git Clone Failed: Authentication failed
```

**Solution:**
Ensure GitHub App credentials are configured:
```bash
# In ai-service/.env
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
```

---

### Issue 5: Partial Ingestion

**Symptoms:**
- Neo4j has data ✅
- Qdrant has NO vectors ❌

**Cause:**
- Embedding model failed to load
- Qdrant connection lost during ingestion
- Batch processing timeout

**Solution:**
1. Check logs for embedding errors:
```bash
tail -f ai-service/logs/app.log | grep -i "embed\|qdrant"
```

2. Re-run ingestion with smaller batch size:
```python
# api.py - reduce batch_size
batch_size = 10  # Instead of 20
```

---

## Enhanced Logging

New debug logging has been added to trace the exact failure point:

### Context Builder Logs
```
[ContextBuilder] Building context for PR #123 in repo my-repo
[ContextBuilder] Input: 5 files, base=main, head=feature-branch
[ContextBuilder] Building context for file: src/api.py
[ContextBuilder]   repo_id=my-repo, language=python
[ContextBuilder]   Found 12 entities from Neo4j for src/api.py
[ContextBuilder]   Found 3 dependencies for src/api.py
```

### Graph DB Logs
```
[GraphDB] find_related_by_file: repo_id=my-repo, file=src/api.py, limit=10
[GraphDB] find_related_by_file returned 12 entities
[GraphDB] find_file_dependencies: repo_id=my-repo, file=src/api.py
[GraphDB] find_file_dependencies returned 3 dependencies
```

### Vector DB Logs
```
[VectorDB] search_similar: repo_id=my-repo, query='function authentication...', limit=5
[VectorDB] search_similar returned 8 results
```

**To enable debug logs:**
```python
# In ai-service/.env or config.py
LOG_LEVEL=DEBUG
```

---

## Debugging Workflow

### Step 1: Quick Check
```bash
python quick_check_ingestion.py <repo_id>
```
- ❌ If data missing → Run ingestion
- ✅ If data exists → Continue to Step 2

### Step 2: Full Diagnostics
```bash
python diagnose_context_retrieval.py <repo_id>
```
- Identify which test fails
- Check specific error messages
- Follow recommendations

### Step 3: Check Logs
```bash
# Check API logs during review
tail -f logs/app.log | grep -E "\[ContextBuilder\]|\[GraphDB\]|\[VectorDB\]"
```

### Step 4: Verify File Paths
```python
# Run in Python REPL with venv activated
from src.graph_builder import GraphBuilder
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

graph = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))

# Check what files are stored
with graph.driver.session() as session:
    result = session.run(
        "MATCH (n) WHERE n.repo_id = $repo_id AND n.file IS NOT NULL "
        "RETURN DISTINCT n.file as file LIMIT 10",
        repo_id="your-repo-id"
    )
    for r in result:
        print(r["file"])
```

### Step 5: Test Queries Directly
```python
# Test graph query
entities = graph.find_related_by_file("your-repo-id", "src/api.py", limit=10)
print(f"Found {len(entities)} entities")
for e in entities:
    print(f"  - {e['name']} ({e['type']}) at line {e['line']}")
```

---

## Code References

### Key Files
- **Ingestion:** `ai-service/src/api.py` (POST /ingest)
- **Context Builder:** `ai-service/src/context_builder.py`
- **Graph Queries:** `ai-service/src/graph_builder.py`
- **Vector Search:** `ai-service/src/vector_builder.py`
- **Review Workflow:** `ai-service/src/workflow.py`

### Query Methods
```python
# Graph queries used during context retrieval
graph_db.find_related_by_file(repo_id, file_path, limit=50)
graph_db.find_file_dependencies(repo_id, file_path)
graph_db.find_impacted_callers(repo_id, file_path, limit=5)

# Vector search
vector_db.search_similar(repo_id, query_text, limit=5)
```

---

## Next Steps After Debugging

1. **If ingestion is broken:**
   - Fix database connections
   - Verify authentication
   - Re-run ingestion
   - Run quick check to confirm

2. **If context retrieval is broken:**
   - Check `repo_id` consistency
   - Normalize file paths
   - Review query methods
   - Check temporary GraphRAG fallback

3. **If both work but reviews still fail:**
   - Check workflow.py analyze node
   - Verify Gemini API key
   - Check review generation logic

---

## Testing Commands

```bash
# 1. Quick health check
python quick_check_ingestion.py my-repo

# 2. Full diagnostics
python diagnose_context_retrieval.py my-repo

# 3. Run ingestion
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/owner/repo.git", "repo_id": "my-repo", "installation_id": "123456"}'

# 4. Check ingestion status
curl http://localhost:8000/health

# 5. Test review (simulate PR)
curl -X POST http://localhost:8000/review \
  -H 'Content-Type: application/json' \
  -d '{
    "pr_number": 1,
    "repo_id": "my-repo",
    "installation_id": "123456",
    "files": [{"filename": "src/api.py", "status": "modified", "additions": 10, "deletions": 5}]
  }'
```

---

## Support

If issues persist:
1. Share output from `diagnose_context_retrieval.py`
2. Share relevant log snippets
3. Share `repo_id` and sample file paths
4. Confirm Neo4j/Qdrant connection strings

---

**Last Updated:** 2026-02-02
