# Deploying AI Service to Hugging Face Spaces

This guide walks you through deploying the Graph Bug AI service to Hugging Face Spaces with external database hosting.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Hugging Face Spaces (Docker)                          │
│  ┌───────────────────────────────────────────────┐    │
│  │   FastAPI Service (api.py)                    │    │
│  │   - Code Analysis                             │    │
│  │   - Review Generation                         │    │
│  │   - GitHub Integration                        │    │
│  └───────────────────────────────────────────────┘    │
│                      ▼ ▼                               │
└──────────────────────┼─┼───────────────────────────────┘
                       │ │
         ┌─────────────┘ └─────────────┐
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│  Neo4j Aura     │          │  Qdrant Cloud   │
│  (Graph DB)     │          │  (Vector DB)    │
│  Free Tier      │          │  Free Tier      │
└─────────────────┘          └─────────────────┘
```

## Step 1: Set Up Neo4j (Graph Database)

### Option A: Neo4j Aura (Recommended - Managed Cloud)

1. **Sign Up**: Go to [https://neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/)
   - Create a free account
   - Select "AuraDB Free" tier (perfect for development/small projects)

2. **Create Instance**:
   - Click "New Instance"
   - Name: `graph-bug-ai`
   - Region: Choose closest to your users (e.g., AWS us-east-1)
   - Click "Create"

3. **Save Credentials**:
   - **IMPORTANT**: Download the credentials when shown (you can't see password again!)
   - You'll get:
     ```
     NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=<generated-password>
     ```

4. **Initialize Schema** (Optional but recommended):
   ```bash
   # Install Neo4j Python driver locally
   pip install neo4j
   
   # Run this Python script to create indexes:
   python3 << 'EOF'
   from neo4j import GraphDatabase
   
   uri = "neo4j+s://xxxxx.databases.neo4j.io"  # Your Aura URI
   user = "neo4j"
   password = "your-password-here"
   
   driver = GraphDatabase.driver(uri, auth=(user, password))
   
   with driver.session() as session:
       # Create indexes for performance
       queries = [
           "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
           "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
           "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
           "CREATE INDEX repo_id IF NOT EXISTS FOR (r:Repository) ON (r.repo_id)",
       ]
       for query in queries:
           session.run(query)
           print(f"✓ Executed: {query}")
   
   driver.close()
   print("✓ Neo4j schema initialized!")
   EOF
   ```

### Option B: Railway (If you want more control)

```bash
# Deploy Neo4j on Railway
railway login
railway init
railway add neo4j

# Get connection string from Railway dashboard
# Format: neo4j://default.railway.app:7687
```

---

## Step 2: Set Up Qdrant (Vector Database)

### Option A: Qdrant Cloud (Recommended - Managed)

1. **Sign Up**: Go to [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
   - Create free account
   - Free tier: 1GB storage (sufficient for small-medium projects)

2. **Create Cluster**:
   - Click "Create Cluster"
   - Name: `graph-bug-vectors`
   - Region: Choose same as Neo4j (for lower latency)
   - Tier: Free
   - Click "Create"

3. **Get API Key**:
   - Go to "Data Access Control"
   - Click "Create API Key"
   - Name: `hf-spaces-api-key`
   - Copy the key (you can't see it again!)
   - You'll get:
     ```
     QDRANT_URL=https://xxxxx-xxxxx.aws.cloud.qdrant.io
     QDRANT_API_KEY=your-api-key-here
     ```

4. **Create Collection**:
   ```bash
   # Install Qdrant client locally
   pip install qdrant-client
   
   # Run this to create the collection:
   python3 << 'EOF'
   from qdrant_client import QdrantClient
   from qdrant_client.models import Distance, VectorParams
   
   url = "https://xxxxx.aws.cloud.qdrant.io"
   api_key = "your-api-key-here"
   
   client = QdrantClient(url=url, api_key=api_key)
   
   # Create collection for code embeddings
   client.create_collection(
       collection_name="code_search",
       vectors_config=VectorParams(size=384, distance=Distance.COSINE)
   )
   
   print("✓ Qdrant collection 'code_search' created!")
   EOF
   ```

### Option B: Qdrant on Railway/Render

```bash
# Deploy Qdrant Docker container
# Railway: railway add (select Docker)
# Use image: qdrant/qdrant:latest
```

---

## Step 3: Create Dockerfile for Hugging Face Spaces

Create `Dockerfile` in `ai-service/` directory:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY .env.example .env

# Download embedding model at build time (cached in container)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 -c "import requests; requests.get('http://localhost:7860/health')"

# Run the service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Step 4: Create Hugging Face Spaces Configuration

Create `.spacesconfig.yaml` in `ai-service/`:

```yaml
sdk: docker
app_port: 7860
emoji: 🐛
title: Graph Bug AI Service
colorFrom: blue
colorTo: purple
license: mit
pinned: false
```

---

## Step 5: Prepare Environment Variables

Create `ai-service/.env.production` for reference (DON'T commit this):

```bash
# Neo4j Configuration (from Step 1)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-aura-password

# Qdrant Configuration (from Step 2)
QDRANT_URL=https://xxxxx.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Application Settings
TEMP_REPOS_DIR=/tmp/repos
LOG_LEVEL=INFO
ENVIRONMENT=production

# GitHub App Credentials
GITHUB_WEBHOOK_SECRET=your-webhook-secret
GITHUB_APP_ID=2604807
GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----\\n"

# CORS (add your frontend URL)
ALLOWED_ORIGINS=https://your-app.vercel.app,https://graphbug.com
```

---

## Step 6: Update Code for Production

Update `src/config.py` to handle Qdrant API key:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional for local
    
    # GitHub
    GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
    GITHUB_PRIVATE_KEY = os.getenv("GITHUB_PRIVATE_KEY")
    GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
    
    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Misc
    TEMP_REPOS_DIR = os.getenv("TEMP_REPOS_DIR", "./temp_repos")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
```

Update `src/vector_builder.py` to support Qdrant API key:

```python
def __init__(self, qdrant_url: str = "http://localhost:6333", api_key: Optional[str] = None):
    """Initialize with optional API key for Qdrant Cloud."""
    if api_key:
        self.qdrant = QdrantClient(url=qdrant_url, api_key=api_key)
    else:
        self.qdrant = QdrantClient(url=qdrant_url)
```

---

## Step 7: Deploy to Hugging Face Spaces

### Method A: Web UI (Easiest)

1. **Create Space**:
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Owner: Your username
   - Space name: `graph-bug-ai-service`
   - License: MIT
   - SDK: **Docker**
   - Hardware: CPU Basic (free) or T4 Small (paid, $0.60/hr)
   - Visibility: Public or Private
   - Click "Create Space"

2. **Upload Files**:
   - Upload `Dockerfile`
   - Upload `.spacesconfig.yaml`
   - Upload `requirements.txt`
   - Upload entire `src/` directory
   - Upload `.env.example` (as `.env`)

3. **Set Environment Variables**:
   - Go to Settings → Variables
   - Add secrets (these are encrypted):
     ```
     NEO4J_URI
     NEO4J_PASSWORD
     QDRANT_URL
     QDRANT_API_KEY
     GITHUB_WEBHOOK_SECRET
     GITHUB_APP_ID
     GITHUB_PRIVATE_KEY
     ```

4. **Deploy**:
   - Click "Build" → Space will build and deploy
   - Wait 5-10 minutes for first build
   - Check logs for any errors

### Method B: Git CLI (Advanced)

```bash
cd ai-service

# Initialize git repo (if not already)
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/graph-bug-ai-service
git lfs install  # For large files

# Add files
git add Dockerfile .spacesconfig.yaml requirements.txt src/

# Commit
git commit -m "Initial deployment"

# Push to HF Spaces
git push space main

# Set secrets via CLI
huggingface-cli login
huggingface-cli repo secret set NEO4J_URI --repo-type space --repo YOUR_USERNAME/graph-bug-ai-service
# Repeat for all secrets
```

---

## Step 8: Update Frontend Environment Variables

Update `frontend/.env.production`:

```bash
# Replace with your HF Spaces URL
NEXT_PUBLIC_AI_SERVICE_URL=https://YOUR_USERNAME-graph-bug-ai-service.hf.space
AI_SERVICE_URL=https://YOUR_USERNAME-graph-bug-ai-service.hf.space
```

---

## Step 9: Test the Deployment

### Test Health Endpoint

```bash
curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/health
# Expected: {"status": "healthy"}
```

### Test Database Connections

```bash
curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/debug/connections
# Should show Neo4j and Qdrant connection status
```

### Test Ingestion

```bash
curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/ingest \\
  -H "Content-Type: application/json" \\
  -d '{
    "repo_url": "https://github.com/octocat/Hello-World",
    "gemini_api_key": "your-test-key"
  }'
```

---

## Step 10: Update GitHub App Webhook URL

1. Go to [GitHub Apps Settings](https://github.com/settings/apps)
2. Select your "Graph Bug AI" app
3. Update Webhook URL:
   ```
   https://YOUR_USERNAME-graph-bug-ai-service.hf.space/webhook/github
   ```
4. Save changes

---

## Troubleshooting

### Build Fails

**Issue**: Dockerfile build errors
```bash
# Check logs in HF Spaces UI
# Common fixes:
# 1. Ensure requirements.txt has all dependencies
# 2. Check Python version compatibility
# 3. Verify Dockerfile syntax
```

### Database Connection Errors

**Issue**: Can't connect to Neo4j/Qdrant
```bash
# Test locally first:
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j+s://xxxx', auth=('neo4j', 'pass'))
driver.verify_connectivity()
print('✓ Neo4j connected')
"

python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='https://xxxx', api_key='key')
print(client.get_collections())
print('✓ Qdrant connected')
"
```

### Memory Issues

**Issue**: Container runs out of memory
```
# Solutions:
# 1. Upgrade to CPU Basic → T4 Small
# 2. Reduce embedding model size
# 3. Add pagination to large queries
```

### CORS Errors

**Issue**: Frontend can't connect
```bash
# Verify ALLOWED_ORIGINS includes your frontend
# Add to HF Spaces secrets:
ALLOWED_ORIGINS=https://your-app.vercel.app,https://graphbug.com
```

---

## Cost Breakdown

| Service | Tier | Cost |
|---------|------|------|
| Hugging Face Spaces | CPU Basic | **Free** |
| Neo4j Aura | Free Tier | **Free** (50MB storage) |
| Qdrant Cloud | Free Tier | **Free** (1GB storage) |
| **Total** | | **$0/month** 🎉 |

### Paid Upgrades (Optional)

- **HF Spaces T4 Small**: $0.60/hr (~$432/month for 24/7)
- **Neo4j Aura Pro**: From $65/month
- **Qdrant Cloud**: From $25/month

---

## Production Checklist

- [ ] Neo4j Aura instance created with credentials saved
- [ ] Qdrant Cloud cluster created with API key saved
- [ ] Collections created in Qdrant (`code_search`)
- [ ] Indexes created in Neo4j (file_path, class_name, etc.)
- [ ] Dockerfile created in `ai-service/`
- [ ] `.spacesconfig.yaml` created
- [ ] Code updated to support Qdrant API key
- [ ] HF Space created and files uploaded
- [ ] All secrets added to HF Spaces settings
- [ ] Space deployed successfully (check logs)
- [ ] Health endpoint responds: `/health`
- [ ] Frontend `.env.production` updated with HF Spaces URL
- [ ] GitHub App webhook URL updated
- [ ] Test end-to-end: Create PR → AI reviews it
- [ ] Monitor HF Spaces logs for errors

---

## Monitoring & Maintenance

### View Logs
```bash
# HF Spaces logs are visible in the UI
# Or use CLI:
huggingface-cli repo logs YOUR_USERNAME/graph-bug-ai-service --repo-type space
```

### Database Backups

**Neo4j Aura**: Automatic daily backups (7-day retention)
**Qdrant Cloud**: Automatic snapshots

### Scaling

When you outgrow free tiers:
1. **HF Spaces**: Upgrade to T4 Small GPU
2. **Neo4j**: Migrate to Aura Professional
3. **Qdrant**: Upgrade to paid cluster
4. **Alternative**: Deploy to Railway/Render with Docker Compose

---

## Next Steps

After deployment:
1. Deploy frontend to Vercel (see `PRODUCTION_ENV_FRONTEND.md`)
2. Test full flow: Sign in → Add API key → Install app → Create PR
3. Monitor costs and performance
4. Set up alerts (HF Spaces + database health checks)
5. Configure CI/CD for automatic deployments

🚀 **You're ready to deploy!**
