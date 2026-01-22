# 🚀 Quick Start: Deploy to Hugging Face Spaces

Follow these steps to deploy your AI service to Hugging Face Spaces with external databases.

## Prerequisites

- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] Hugging Face account: [https://huggingface.co/join](https://huggingface.co/join)

## Step 1: Set Up Neo4j Aura (5 minutes)

**Why Neo4j?** Stores your code's structure as a graph (files → functions → calls → dependencies)

1. **Sign up**: Go to [https://neo4j.com/cloud/aura-free/](https://neo4j.com/cloud/aura-free/)
   - Click "Start Free"
   - Sign up with email or Google

2. **Create instance**:
   - Click "New Instance"
   - Name: `graph-bug-ai`
   - Select: **AuraDB Free** (perfect for testing)
   - Region: Choose closest to you (e.g., `aws-us-east-1`)
   - Click "Create"

3. **Save credentials** (CRITICAL - you can't see password again!):
   ```bash
   # Download the .txt file or copy:
   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=<generated-password>
   ```

4. **Initialize database**:
   ```bash
   cd ai-service
   
   # Set credentials
   export NEO4J_URI="neo4j+s://xxxxx.databases.neo4j.io"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your-password"
   
   # Run initialization script
   python3 init_neo4j.py
   
   # Expected output:
   # ✅ Connected to Neo4j successfully!
   # ✅ Repository.repo_id index
   # ✅ File.path index
   # ... more indexes ...
   # 🎉 Neo4j initialization complete!
   ```

✅ **Neo4j is ready!**

---

## Step 2: Set Up Qdrant Cloud (5 minutes)

**Why Qdrant?** Powers semantic code search using vector embeddings (find similar code by meaning)

1. **Sign up**: Go to [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
   - Click "Sign Up"
   - Use GitHub or email

2. **Create cluster**:
   - Click "Create Cluster"
   - Name: `graph-bug-vectors`
   - Cloud: AWS (or same provider as Neo4j)
   - Region: Same as Neo4j for low latency
   - Select: **Free Tier** (1GB storage)
   - Click "Create"

3. **Get API credentials**:
   - Go to "Data Access Control" tab
   - Click "Create API Key"
   - Name: `hf-spaces-api-key`
   - **Copy the key immediately** (you can't see it again!)
   
   ```bash
   QDRANT_URL=https://xxxxx-xxxxx.aws.cloud.qdrant.io
   QDRANT_API_KEY=your-api-key-here
   ```

4. **Initialize collection**:
   ```bash
   cd ai-service
   
   # Set credentials
   export QDRANT_URL="https://xxxxx.aws.cloud.qdrant.io"
   export QDRANT_API_KEY="your-api-key"
   
   # Run initialization script
   python3 init_qdrant.py
   
   # Expected output:
   # ✅ Connected with API key authentication
   # 📦 Creating collection 'repo_code'...
   # ✅ Collection 'repo_code' created successfully!
   # 🎉 Qdrant initialization complete!
   ```

✅ **Qdrant is ready!**

---

## Step 3: Deploy to Hugging Face Spaces (10 minutes)

1. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   # Paste your token from: https://huggingface.co/settings/tokens
   ```

3. **Verify files exist**:
   ```bash
   cd ai-service
   ls -la
   
   # You should see:
   # - Dockerfile ✅
   # - README_HF.md ✅
   # - requirements.txt ✅
   # - src/ directory ✅
   ```

4. **Deploy** (automated script):
   ```bash
   ./deploy-hf.sh
   
   # Or manually:
   huggingface-cli repo create graph-bug-ai-service --type space --space-sdk docker
   git init
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/graph-bug-ai-service
   git add Dockerfile README_HF.md requirements.txt src/
   git commit -m "Initial deployment"
   git push space main
   ```

5. **Add environment variables** (CRITICAL):
   - Go to: `https://huggingface.co/spaces/YOUR_USERNAME/graph-bug-ai-service/settings`
   - Scroll to "Repository secrets"
   - Add each secret:

   | Variable | Value | Example |
   |----------|-------|---------|
   | `NEO4J_URI` | From Step 1 | `neo4j+s://xxxxx.databases.neo4j.io` |
   | `NEO4J_USER` | From Step 1 | `neo4j` |
   | `NEO4J_PASSWORD` | From Step 1 | `your-password` |
   | `QDRANT_URL` | From Step 2 | `https://xxxxx.aws.cloud.qdrant.io` |
   | `QDRANT_API_KEY` | From Step 2 | `your-api-key` |
   | `GITHUB_APP_ID` | From .env | `2604807` |
   | `GITHUB_PRIVATE_KEY` | From .env | Full private key |
   | `GITHUB_WEBHOOK_SECRET` | From .env | Your webhook secret |
   | `ALLOWED_ORIGINS` | Your frontend URL | `https://your-app.vercel.app` |

6. **Wait for build** (5-10 minutes):
   - Check build logs in the "Logs" tab
   - Look for: `✓ Starting...` and `✓ Ready in XXXXms`

7. **Test deployment**:
   ```bash
   # Replace YOUR_USERNAME with your HF username
   curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/health
   
   # Expected response:
   # {"status":"healthy","neo4j":"connected","qdrant":"connected"}
   ```

✅ **AI Service is live!**

Your service URL: `https://YOUR_USERNAME-graph-bug-ai-service.hf.space`

---

## Step 4: Update Frontend Configuration

Update your frontend to point to the Hugging Face Space:

```bash
cd ../frontend

# Create or edit .env.production
cat >> .env.production << EOF
# AI Service (Hugging Face Spaces)
NEXT_PUBLIC_AI_SERVICE_URL=https://YOUR_USERNAME-graph-bug-ai-service.hf.space
AI_SERVICE_URL=https://YOUR_USERNAME-graph-bug-ai-service.hf.space
EOF
```

---

## Step 5: Update GitHub App Webhook

1. Go to [GitHub Apps Settings](https://github.com/settings/apps/graph-bug-ai)
2. Click "Edit"
3. Update **Webhook URL**:
   ```
   https://YOUR_USERNAME-graph-bug-ai-service.hf.space/webhook/github
   ```
4. Click "Save changes"

---

## Step 6: Test End-to-End

1. **Test ingestion**:
   ```bash
   curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/ingest \
     -H "Content-Type: application/json" \
     -d '{
       "repo_url": "https://github.com/octocat/Hello-World",
       "gemini_api_key": "your-gemini-api-key"
     }'
   
   # Expected: {"message": "Ingestion started...", "status": "success"}
   ```

2. **Check Neo4j** (has data?):
   - Go to Neo4j Aura Console: https://console.neo4j.io/
   - Click your instance
   - Click "Query"
   - Run: `MATCH (n) RETURN count(n)`
   - Should show nodes > 0

3. **Check Qdrant** (has vectors?):
   - Go to Qdrant Console: https://cloud.qdrant.io/
   - Click your cluster
   - Go to "Collections" → `repo_code`
   - Should show points count > 0

4. **Test full PR review**:
   - Create a PR in a repository with your GitHub App installed
   - Check HF Spaces logs for webhook activity
   - Verify AI comment appears on PR

---

## Troubleshooting

### Build fails with "requirements.txt not found"
```bash
# Ensure you're in ai-service directory:
cd ai-service
ls -la requirements.txt  # Should exist
```

### "Cannot connect to Neo4j"
```bash
# Test locally first:
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j+s://xxxxx.databases.neo4j.io', auth=('neo4j', 'password'))
driver.verify_connectivity()
print('✅ Connected!')
"
```

### "Cannot connect to Qdrant"
```bash
# Test locally:
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='https://xxxxx.aws.cloud.qdrant.io', api_key='key')
print(client.get_collections())
print('✅ Connected!')
"
```

### "Port 7860 required but not exposed"
- HF Spaces requires port **7860**
- Check your Dockerfile: `EXPOSE 7860`
- Check CMD: `--port 7860`

### Space shows "Building" forever
- Check logs in HF Spaces UI
- Common issues:
  - Missing dependencies in requirements.txt
  - Syntax errors in Dockerfile
  - Import errors in Python code

### Webhook not triggering
- Verify webhook URL in GitHub App settings
- Check webhook secret matches
- Test webhook delivery in GitHub App → Advanced → Recent Deliveries

---

## Monitoring

### View Logs
```bash
# Real-time logs
huggingface-cli repo logs YOUR_USERNAME/graph-bug-ai-service --repo-type space --follow
```

### Check Database Health
```bash
curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/debug/connections
```

### Monitor Neo4j
- Console: https://console.neo4j.io/
- Check query performance
- Monitor storage usage (50MB free tier limit)

### Monitor Qdrant
- Console: https://cloud.qdrant.io/
- Check point count
- Monitor storage usage (1GB free tier limit)

---

## Costs

| Service | Tier | Storage | Monthly Cost |
|---------|------|---------|--------------|
| **HF Spaces** | CPU Basic (free) | N/A | **$0** |
| **Neo4j Aura** | Free | 50MB | **$0** |
| **Qdrant Cloud** | Free | 1GB | **$0** |
| **Total** | | | **$0/month** 🎉 |

### When to Upgrade?

**Upgrade HF Spaces** ($0.60/hr) if:
- Need faster response times
- High concurrent requests
- GPU acceleration needed

**Upgrade Neo4j** ($65/mo) if:
- Storage > 50MB
- Need > 1 database
- Need backups

**Upgrade Qdrant** ($25/mo) if:
- Vectors > 1GB
- Need > 100k points
- High query volume

---

## What's Next?

- [ ] Deploy frontend to Vercel (see `PRODUCTION_ENV_FRONTEND.md`)
- [ ] Set up custom domain
- [ ] Configure monitoring/alerts
- [ ] Add usage analytics
- [ ] Set up CI/CD pipeline

🎉 **Congratulations! Your AI code review service is live!**

Need help? Check:
- Full guide: `HUGGINGFACE_DEPLOYMENT.md`
- Frontend guide: `PRODUCTION_ENV_FRONTEND.md`
- Troubleshooting: Open an issue on GitHub
