# 🚀 Hugging Face Spaces Deployment - Complete Guide

## 📋 What You Need

Your AI service will be deployed to **Hugging Face Spaces** with:
- **Neo4j Aura** (free tier) - Graph database for code structure
- **Qdrant Cloud** (free tier) - Vector database for semantic search
- **HF Spaces** (free tier) - Docker container hosting

**Total Cost: $0/month** 🎉

---

## 📚 Documentation Files Created

| File | Purpose | When to Use |
|------|---------|-------------|
| **QUICK_START_HF.md** | ⭐ Start here! Step-by-step guide | First-time deployment |
| **HUGGINGFACE_DEPLOYMENT.md** | Complete reference documentation | Detailed info & troubleshooting |
| **HF_DEPLOYMENT_CHECKLIST.md** | Verification checklist | During & after deployment |
| **Dockerfile** | Container configuration | Automatic (used by HF Spaces) |
| **README_HF.md** | Space documentation | Shown on your HF Space page |
| **deploy-hf.sh** | Automated deployment script | Quick deploy to HF Spaces |
| **init_neo4j.py** | Neo4j database setup | Initialize graph database |
| **init_qdrant.py** | Qdrant database setup | Initialize vector database |

---

## 🎯 Quick Start (3 Steps)

### Step 1: Set Up Databases (10 minutes)

#### Neo4j Aura (Graph Database)
```bash
# 1. Sign up: https://neo4j.com/cloud/aura-free/
# 2. Create free instance
# 3. Save credentials
# 4. Initialize:
cd ai-service
export NEO4J_URI="neo4j+s://xxxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
python3 init_neo4j.py
```

#### Qdrant Cloud (Vector Database)
```bash
# 1. Sign up: https://cloud.qdrant.io/
# 2. Create free cluster
# 3. Get API key
# 4. Initialize:
export QDRANT_URL="https://xxxxx.aws.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key"
python3 init_qdrant.py
```

### Step 2: Deploy to Hugging Face (5 minutes)

```bash
# 1. Install CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Deploy (automated)
cd ai-service
./deploy-hf.sh

# Or manually:
huggingface-cli repo create graph-bug-ai-service --type space --space-sdk docker
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/graph-bug-ai-service
git add Dockerfile README_HF.md requirements.txt src/
git commit -m "Initial deployment"
git push space main
```

### Step 3: Configure Environment Variables

Go to your HF Space settings and add these secrets:

```bash
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
QDRANT_URL=https://xxxxx.aws.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
GITHUB_APP_ID=2604807
GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
GITHUB_WEBHOOK_SECRET=your-webhook-secret
ALLOWED_ORIGINS=https://your-app.vercel.app
```

✅ **Done! Your service is live at:**
`https://YOUR_USERNAME-graph-bug-ai-service.hf.space`

---

## 🧪 Testing

### Health Check
```bash
curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/health
# Expected: {"status":"healthy","neo4j":"connected","qdrant":"connected"}
```

### Test Ingestion
```bash
curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/octocat/Hello-World",
    "gemini_api_key": "your-api-key"
  }'
```

### Update GitHub Webhook
```
https://YOUR_USERNAME-graph-bug-ai-service.hf.space/webhook/github
```

---

## 📖 Detailed Guides

### For First-Time Deployment
**Start with:** [QUICK_START_HF.md](./QUICK_START_HF.md)
- Step-by-step instructions
- Screenshots and examples
- Expected outputs
- Common issues

### For Detailed Information
**Read:** [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md)
- Architecture overview
- Database setup options
- Code modifications needed
- Troubleshooting guide
- Cost breakdown
- Scaling options

### For Verification
**Use:** [HF_DEPLOYMENT_CHECKLIST.md](./HF_DEPLOYMENT_CHECKLIST.md)
- Pre-deployment checklist
- Deployment steps
- Testing procedures
- Post-deployment tasks
- Monitoring setup

---

## 🔧 Code Changes Made

### 1. Added Qdrant API Key Support
**File:** `src/vector_builder.py`
- Added `api_key` parameter to `__init__`
- Supports both local and cloud Qdrant

### 2. Updated Configuration
**File:** `src/config.py`
- Added `QDRANT_API_KEY` environment variable
- Loads from `.env` file

### 3. Updated API Initialization
**File:** `src/api.py`
- Uses config values instead of hardcoded localhost
- Passes API key to VectorBuilder

---

## 🚨 Important Notes

### Database Credentials
- **Neo4j password**: You can't see it again after creation! Save immediately.
- **Qdrant API key**: You can't see it again after creation! Save immediately.
- Store credentials securely (1Password, LastPass, etc.)

### Free Tier Limits
- **Neo4j Aura Free**: 50MB storage, 1 database
- **Qdrant Cloud Free**: 1GB storage, 1 cluster
- **HF Spaces CPU Basic**: Unlimited uptime (with usage)

### When to Upgrade?
- **Neo4j**: When you exceed 50MB (upgrade to ~$65/mo)
- **Qdrant**: When you exceed 1GB vectors (upgrade to ~$25/mo)
- **HF Spaces**: For GPU or faster response (~$0.60/hr)

---

## 🔍 Architecture

```
┌──────────────────────────────────────────────────────┐
│  User's Browser                                       │
│  ┌──────────────────────────────────────────┐        │
│  │  Frontend (Vercel)                       │        │
│  │  - Dashboard                             │        │
│  │  - GitHub OAuth                          │        │
│  └──────────────┬───────────────────────────┘        │
└─────────────────┼────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│  Hugging Face Spaces (Docker)                        │
│  ┌──────────────────────────────────────────┐        │
│  │  FastAPI Service                         │        │
│  │  - Code Analysis                         │        │
│  │  - PR Review Generation                  │        │
│  │  - GitHub Webhook Handler                │        │
│  └──────────────┬───────────┬───────────────┘        │
└─────────────────┼───────────┼──────────────────────┘
                  │           │
         ┌────────┘           └────────┐
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│  Neo4j Aura     │          │  Qdrant Cloud   │
│                 │          │                 │
│  Graph DB       │          │  Vector DB      │
│  - Code Graph   │          │  - Embeddings   │
│  - Relationships│          │  - Semantic     │
│  - Queries      │          │    Search       │
└─────────────────┘          └─────────────────┘
         ▲                              ▲
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  GitHub         │
              │  - Webhooks     │
              │  - PR Events    │
              └─────────────────┘
```

---

## 📝 Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `NEO4J_URI` | ✅ | Neo4j connection string | `neo4j+s://xxx.databases.neo4j.io` |
| `NEO4J_USER` | ✅ | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | ✅ | Neo4j password | From Neo4j Aura |
| `QDRANT_URL` | ✅ | Qdrant server URL | `https://xxx.cloud.qdrant.io` |
| `QDRANT_API_KEY` | ✅ | Qdrant API key | From Qdrant Cloud |
| `GITHUB_APP_ID` | ✅ | GitHub App ID | `2604807` |
| `GITHUB_PRIVATE_KEY` | ✅ | GitHub App private key | Full PEM key with `\n` |
| `GITHUB_WEBHOOK_SECRET` | ✅ | Webhook secret | From GitHub App settings |
| `ALLOWED_ORIGINS` | ⚠️ | CORS origins (comma-separated) | `https://app.vercel.app` |
| `EMBEDDING_MODEL` | ❌ | Sentence transformer model | `all-MiniLM-L6-v2` (default) |
| `LOG_LEVEL` | ❌ | Logging verbosity | `INFO` (default) |
| `ENVIRONMENT` | ❌ | Environment name | `production` |

---

## 🆘 Quick Troubleshooting

### Build Fails
```bash
# Test locally first:
cd ai-service
docker build -t test-service .
docker run -p 7860:7860 test-service
```

### Can't Connect to Neo4j
```bash
# Verify credentials:
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j+s://xxx', auth=('neo4j', 'pass'))
driver.verify_connectivity()
print('✅ Connected')
"
```

### Can't Connect to Qdrant
```bash
# Verify credentials:
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='https://xxx', api_key='key')
print(client.get_collections())
print('✅ Connected')
"
```

### Webhook Not Working
1. Check URL in GitHub App settings
2. Verify webhook secret matches
3. Check HF Spaces logs for incoming requests
4. Test webhook delivery in GitHub App → Advanced

---

## 🎉 Next Steps

After successful deployment:

1. **Test End-to-End**
   - Create a PR in a test repository
   - Verify AI review comment appears

2. **Update Frontend**
   - Set `AI_SERVICE_URL` to your HF Space URL
   - Deploy frontend to Vercel

3. **Monitor**
   - Check HF Spaces logs regularly
   - Monitor Neo4j storage usage
   - Monitor Qdrant point count

4. **Optimize**
   - Tune rate limits if needed
   - Adjust embedding batch sizes
   - Optimize database queries

---

## 📞 Support

- **Documentation Issues**: Check [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md)
- **Database Issues**: See database provider docs (Neo4j Aura, Qdrant Cloud)
- **HF Spaces Issues**: Check [HF Spaces documentation](https://huggingface.co/docs/hub/spaces)
- **GitHub App Issues**: Check GitHub App settings and webhook deliveries

---

## ✅ Success Criteria

Your deployment is successful when:

- [x] Health endpoint returns `200 OK`
- [x] Neo4j connection shows "connected"
- [x] Qdrant connection shows "connected"
- [x] Ingestion creates nodes in Neo4j
- [x] Ingestion creates vectors in Qdrant
- [x] Webhook receives GitHub events
- [x] PR reviews are posted automatically
- [x] Frontend can connect to AI service

---

**Ready to deploy?** Start with [QUICK_START_HF.md](./QUICK_START_HF.md) 🚀

**Need more details?** Read [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md) 📖

**Deploying now?** Use [HF_DEPLOYMENT_CHECKLIST.md](./HF_DEPLOYMENT_CHECKLIST.md) ✅
