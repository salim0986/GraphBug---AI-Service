# Hugging Face Spaces Deployment Checklist

Use this checklist to ensure a smooth deployment.

## Pre-Deployment

### Database Setup
- [ ] **Neo4j Aura account created**
  - URL: https://neo4j.com/cloud/aura-free/
  - [ ] Instance created (Free tier)
  - [ ] Credentials saved securely (URI, user, password)
  - [ ] Indexes initialized (`python3 init_neo4j.py`)
  - [ ] Connection tested successfully

- [ ] **Qdrant Cloud account created**
  - URL: https://cloud.qdrant.io/
  - [ ] Cluster created (Free tier)
  - [ ] API key generated and saved
  - [ ] Collection created (`python3 init_qdrant.py`)
  - [ ] Connection tested successfully

### Code Preparation
- [ ] **Files ready in `ai-service/` directory**
  - [ ] `Dockerfile` exists
  - [ ] `README_HF.md` exists
  - [ ] `requirements.txt` exists
  - [ ] `src/` directory with all Python files
  - [ ] `.dockerignore` configured

- [ ] **Environment variables documented**
  - [ ] Neo4j credentials noted
  - [ ] Qdrant credentials noted
  - [ ] GitHub App credentials available
  - [ ] Frontend URL for CORS noted

### Hugging Face Setup
- [ ] **HF account created**
  - [ ] Account verified
  - [ ] Access token created (https://huggingface.co/settings/tokens)
  - [ ] CLI installed (`pip install huggingface_hub`)
  - [ ] Logged in (`huggingface-cli login`)

## Deployment

### Create Space
- [ ] **Space created on Hugging Face**
  - Method: Web UI or CLI
  - [ ] Space name: `graph-bug-ai-service`
  - [ ] SDK: Docker
  - [ ] Visibility: Public or Private
  - [ ] Hardware: CPU Basic (free)

### Upload Code
- [ ] **Code pushed to space**
  - [ ] Git repository initialized
  - [ ] Remote added: `git remote add space ...`
  - [ ] Files committed
  - [ ] Pushed to space: `git push space main`
  - [ ] Build started (check Logs tab)

### Configure Environment
- [ ] **All secrets added** (Settings → Repository secrets):
  - [ ] `NEO4J_URI` = `neo4j+s://xxxxx.databases.neo4j.io`
  - [ ] `NEO4J_USER` = `neo4j`
  - [ ] `NEO4J_PASSWORD` = `<your-password>`
  - [ ] `QDRANT_URL` = `https://xxxxx.aws.cloud.qdrant.io`
  - [ ] `QDRANT_API_KEY` = `<your-api-key>`
  - [ ] `GITHUB_APP_ID` = `2604807`
  - [ ] `GITHUB_PRIVATE_KEY` = `-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n`
  - [ ] `GITHUB_WEBHOOK_SECRET` = `<your-secret>`
  - [ ] `ALLOWED_ORIGINS` = `https://your-app.vercel.app` (optional)
  - [ ] `LOG_LEVEL` = `INFO` (optional)
  - [ ] `ENVIRONMENT` = `production` (optional)

### Build Verification
- [ ] **Build completed successfully**
  - [ ] No errors in build logs
  - [ ] Container started
  - [ ] Health check passing
  - [ ] Status shows "Running"

## Testing

### Basic Connectivity
- [ ] **Health endpoint responds**
  ```bash
  curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/health
  # Expected: {"status":"healthy","neo4j":"connected","qdrant":"connected"}
  ```

- [ ] **API documentation accessible**
  ```bash
  curl https://YOUR_USERNAME-graph-bug-ai-service.hf.space/docs
  # Should return FastAPI Swagger UI HTML
  ```

### Database Connections
- [ ] **Neo4j connection working**
  - [ ] Test query returns results
  - [ ] Indexes present
  - [ ] No authentication errors in logs

- [ ] **Qdrant connection working**
  - [ ] Collection accessible
  - [ ] Can query collections
  - [ ] No authentication errors in logs

### Feature Testing
- [ ] **Ingestion endpoint works**
  ```bash
  curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/ingest \
    -H "Content-Type: application/json" \
    -d '{"repo_url": "https://github.com/octocat/Hello-World", "gemini_api_key": "test-key"}'
  ```
  - [ ] Returns success response
  - [ ] Data appears in Neo4j
  - [ ] Vectors appear in Qdrant

- [ ] **PR analysis works**
  ```bash
  curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/analyze/pr \
    -H "Content-Type: application/json" \
    -d '{"repo_id": "owner/repo", "pr_number": 1, "gemini_api_key": "test-key"}'
  ```
  - [ ] Returns analysis result
  - [ ] No errors in logs

- [ ] **Webhook endpoint accessible**
  ```bash
  curl -X POST https://YOUR_USERNAME-graph-bug-ai-service.hf.space/webhook/github \
    -H "Content-Type: application/json" \
    -d '{}'
  # Should return 400 (missing signature) - this is expected
  ```

## Integration

### GitHub App
- [ ] **Webhook URL updated**
  - [ ] URL: `https://YOUR_USERNAME-graph-bug-ai-service.hf.space/webhook/github`
  - [ ] Webhook secret matches environment variable
  - [ ] Webhook active and delivering
  - [ ] Test delivery successful

- [ ] **GitHub App installed**
  - [ ] Installed on test repository
  - [ ] Permissions granted
  - [ ] Webhook events configured

### Frontend Integration
- [ ] **Frontend environment variables updated**
  - [ ] `NEXT_PUBLIC_AI_SERVICE_URL` points to HF Space
  - [ ] `AI_SERVICE_URL` points to HF Space
  - [ ] CORS configured correctly

- [ ] **Frontend can connect to AI service**
  - [ ] API calls succeed
  - [ ] No CORS errors
  - [ ] Authentication works

## End-to-End Testing

### Full Flow Test
- [ ] **Test complete PR review flow**
  1. [ ] User signs in to frontend
  2. [ ] User adds Gemini API key
  3. [ ] User installs GitHub App on repository
  4. [ ] Repository ingestion triggered
  5. [ ] Data appears in Neo4j/Qdrant
  6. [ ] User creates PR
  7. [ ] Webhook received by HF Space
  8. [ ] AI analysis triggered
  9. [ ] Review comment posted on PR
  10. [ ] Review visible on frontend dashboard

### Performance Check
- [ ] **Response times acceptable**
  - [ ] Health check: < 1s
  - [ ] Ingestion starts: < 5s
  - [ ] PR analysis: < 30s
  - [ ] Webhook processing: < 5s

### Error Handling
- [ ] **Graceful error handling**
  - [ ] Invalid requests return proper error codes
  - [ ] Database connection failures logged
  - [ ] GitHub API errors handled
  - [ ] Gemini API errors handled

## Monitoring

### Logs
- [ ] **Log monitoring set up**
  - [ ] Can access HF Spaces logs
  - [ ] Log level appropriate (INFO recommended)
  - [ ] No unexpected errors or warnings
  - [ ] Request/response logging working

### Database Health
- [ ] **Neo4j monitoring**
  - [ ] Can access Neo4j Console
  - [ ] Storage usage < 50MB (free tier limit)
  - [ ] Query performance acceptable
  - [ ] No memory warnings

- [ ] **Qdrant monitoring**
  - [ ] Can access Qdrant Console
  - [ ] Storage usage < 1GB (free tier limit)
  - [ ] Point count growing as expected
  - [ ] Query performance acceptable

### Alerts
- [ ] **Basic alerting configured**
  - [ ] HF Space downtime notifications
  - [ ] Database connection failures
  - [ ] Webhook delivery failures (GitHub)

## Documentation

### User-Facing
- [ ] **README updated**
  - [ ] HF Space URL documented
  - [ ] API endpoints listed
  - [ ] Setup instructions clear
  - [ ] Example requests provided

### Internal
- [ ] **Deployment documented**
  - [ ] Environment variables documented
  - [ ] Deployment process documented
  - [ ] Rollback procedure documented
  - [ ] Troubleshooting guide available

## Security

### Credentials
- [ ] **All secrets stored securely**
  - [ ] Not committed to git
  - [ ] Only in HF Space secrets
  - [ ] Access restricted
  - [ ] Backup stored securely

### API Security
- [ ] **Security measures in place**
  - [ ] Rate limiting configured
  - [ ] CORS properly configured
  - [ ] Webhook signature verification working
  - [ ] GitHub App private key secure

### Compliance
- [ ] **Privacy & compliance**
  - [ ] User data handling documented
  - [ ] Privacy policy updated
  - [ ] GDPR compliance (if applicable)
  - [ ] Terms of service updated

## Post-Deployment

### Communication
- [ ] **Stakeholders notified**
  - [ ] Users informed of new deployment
  - [ ] Team notified of URL change
  - [ ] Documentation links shared

### Cleanup
- [ ] **Old resources decommissioned**
  - [ ] Local development containers stopped
  - [ ] Old deployments removed
  - [ ] Unused secrets deleted

### Future Planning
- [ ] **Roadmap updated**
  - [ ] Next features planned
  - [ ] Upgrade path defined (if needed)
  - [ ] Scaling strategy documented
  - [ ] Backup/disaster recovery plan

---

## Troubleshooting Guide

### Issue: Build Fails

**Symptoms**: Build logs show errors, status stuck on "Building"

**Check**:
- [ ] Dockerfile syntax correct
- [ ] All files referenced exist
- [ ] requirements.txt has all dependencies
- [ ] Port 7860 exposed

**Fix**:
```bash
# Test Dockerfile locally
docker build -t test-ai-service .
docker run -p 7860:7860 test-ai-service
```

---

### Issue: Cannot Connect to Neo4j

**Symptoms**: Logs show "Neo4j connection failed"

**Check**:
- [ ] `NEO4J_URI` secret is correct (neo4j+s:// for Aura)
- [ ] `NEO4J_PASSWORD` matches what Neo4j gave you
- [ ] Neo4j instance is running (check Aura console)
- [ ] Firewall allows connections from HF Spaces

**Fix**:
```bash
# Test connection locally
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j+s://xxx', auth=('neo4j', 'pass'))
driver.verify_connectivity()
"
```

---

### Issue: Cannot Connect to Qdrant

**Symptoms**: Logs show "Qdrant connection failed"

**Check**:
- [ ] `QDRANT_URL` secret is correct (https:// for cloud)
- [ ] `QDRANT_API_KEY` secret is set and correct
- [ ] Qdrant cluster is running (check cloud console)
- [ ] Collection exists

**Fix**:
```bash
# Test connection locally
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='https://xxx', api_key='key')
print(client.get_collections())
"
```

---

### Issue: Webhook Not Triggering

**Symptoms**: PRs created but no AI reviews

**Check**:
- [ ] Webhook URL correct in GitHub App settings
- [ ] Webhook secret matches `GITHUB_WEBHOOK_SECRET`
- [ ] GitHub App installed on repository
- [ ] GitHub showing successful deliveries

**Fix**:
1. Check GitHub App → Settings → Advanced → Recent Deliveries
2. Look for 200 response codes
3. If 4xx/5xx, check HF Space logs for errors

---

### Issue: CORS Errors

**Symptoms**: Frontend shows CORS errors in browser console

**Check**:
- [ ] `ALLOWED_ORIGINS` includes frontend URL
- [ ] Frontend URL uses https:// (not http://)
- [ ] No trailing slash in URL

**Fix**:
Add frontend URL to `ALLOWED_ORIGINS` in HF Space secrets:
```
ALLOWED_ORIGINS=https://your-app.vercel.app,https://graphbug.com
```

---

### Issue: Out of Memory

**Symptoms**: Space crashes, restarts frequently

**Check**:
- [ ] Embedding model too large
- [ ] Too many concurrent requests
- [ ] Memory leaks in code

**Fix**:
- Upgrade to T4 Small (paid tier) for more memory
- Optimize batch processing
- Add request queuing

---

## Support

If issues persist:
1. Check HF Spaces logs
2. Check Neo4j Aura logs
3. Check Qdrant Cloud logs
4. Review `HUGGINGFACE_DEPLOYMENT.md` for detailed troubleshooting
5. Open issue on GitHub with logs

---

**Deployment Date**: _______________
**Deployed By**: _______________
**Space URL**: _______________
**Status**: [ ] Success [ ] Partial [ ] Failed

**Notes**:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
