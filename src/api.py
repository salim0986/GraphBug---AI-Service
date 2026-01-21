from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from .parser import UniversalParser, EXTENSION_MAP
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder
from .analyzer import (
    CodeAnalyzer,
    PRAnalysisRequest,
    FileAnalysisRequest,
    DiffAnalysisRequest,
    PRAnalysisResult,
    FileAnalysisResult
)
from .context_builder import ContextBuilder, PRContext
from .workflow import CodeReviewWorkflow, WorkflowConfig, create_review_workflow
from .logger import setup_logger
from .incremental_ingest import ingest_repo_incremental
from sentence_transformers import SentenceTransformer
from .github_client import GitHubClient, GitHubConfig
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .security_utils import (
    safe_join,
    sanitize_repo_id,
    sanitize_owner_repo,
    sanitize_file_path,
    sanitize_error_message,
    validate_request_size
)
import git
import shutil
import os
import asyncio

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Graph Bug AI Service", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = setup_logger(__name__)

# Configure CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

@app.on_event("startup")
async def startup_event():
    """Log service status on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Graph Bug AI Service Starting...")
    logger.info(f"   Version: 2.0.0")
    logger.info(f"   GitHub Client: {'✅ Ready' if github_client else '❌ Not configured'}")
    logger.info("=" * 80)

# --- CONFIGURATION ---

IGNORE_DIRS = {
    ".git", ".github", ".vscode", ".idea",
    "node_modules", "venv", "env",
    "dist", "build", "out", "target", "bin", "obj",
    "__pycache__", "coverage", "tmp", "temp", "migrations"
}

# Initialize Singletons
parser = UniversalParser()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# FIX: Use localhost for local execution
graph_db = GraphBuilder("neo4j://localhost:7687", ("neo4j", "graphbug123"))
vector_db = VectorBuilder("http://localhost:6333", embed_model)

# Initialize Code Analyzer and Context Builder
code_analyzer = CodeAnalyzer(graph_db, vector_db, parser)
context_builder = ContextBuilder(code_analyzer, graph_db, vector_db)

# Initialize GitHub Client (Phase 5.3)
try:
    github_config = GitHubConfig.from_env()
    github_client = GitHubClient(github_config)
    logger.info("=" * 80)
    logger.info("✅ GitHub Client Initialized Successfully!")
    logger.info(f"   App ID: {github_config.app_id}")
    logger.info(f"   Private key loaded: {len(github_config.private_key)} bytes")
    logger.info("   PR comments will be posted automatically")
    logger.info("=" * 80)
except Exception as e:
    logger.error("=" * 80)
    logger.error(f"❌ GitHub client initialization FAILED: {e}")
    logger.error("   PR comments will NOT be posted!")
    logger.error("   Set GITHUB_APP_ID and GITHUB_PRIVATE_KEY in .env file")
    logger.error("=" * 80)
    github_client = None

# Initialize LangGraph Workflow (Phase 4) with dependencies
review_workflow = create_review_workflow(
    context_builder=context_builder
)

# --- MODELS ---

class RepoRequest(BaseModel):
    repo_url: str = Field(..., max_length=500, pattern=r'^https://github\.com/[\w-]+/[\w.-]+(.git)?$')
    repo_id: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')
    installation_id: str = Field(..., pattern=r'^\d+$', max_length=20)
    incremental: Optional[bool] = False  # Enable incremental ingestion
    last_commit: Optional[str] = Field(None, min_length=7, max_length=40, pattern=r'^[a-fA-F0-9]+$')  # Previous commit SHA for incremental

class QueryRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')
    query: str = Field(..., min_length=1, max_length=1000)

def process_repo_task(req: RepoRequest):
    """The Heavy Worker Function: Clones -> Parses -> Indexes"""
    logger.info(f"Starting processing for {req.repo_url} (ID: {req.repo_id})")
    
    # Sanitize repo_id to prevent path traversal
    try:
        safe_repo_id = sanitize_repo_id(req.repo_id)
    except ValueError as e:
        logger.error(f"Invalid repo_id: {e}")
        return
    
    # Use safe_join to prevent directory traversal
    try:
        local_path = safe_join("./temp_repos", safe_repo_id)
    except ValueError as e:
        logger.error(f"Path traversal attempt detected: {e}")
        return
    
    # 1. Clean & Clone
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    try:
        logger.info(f"Cloning repository from {req.repo_url}")
        git.Repo.clone_from(req.repo_url, local_path)
    except Exception as e:
        logger.error(f"Git Clone Failed: {e}")
        return

    # 2. Prepare Vector DB
    vector_db.ensure_collection()
    
    indexed_count = 0
    file_count = 0

    # 3. Walk Files with IGNORE Logic
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Use Parser's EXTENSION_MAP to check if valid
            filename = os.path.basename(file)
            _, ext = os.path.splitext(filename)
            is_valid = filename in EXTENSION_MAP or ext in EXTENSION_MAP
            
            if not is_valid:
                continue

            try:
                # 4. Parse File
                captures, code_bytes = parser.parse_file(file_path)
                
                if not captures:
                    continue
                
                file_count += 1
                rel_path = os.path.relpath(file_path, local_path)

                # 5. Update Graph (Structure)
                graph_db.process_file_nodes(req.repo_id, rel_path, captures, code_bytes)
                
                # 6. Update Vectors (Semantics) - THE FIX IS HERE
                for node, capture_name in captures:
                    
                    # 1. Line Count Check (The most effective filter)
                    # Skip tiny one-liners (variables, imports, properties)
                    # We want "Logic Chunks", which are usually multi-line.
                    lines_of_code = node.end_point[0] - node.start_point[0]
                    if lines_of_code < 3:
                        continue  # Skip noise

                    # 2. Strict Type Check
                    # Only embed "Callable" logic or "Structural" definitions
                    node_type = node.type
                    ALLOWED_TYPES = [
                        "function_definition", "function_declaration", "function_item",
                        "method_definition", "method_declaration",
                        "class_definition", "class_declaration", 
                        "interface_declaration", "impl_item",
                        # Config files (YAML/TOML) are exceptions, we want their small chunks
                        "table", "block" 
                    ]
                    
                    # Allow if type matches OR if it's a "definition" tag and long enough
                    is_logic_node = node_type in ALLOWED_TYPES
                    is_config_file = ext in [".yaml", ".yml", ".toml", ".tf", ".dockerfile"]
                    
                    if not (is_logic_node or (is_config_file and lines_of_code > 0)):
                        continue

                    try:
                        func_code = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
                        
                        # Extract a cleaner name
                        first_line = func_code.splitlines()[0]
                        func_name = first_line[:100].strip()

                        vector_db.ingest_function_chunk(
                            repo_id=req.repo_id,
                            func_name=func_name,
                            func_code=func_code, # Now it's the full function body
                            file_path=rel_path,
                            start_line=node.start_point[0]
                        )
                        indexed_count += 1
                    except Exception as e:
                        logger.warning(f"Vector Embedding Error: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                continue
    
    # 7. Post-Process Graph (Link dependencies)
    try:
        graph_db.build_dependencies(req.repo_id)
        logger.info(f"Processed {file_count} files, indexed {indexed_count} functions")
        logger.info(f"Successfully finished processing {req.repo_id}")
    except Exception as e:
        logger.error(f"Error building graph dependencies: {e}")
    
    # Cleanup
    try:
        shutil.rmtree(local_path)
    except:
        pass

# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_repo(request: Request, req: RepoRequest, background_tasks: BackgroundTasks):
    """
    Ingest repository with optional incremental mode
    
    Modes:
    - Full ingestion: incremental=False (default) - processes all files
    - Incremental: incremental=True with last_commit - only processes changed files
    
    Performance:
    - Full: ~10-60s depending on repo size
    - Incremental: ~2-10s for typical updates
    """
    if req.incremental and req.last_commit:
        # Incremental mode - much faster!
        background_tasks.add_task(process_repo_incremental_task, req)
        return {
            "status": "queued",
            "repo_id": req.repo_id,
            "mode": "incremental",
            "from_commit": req.last_commit
        }
    else:
        # Full ingestion mode
        background_tasks.add_task(process_repo_task, req)
        return {
            "status": "queued",
            "repo_id": req.repo_id,
            "mode": "full"
        }

async def process_repo_incremental_task(req: RepoRequest):
    """
    High-performance incremental ingestion
    Only processes files that changed since last_commit
    """
    logger.info(f"Starting INCREMENTAL processing for {req.repo_url} (ID: {req.repo_id})")
    logger.info(f"   From commit: {req.last_commit}")
    
    local_path = f"./temp_repos/{req.repo_id}"
    
    try:
        # Clone or pull repository
        if os.path.exists(local_path):
            logger.info(f"Repository exists, pulling latest changes...")
            repo = git.Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
        else:
            logger.info(f"Cloning repository from {req.repo_url}")
            git.Repo.clone_from(req.repo_url, local_path)
        
        # Run incremental ingestion
        stats = await ingest_repo_incremental(
            repo_id=req.repo_id,
            repo_url=req.repo_url,
            local_path=local_path,
            parser=parser,
            graph_db=graph_db,
            vector_db=vector_db,
            ignore_dirs=IGNORE_DIRS,
            last_commit=req.last_commit
        )
        
        logger.info(f"✅ Incremental ingestion complete: {stats}")
        logger.info(f"   Files processed: {stats['files_processed']}")
        logger.info(f"   Nodes added: {stats['nodes_added']}")
        logger.info(f"   Vectors updated: {stats['vectors_updated']}")
        
    except Exception as e:
        logger.error(f"Incremental ingestion failed: {e}", exc_info=True)
    
    # Cleanup (optional - keep for next incremental update)
    # try:
    #     shutil.rmtree(local_path)
    # except:
    #     pass

# --- ENDPOINTS ---

@app.post("/ingest_legacy")
async def ingest_repo_legacy(req: RepoRequest, background_tasks: BackgroundTasks):
    """Legacy full ingestion endpoint (kept for backwards compatibility)"""
    background_tasks.add_task(process_repo_task, req)
    return {"status": "queued", "repo_id": req.repo_id}

@app.post("/query")
async def search_repo(http_request: Request, req: QueryRequest):
    """
    GraphRAG Search:
    1. Vector Search finds the relevant 'entry points'.
    2. Graph Search expands those points to find dependencies.
    """
    # 1. Vector Search (Find top 3 matches)
    results = vector_db.search_similar(req.repo_id, req.query)
    
    data = []
    for hit in results:
        payload = hit.payload
        
        # 2. Graph Expansion (Fetch context)
        # "What does this function call? What does it rely on?"
        dependencies = graph_db.get_dependencies(
            repo_id=req.repo_id, 
            file_path=payload.get("file"), 
            start_line=payload.get("start_line")
        )
        
        data.append({
            "score": hit.score,
            "type": "Primary Match",
            "name": payload.get("name"),
            "file": payload.get("file"),
            "start_line": payload.get("start_line"),
            "code": payload.get("raw_code"),
            "related_dependencies": dependencies  # <--- The Graph Value
        })
    
    return {"results": data}


@app.delete("/repos/{repo_id}")
async def delete_repo_data(repo_id: str):
    """
    Cleanup Endpoint:
    Called when a user uninstalls the app or deletes a repository.
    Wipes data from both Vector DB and Graph DB.
    """
    logger.info(f"Received delete request for repo {repo_id}")
    
    # 1. Delete from Vector DB (Search Index)
    vector_db.delete_repo(repo_id)
    
    # 2. Delete from Graph DB (Structure)
    graph_db.delete_repo(repo_id)
    
    return {"status": "success", "message": f"Data for {repo_id} wiped."}


# ============================================================================
# CODE ANALYSIS ENDPOINTS (Phase 3)
# ============================================================================

@app.post("/analyze/pr", response_model=PRAnalysisResult)
async def analyze_pr(request: PRAnalysisRequest):
    """
    Analyze an entire Pull Request
    
    Returns comprehensive analysis including:
    - Code issues by severity and category
    - Similar code patterns found
    - Related code via graph traversal
    - Metrics for each file
    """
    try:
        logger.info(f"Received PR analysis request for PR #{request.pr_number} in repo {request.repo_id}")
        result = await code_analyzer.analyze_pr(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing PR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file", response_model=FileAnalysisResult)
async def analyze_file(request: FileAnalysisRequest):
    """
    Analyze a single file
    
    Returns:
    - Code issues and smells
    - Similar code patterns
    - Related code dependencies
    - File metrics
    """
    try:
        logger.info(f"Received file analysis request for {request.filename} in repo {request.repo_id}")
        result = await code_analyzer.analyze_file(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/diff", response_model=FileAnalysisResult)
async def analyze_diff(request: DiffAnalysisRequest):
    """
    Analyze a diff/patch
    
    Focuses on changed lines and their context.
    Returns issues found in new code.
    """
    try:
        logger.info(f"Received diff analysis request for {request.filename} in repo {request.repo_id}")
        result = await code_analyzer.analyze_diff(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing diff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "neo4j": "connected" if graph_db else "not configured",
            "qdrant": "connected" if vector_db else "not configured"
        }
    }


# ========================================================================
# WEBHOOK INTEGRATION ENDPOINT (GitHub PR Review)
# ========================================================================

from typing import Any

class PRReviewWebhookRequest(BaseModel):
    """Request from frontend webhook after PR processing"""
    owner: str
    repo: str
    pr_number: int
    installation_id: str
    pull_request_id: str
    context: Any  # Flexible type for PR context from frontend
    gemini_api_key: Optional[str] = None  # User's Gemini API key (BYO system)
    
    class Config:
        extra = "allow"  # Allow extra fields


@app.post("/review")
async def process_pr_review_webhook(http_request: Request, request: PRReviewWebhookRequest, background_tasks: BackgroundTasks):
    """
    Main webhook endpoint for PR review processing
    
    Called by frontend after GitHub webhook processes PR data.
    Orchestrates the complete AI review workflow:
    1. Receives PR context from frontend
    2. Triggers LangGraph workflow
    3. Posts results back to GitHub
    
    Runs as background task to avoid timeout.
    """
    try:
        logger.info(f"🔔 Received review webhook for PR #{request.pr_number} in {request.owner}/{request.repo}")
        logger.info(f"   Installation ID: {request.installation_id}")
        logger.info(f"   Pull Request DB ID: {request.pull_request_id}")
        
        # Validate required fields
        if not request.context:
            raise HTTPException(status_code=400, detail="PR context is required")
        
        # Extract context data
        context = request.context
        files = context.get("files", [])
        
        if not files:
            logger.warning(f"No files found in PR #{request.pr_number} context")
            return {
                "status": "skipped",
                "message": "No files to review",
                "pr_number": request.pr_number,
                "review_id": None
            }
        
        logger.info(f"📝 Processing {len(files)} files from PR #{request.pr_number}")
        
        # Build repo_id in format expected by internal services
        repo_id = f"{request.owner}/{request.repo}"
        
        # Extract branch refs from context
        base_ref = context.get("base_ref", "main")
        head_ref = context.get("head_ref", "unknown")
        
        # Queue the review workflow as a background task
        background_tasks.add_task(
            execute_review_task,
            pr_number=request.pr_number,
            repo_id=repo_id,
            pr_title=context.get("title", ""),
            pr_description=context.get("description"),
            files=files,
            base_ref=base_ref,
            head_ref=head_ref,
            installation_id=request.installation_id,
            pull_request_db_id=request.pull_request_id,
            owner=request.owner,
            repo=request.repo,
            gemini_api_key=request.gemini_api_key
        )
        
        # Return immediately with queued status
        review_id = f"review-{request.pull_request_id}"
        logger.info(f"✅ Review workflow queued: {review_id}")
        
        return {
            "status": "queued",
            "message": f"Review workflow started for PR #{request.pr_number}",
            "pr_number": request.pr_number,
            "review_id": review_id,
            "files_count": len(files)
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing review webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# REVIEW FORMATTING HELPER
# ========================================================================

def format_review_for_github(
    pr_number: int,
    pr_title: str,
    overall_summary: str,
    file_reviews: dict,
    status: str
) -> str:
    """
    Format review for GitHub PR comment with rich markdown
    
    Features:
    - Severity badges (shields.io)
    - Collapsible sections for long content
    - Syntax-highlighted code blocks
    - Color-coded severity indicators
    
    Args:
        pr_number: PR number
        pr_title: PR title
        overall_summary: Generated review summary
        file_reviews: Dict of file-level reviews
        status: Workflow status
    
    Returns:
        Formatted markdown for GitHub comment
    """
    parts = []
    
    # Header with badges
    parts.append(f"# 🤖 AI Code Review - PR #{pr_number}")
    parts.append(f"**{pr_title}**")
    parts.append("")
    
    # Status badge
    if status == "completed":
        parts.append("![Status](https://img.shields.io/badge/status-completed-success)")
    else:
        parts.append("![Status](https://img.shields.io/badge/status-in__progress-yellow)")
    
    # Count severity levels in summary
    critical_count = overall_summary.lower().count("critical")
    high_count = overall_summary.lower().count("high priority") + overall_summary.lower().count("high:")
    
    if critical_count > 0:
        parts.append("![Critical](https://img.shields.io/badge/critical-" + str(critical_count) + "-critical)")
    if high_count > 0:
        parts.append("![High](https://img.shields.io/badge/high-" + str(high_count) + "-orange)")
    
    parts.append("")
    
    # Overall summary with collapsible if long
    parts.append("## 📋 Overall Review")
    parts.append("")
    
    if len(overall_summary) > 3000:
        parts.append("<details>")
        parts.append("<summary><b>Click to expand full review</b></summary>")
        parts.append("")
        parts.append(overall_summary)
        parts.append("")
        parts.append("</details>")
    else:
        parts.append(overall_summary)
    
    parts.append("")
    
    # File-level reviews with rich formatting
    if file_reviews:
        parts.append("---")
        parts.append("")
        parts.append("## 📁 File-by-File Analysis")
        parts.append("")
        
        # Sort files by issues_found (most issues first)
        sorted_files = sorted(
            file_reviews.items(),
            key=lambda x: x[1].get("issues_found", 0),
            reverse=True
        )
        
        for filename, review in sorted_files[:8]:  # Show top 8 files
            if review.get("error"):
                continue
            
            issues_found = review.get("issues_found", 0)
            
            # File header with badge
            parts.append(f"### 📄 `{filename}`")
            parts.append("")
            
            # Language and severity badges
            badge_parts = []
            if review.get("language"):
                badge_parts.append(f"![{review['language']}](https://img.shields.io/badge/language-{review['language']}-blue)")
            
            if issues_found > 0:
                if issues_found >= 3:
                    badge_parts.append(f"![Issues](https://img.shields.io/badge/issues-{issues_found}-red)")
                elif issues_found >= 1:
                    badge_parts.append(f"![Issues](https://img.shields.io/badge/issues-{issues_found}-orange)")
            else:
                badge_parts.append("![Clean](https://img.shields.io/badge/issues-0-success)")
            
            if badge_parts:
                parts.append(" ".join(badge_parts))
                parts.append("")
            
            summary = review.get("summary", "")
            
            # Use collapsible for long reviews
            if len(summary) > 800:
                parts.append("<details>")
                parts.append("<summary><b>View detailed review</b></summary>")
                parts.append("")
                parts.append(summary)
                parts.append("")
                parts.append("</details>")
            else:
                parts.append(summary)
            
            parts.append("")
        
        if len(file_reviews) > 8:
            parts.append(f"<details>")
            parts.append(f"<summary>📋 <i>... and {len(file_reviews) - 8} more files reviewed</i></summary>")
            parts.append("")
            for filename in list(file_reviews.keys())[8:]:
                parts.append(f"- `{filename}`")
            parts.append("")
            parts.append("</details>")
    
    # Footer with GraphRAG branding
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("🔍 **Powered by GraphRAG Technology**")
    parts.append("")
    parts.append("*This review uses Graph Database + Vector Search to find similar code patterns, ")
    parts.append("dependencies, and architectural insights across your codebase.*")
    parts.append("")
    parts.append("**Generated by [Graph Bug AI Reviewer](https://github.com) • Google Gemini 2.5**")
    
    return "\n".join(parts)


async def execute_review_task(
    pr_number: int,
    repo_id: str,
    pr_title: str,
    pr_description: Optional[str],
    files: List[dict],
    base_ref: str,
    head_ref: str,
    installation_id: str,
    pull_request_db_id: str,
    owner: str,
    repo: str,
    gemini_api_key: Optional[str] = None
):
    """
    Background task that executes the complete review workflow
    
    This is where the actual AI review happens:
    1. Run LangGraph workflow
    2. Generate review using Gemini
    3. Post results to GitHub PR
    """
    try:
        logger.info(f"🚀 Starting review workflow for PR #{pr_number} in {repo_id}")
        
        # Create user-specific workflow with their Gemini API key
        if gemini_api_key:
            from .gemini_client import create_gemini_client
            user_gemini_client = create_gemini_client(api_key=gemini_api_key)
            user_workflow = create_review_workflow(
                context_builder=context_builder,
                gemini_client=user_gemini_client
            )
            logger.info("✅ Using user's Gemini API key for review")
        else:
            user_workflow = review_workflow
            logger.info("⚠️ Using default Gemini API key (fallback)")
        
        # Execute the LangGraph workflow
        final_state = await user_workflow.run_review(
            pr_number=pr_number,
            repo_id=repo_id,
            pr_title=pr_title,
            pr_description=pr_description,
            files=files,
            base_ref=base_ref,
            head_ref=head_ref
        )
        
        logger.info(f"✅ Review workflow completed for PR #{pr_number}")
        logger.info(f"   Status: {final_state.get('status')}")
        logger.info(f"   Files processed: {final_state.get('processed_files', 0)}/{final_state.get('total_files', 0)}")
        
        # Post review to GitHub PR
        overall_summary = final_state.get("overall_summary", "")
        
        if overall_summary and github_client:
            try:
                logger.info("=" * 80)
                logger.info(f"📤 Posting review to GitHub PR #{pr_number}...")
                logger.info(f"   Repo: {repo_id}")
                logger.info(f"   Installation ID: {installation_id}")
                logger.info(f"   Review length: {len(overall_summary)} chars")
                
                # Format review for GitHub
                review_body = format_review_for_github(
                    pr_number=pr_number,
                    pr_title=pr_title,
                    overall_summary=overall_summary,
                    file_reviews=final_state.get("file_reviews", {}),
                    status=final_state.get("status", "completed")
                )
                
                logger.info(f"   Formatted review: {len(review_body)} chars")
                
                # Post review comment
                result = await github_client.post_review_comment(
                    repo_full_name=repo_id,
                    pr_number=pr_number,
                    body=review_body,
                    installation_id=int(installation_id),
                    event="COMMENT"
                )
                
                logger.info("✅ Review posted successfully to GitHub!")
                logger.info(f"   Review ID: {result['id']}")
                logger.info(f"   Review URL: {result['html_url']}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"❌ Failed to post review to GitHub: {e}")
                logger.error(f"   Review was generated but not posted")
                logger.error(f"   Check GitHub App permissions (pull_requests: write)")
                logger.error("=" * 80, exc_info=True)
        elif not overall_summary:
            logger.warning(f"⚠️ No review summary generated, skipping GitHub comment")
        elif not github_client:
            logger.warning(f"⚠️ GitHub client not available, skipping comment posting")
        
        # For now, just log the results
        review_summary = final_state.get("review_summary", {})
        logger.info(f"📊 Review Summary:")
        logger.info(f"   Issues found: {review_summary.get('total_issues', 0)}")
        logger.info(f"   Suggestions: {review_summary.get('total_suggestions', 0)}")
        
        logger.info(f"✨ Review task completed for PR #{pr_number}")
        
    except Exception as e:
        logger.error(f"❌ Review workflow failed for PR #{pr_number}: {e}", exc_info=True)
        # TODO: Update PR status in frontend database with error
        # TODO: Optionally post error comment to GitHub PR


# ========================================================================
# VECTOR SEARCH ENDPOINTS (Phase 3.3)
# ========================================================================

class SemanticSearchRequest(BaseModel):
    repo_id: str
    query: str
    limit: int = 10

class SimilarCodeRequest(BaseModel):
    repo_id: str
    code_snippet: str
    limit: int = 5
    min_score: float = 0.7

class RepositoryStatsResponse(BaseModel):
    total_snippets: int
    by_type: dict
    total_files: int
    top_files: List[tuple]


@app.post("/search/semantic")
async def semantic_search(http_request: Request, request: SemanticSearchRequest):
    """
    Natural language code search
    
    Examples:
    - "functions that handle user authentication"
    - "code that processes payment transactions"
    - "error handling logic"
    
    Returns relevant code snippets based on semantic similarity.
    """
    try:
        logger.info(f"Semantic search for repo {request.repo_id}: '{request.query}'")
        results = vector_db.semantic_code_search(
            repo_id=request.repo_id,
            natural_language_query=request.query,
            limit=request.limit
        )
        return {"query": request.query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/similar")
async def find_similar_code(http_request: Request, request: SimilarCodeRequest):
    """
    Find code similar to a given snippet
    
    Useful for:
    - Detecting duplicate code
    - Finding similar implementations
    - Code pattern analysis
    
    Returns code snippets with similarity scores.
    """
    try:
        logger.info(f"Finding similar code in repo {request.repo_id}")
        results = vector_db.search_similar_code(
            repo_id=request.repo_id,
            code_snippet=request.code_snippet,
            limit=request.limit,
            min_score=request.min_score
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Similar code search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/duplicates")
async def find_duplicates(http_request: Request, request: SimilarCodeRequest):
    """
    Find duplicate code (high similarity threshold: 90%+)
    
    Helps identify:
    - Copy-pasted code
    - Refactoring opportunities
    - Code that should be extracted into shared functions
    """
    try:
        logger.info(f"Finding duplicate code in repo {request.repo_id}")
        # Override min_score to 0.9 for duplicates
        results = vector_db.find_duplicate_code(
            repo_id=request.repo_id,
            code_snippet=request.code_snippet,
            limit=request.limit,
            threshold=0.9
        )
        return {
            "results": results,
            "count": len(results),
            "threshold": 0.9,
            "message": "High similarity matches (90%+) - potential duplicates"
        }
    except Exception as e:
        logger.error(f"Duplicate search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repos/{repo_id}/stats")
async def get_repository_statistics(repo_id: str):
    """
    Get statistics about indexed code in repository
    
    Returns:
    - Total code snippets indexed
    - Breakdown by type (functions, classes, etc.)
    - Total files indexed
    - Top files by snippet count
    """
    try:
        logger.info(f"Getting stats for repo {repo_id}")
        stats = vector_db.get_repository_stats(repo_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting repository stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CONTEXT BUILDER ENDPOINT (Phase 3.5)
# ========================================================================

class PRContextRequest(BaseModel):
    pr_number: int
    repo_id: str
    title: str
    description: Optional[str] = None
    files: List[dict]  # List of FileChange-like dicts


@app.post("/context/pr", response_model=PRContext)
async def build_pr_context(request: PRContextRequest):
    """
    Build comprehensive context for PR review
    
    Combines:
    - Code analysis (issues, patterns, complexity)
    - Graph queries (dependencies, impact, coupling)
    - Vector search (similar code, duplicates)
    
    Returns unified context ready for LangGraph orchestration.
    
    This is the main entry point for the review pipeline.
    """
    try:
        logger.info(f"Building PR context for PR #{request.pr_number} in repo {request.repo_id}")
        
        # Convert dict files to FileChange objects
        from .analyzer import FileChange
        file_changes = [
            FileChange(
                filename=f.get("filename", ""),
                status=f.get("status", "modified"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                changes=f.get("changes", 0),
                patch=f.get("patch"),
                language=f.get("language")
            )
            for f in request.files
        ]
        
        # Build context
        pr_context = await context_builder.build_pr_context(
            pr_number=request.pr_number,
            repo_id=request.repo_id,
            title=request.title,
            description=request.description,
            files=file_changes
        )
        
        # Log summary
        summary = context_builder.get_context_summary(pr_context)
        logger.info(f"PR context built:\n{summary}")
        
        return pr_context
        
    except Exception as e:
        logger.error(f"Error building PR context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/pr/{pr_number}/summary")
async def get_pr_context_summary(pr_number: int, repo_id: str):
    """
    Get a text summary of PR context
    
    Useful for quick overview without full context object.
    """
    try:
        # This would typically fetch from cache or rebuild
        # For now, return a placeholder
        return {
            "pr_number": pr_number,
            "repo_id": repo_id,
            "message": "Summary endpoint - implement caching in production"
        }
    except Exception as e:
        logger.error(f"Error getting PR summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# LANGGRAPH WORKFLOW ENDPOINTS (Phase 4.1)
# ========================================================================

class WorkflowRequest(BaseModel):
    pr_number: int
    repo_id: str
    pr_title: str
    pr_description: Optional[str] = None
    files: List[dict]


@app.post("/workflow/review")
async def execute_review_workflow(http_request: Request, request: WorkflowRequest):
    """
    Execute the complete LangGraph review workflow
    
    This orchestrates:
    1. Context building (Phase 3)
    2. Review strategy routing
    3. Gemini-powered review generation
    4. Result aggregation
    
    Returns final review state.
    """
    try:
        logger.info(f"Starting workflow for PR #{request.pr_number} in {request.repo_id}")
        
        # Execute workflow
        final_state = await review_workflow.run_review(
            pr_number=request.pr_number,
            repo_id=request.repo_id,
            pr_title=request.pr_title,
            pr_description=request.pr_description,
            files=request.files
        )
        
        return {
            "status": "success",
            "pr_number": request.pr_number,
            "workflow_state": final_state
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/review/stream")
async def stream_review_workflow(request: WorkflowRequest):
    """
    Stream the review workflow execution
    
    Returns Server-Sent Events (SSE) with state updates
    as the workflow progresses through each node.
    """
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        async def event_generator():
            logger.info(f"Starting streaming workflow for PR #{request.pr_number}")
            
            async for state in review_workflow.stream_review(
                pr_number=request.pr_number,
                repo_id=request.repo_id,
                pr_title=request.pr_title,
                pr_description=request.pr_description,
                files=request.files
            ):
                # Send state update as SSE
                event_data = json.dumps({
                    "pr_number": request.pr_number,
                    "status": state.get("status"),
                    "current_step": state.get("current_step"),
                    "processed_files": state.get("processed_files"),
                    "total_files": state.get("total_files")
                })
                yield f"data: {event_data}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Streaming workflow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/status/{pr_number}")
async def get_workflow_status(pr_number: int, repo_id: str):
    """
    Get current status of a workflow execution
    
    In production, this would query a state store.
    For now, returns placeholder.
    """
    return {
        "pr_number": pr_number,
        "repo_id": repo_id,
        "message": "Status tracking - implement with Redis/DB in production"
    }
