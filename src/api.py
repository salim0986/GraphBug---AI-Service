from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
from .parser import UniversalParser, EXTENSION_MAP
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder
from .logger import setup_logger
from sentence_transformers import SentenceTransformer
import git
import shutil
import os

app = FastAPI()
logger = setup_logger(__name__)

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

# --- MODELS ---

class RepoRequest(BaseModel):
    repo_url: str
    repo_id: str
    installation_id: str

class QueryRequest(BaseModel):
    repo_id: str
    query: str

def process_repo_task(req: RepoRequest):
    """The Heavy Worker Function: Clones -> Parses -> Indexes"""
    logger.info(f"Starting processing for {req.repo_url} (ID: {req.repo_id})")
    
    local_path = f"./temp_repos/{req.repo_id}"
    
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
async def ingest_repo(req: RepoRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_repo_task, req)
    return {"status": "queued", "repo_id": req.repo_id}

@app.post("/query")
async def search_repo(req: QueryRequest):
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