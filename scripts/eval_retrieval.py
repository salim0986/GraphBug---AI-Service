import sys
import os
import argparse
from sentence_transformers import SentenceTransformer

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_builder import VectorBuilder

def run_eval(repo_id: str, model_name: str, use_reranker: bool):
    print(f"=== Running Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Reranker: {'Enabled' if use_reranker else 'Disabled'}")
    
    # Initialize VectorBuilder (note: assumes Qdrant is running and populated)
    embed_model = SentenceTransformer(model_name, trust_remote_code=True)
    vb = VectorBuilder(url="http://localhost:6333", model=embed_model, use_reranker=use_reranker)
    
    # Fixed query set
    queries = [
        "function that parses files",
        "semantic search for code",
        "building a graph of dependencies",
        "incremental ingestion of changed files",
        "rate limiting mechanism"
    ]
    
    results = []
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        matches = vb.semantic_code_search(repo_id=repo_id, natural_language_query=q, limit=5)
        
        if not matches:
            print("  No matches found (Is the repository ingested?)")
            continue
            
        for i, m in enumerate(matches):
            print(f"  {i+1}. {m.get('name', 'unknown')} ({m.get('file', 'unknown')}) - Score: {m.get('relevance', 0):.4f}")
            
    print("\nEvaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate code retrieval before/after M2 changes.")
    parser.add_argument("--repo-id", type=str, default="test-repo", help="Repository ID in Qdrant")
    parser.add_argument("--baseline", action="store_true", help="Run with all-MiniLM-L6-v2 and no reranker (M1 baseline)")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Pre-M2 configuration
        run_eval(args.repo_id, "all-MiniLM-L6-v2", use_reranker=False)
    else:
        # Post-M2 configuration
        run_eval(args.repo_id, "jina-embeddings-v2-base-code", use_reranker=True)
