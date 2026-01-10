import numpy as np
from tree_sitter_languages import get_language, get_parser
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import sys
import os

# Add src to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, EMBEDDING_MODEL

# --- CONFIGURATION ---
model = SentenceTransformer(EMBEDDING_MODEL)

# 1. INPUT: The Raw Code
# Imagine this came from a GitHub Webhook
source_code = """
function calculateTax(amount: number) {
    return amount * 1.18;
}
"""

def run_pipeline():
    print("🚀 Starting Ingestion Pipeline...")

    # --- PART A: THE PARSER (Tree-sitter) ---
    print("\n1️⃣  PARSING CODE...")
    language = get_language('typescript')
    parser = get_parser('typescript')
    tree = parser.parse(bytes(source_code, "utf8"))

    # Simple Query to find the function name
    # (We are borrowing the query style from nvim-treesitter)
    query_scm = """
    (function_declaration
      name: (identifier) @func_name)
    """
    query = language.query(query_scm)
    captures = query.captures(tree.root_node)
    
    # Extract the function name string
    func_name_node = captures[0][0] # Get the first match
    func_name = source_code[func_name_node.start_byte : func_name_node.end_byte]
    
    print(f"   -> Found Function: '{func_name}'")

    # --- PART B: THE GRAPH DB (Neo4j) ---
    print("\n2️⃣  UPDATING GRAPH MEMORY (Neo4j)...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def create_node_tx(tx, name):
        # Cypher Query: MERGE ensures we don't create duplicates
        query = "MERGE (f:Function {name: $name}) RETURN elementId(f)"
        result = tx.run(query, name=name)
        return result.single()[0]

    with driver.session() as session:
        node_id = session.execute_write(create_node_tx, func_name)
        print(f"   -> Created Node (:Function {{name: '{func_name}'}})")
    
    driver.close()

    # --- PART C: THE VECTOR DB (Qdrant) ---
    print("\n3️⃣  UPDATING INTUITIVE MEMORY (Qdrant)...")
    client = QdrantClient(url=QDRANT_URL)
    
    collection_name = "code_snippets"
    
    # Initialize collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
        print("   -> Created new 'code_snippets' collection")

    print("   -> Generating Real Vector (using all-MiniLM-L6-v2)...")
    real_vector = model.encode(source_code).tolist()

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=1,  # In prod, use a UUID
                vector=real_vector,
                payload={"name": func_name, "raw_code": source_code}
            )
        ]
    )
    print(f"   -> Stored Vector for '{func_name}'")

    print("\n✅ PIPELINE SUCCESS! The brain has been updated.")

if __name__ == "__main__":
    run_pipeline()