from neo4j import GraphDatabase
from .logger import setup_logger

logger = setup_logger(__name__)

# HYBRID MAP: Covers Zed Node Types, Standard Tags, AND Ops Tags
HYBRID_MAP = {
    # --- ZED STYLE (Node Types) ---
    # These match the actual grammar structure, regardless of the query file
    "function_definition": "Function",  # Python
    "function_declaration": "Function", # JS/TS/Go
    "method_definition": "Function",    # JS/TS
    "method_declaration": "Function",   # Go/Java
    "class_definition": "Class",        # Python
    "class_declaration": "Class",       # JS/TS/Java
    "interface_declaration": "Interface", # TS/Java
    "struct_item": "Struct",            # Rust
    "function_item": "Function",        # Rust
    "impl_item": "Class",               # Rust

    # --- STANDARD TREE-SITTER TAGS ---
    # These match the capture names in .scm files
    "definition.function": "Function",
    "definition.method": "Function",
    "definition.macro": "Function",
    "definition.entrypoint": "Function",
    "definition.class": "Class",
    "definition.interface": "Interface",
    "definition.module": "Module",
    "definition.import": "Import",
    "definition.implementation": "Class",

    # --- OPS & INFRASTRUCTURE TAGS ---
    "definition.resource": "Resource",    # Terraform/HCL
    "definition.variable": "Variable",    # Bash/Terraform/Docker/Env
    "definition.section": "Section",      # TOML/INI Tables
    "definition.key": "ConfigKey",        # TOML/YAML Keys
    "definition.config": "Config",        # General Configuration
    "definition.base_image": "BaseImage", # Dockerfile
    "definition.instruction": "Instruction", # Dockerfile
    "definition.stage": "Stage",          # Dockerfile
    "definition.target": "Target",        # Makefile
    "definition.script": "Script",        # Vue
    "definition.style": "Style",          # Vue
}

class GraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def process_file_nodes(self, repo_id, file_path, captures, code_bytes):
        nodes_to_create = []
        file_uid = f"{repo_id}::{file_path}"

        for node, capture_name in captures:
            clean_tag = capture_name.strip("@")
            node_type = node.type
            
            # 1. TRY NODE TYPE (Most reliable for Zed/Outline queries)
            node_label = HYBRID_MAP.get(node_type)
            
            # 2. TRY TAG NAME (Reliable for Standard/Manual queries)
            if not node_label:
                node_label = HYBRID_MAP.get(clean_tag)
            
            # 3. FALLBACK HEURISTICS
            if not node_label:
                if "function" in node_type or "method" in node_type:
                    node_label = "Function"
                elif "class" in node_type or "struct" in node_type:
                    node_label = "Class"
                else:
                    # Skip unrelated captures (like comments or noise)
                    continue

            # 4. ROBUST NAME EXTRACTION
            # Sometimes the capture is the whole function body (Zed), 
            # sometimes it is just the name (Standard).
            
            # First, try to find a child field explicitly named "name"
            name_node = node.child_by_field_name("name")
            
            if name_node:
                # We found a specific name node, use it
                name_text = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf8", errors="ignore")
            else:
                # If no "name" field, assume the captured node IS the name 
                # OR fallback to finding the first identifier
                name_text = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
                
                # Cleanup: If name captures multiple lines, it's likely the whole body. 
                # Fallback to "anon" or first line to prevent DB errors.
                if "\n" in name_text:
                    # Try to find an identifier child
                    found_id = False
                    for child in node.children:
                        if "identifier" in child.type or "string_lit" in child.type:
                            name_text = code_bytes[child.start_byte : child.end_byte].decode("utf8", errors="ignore")
                            found_id = True
                            break
                    if not found_id:
                        # Final safety net
                        name_text = f"anon_{node.start_point[0]}"

            # Clean up quotes if it's a string literal (common in JSON/YAML/TF)
            name_text = name_text.strip('"').strip("'")

            # Extract Raw Code (Always the full node)
            raw_code = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
            
            entity_uid = f"{file_uid}::{name_text}"

            nodes_to_create.append({
                "label": node_label,
                "name": name_text,
                "uid": entity_uid,
                "raw_code": raw_code,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            })

        if nodes_to_create:
            self._batch_insert(repo_id, file_uid, file_path, nodes_to_create)

    def _batch_insert(self, repo_id, file_uid, file_path, node_list):
        query = """
        MERGE (f:File {uid: $file_uid})
        SET f.path = $file_path, f.repo_id = $repo_id
        WITH f
        UNWIND $batch AS item
        MERGE (e:Entity {uid: item.uid})
        SET e.name = item.name, e.repo_id = $repo_id, e.file = $file_path,
            e.type = item.label, e.start_line = item.start_line,
            e.end_line = item.end_line, e.raw_code = item.raw_code
        MERGE (f)-[:DEFINES]->(e)
        """
        try:
            with self.driver.session() as session:
                session.run(query, repo_id=repo_id, file_uid=file_uid, file_path=file_path, batch=node_list)
                logger.debug(f"Inserted {len(node_list)} nodes for {file_path}")
        except Exception as e:
            logger.error(f"Graph insert error for {file_path}: {e}")

    def build_dependencies(self, repo_id):
        logger.info(f"Linking dependencies for repo {repo_id}...")
        query = """
        MATCH (source:Entity {repo_id: $repo_id}), (target:Entity {repo_id: $repo_id})
        WHERE source.uid <> target.uid AND source.file <> target.file
        AND source.type = 'Function' 
        AND target.type IN ['Function', 'Class', 'Struct', 'Interface', 'Resource', 'Variable']
        AND size(target.name) > 3
        AND source.raw_code CONTAINS target.name
        MERGE (source)-[:MAY_CALL]->(target)
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id)
                summary = result.consume()
                logger.info(f"Created {summary.counters.relationships_created} dependency links")
        except Exception as e:
            logger.error(f"Graph dependency error: {e}")
    
    def get_dependencies(self, repo_id, file_path, start_line):
        """
        Robust Lookup: Uses File + Line Number to find the exact node 
        in the graph, then fetches what it calls.
        """
        # We use a small range (start_line +/- 1) to handle minor parser discrepancies
        query = """
        MATCH (source:Entity {repo_id: $repo_id, file: $file_path})
        WHERE abs(source.start_line - $start_line) <= 1
        MATCH (source)-[:MAY_CALL]->(target)
        RETURN target.name AS name, target.file AS file, target.raw_code AS code, target.start_line AS start_line
        LIMIT 5
        """
        try:
            with self.driver.session() as session:
                # Pass start_line as integer
                result = session.run(query, repo_id=repo_id, file_path=file_path, start_line=int(start_line))
                dependencies = [
                    {
                        "name": record["name"],
                        "file": record["file"],
                        "code": record["code"],
                        "line": record["start_line"]
                    }
                    for record in result
                ]
                return dependencies
        except Exception as e:
            logger.error(f"Graph retrieval error: {e}")
            return []
    
    def delete_repo(self, repo_id: str):
        """Deletes all nodes and relationships for a specific repo_id."""
        query = "MATCH (n {repo_id: $repo_id}) DETACH DELETE n"
        try:
            with self.driver.session() as session:
                session.run(query, repo_id=repo_id)
                logger.info(f"🗑️  Deleted graph nodes for repo {repo_id}")
        except Exception as e:
            logger.error(f"⚠️ Graph delete failed: {e}")