from neo4j import GraphDatabase
from .logger import setup_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .parser import ParseReport

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
    "internal_module": "Module",        # JS/TS (namespace/module declarations)
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
    def __init__(self, uri, auth, max_retries=3, retry_delay=2.0):
        """Initialize GraphBuilder with connection retry logic
        
        Args:
            uri: Neo4j connection URI
            auth: (username, password) tuple
            max_retries: Maximum connection retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.uri = uri
        self.auth = auth
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Connect to Neo4j with exponential backoff retry"""
        import time
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to Neo4j at {self.uri} (attempt {attempt + 1}/{self.max_retries})")
                driver = GraphDatabase.driver(self.uri, auth=self.auth)
                
                # Verify connectivity
                with driver.session() as session:
                    session.run("RETURN 1")
                
                logger.info(f"✅ Neo4j connection established successfully")
                
                # Create indexes for performance
                self._ensure_indexes(driver)
                
                return driver
                
            except Exception as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to connect to Neo4j after {self.max_retries} attempts")
                    logger.error(f"   URI: {self.uri}")
                    logger.error(f"   Error: {e}")
                    raise ConnectionError(f"Cannot connect to Neo4j: {e}")
    
    def _ensure_indexes(self, driver):
        """Create Neo4j indexes for performance on large repositories"""
        indexes = [
            "CREATE INDEX entity_repo_id IF NOT EXISTS FOR (e:Entity) ON (e.repo_id)",
            "CREATE INDEX entity_uid IF NOT EXISTS FOR (e:Entity) ON (e.uid)",
            "CREATE INDEX file_repo_id IF NOT EXISTS FOR (f:File) ON (f.repo_id)",
            "CREATE INDEX file_uid IF NOT EXISTS FOR (f:File) ON (f.uid)",
            # M4: indexes needed for fast name-based call/inherit resolution
            "CREATE INDEX entity_name_repo IF NOT EXISTS FOR (e:Entity) ON (e.repo_id, e.name)",
            "CREATE INDEX entity_name_file IF NOT EXISTS FOR (e:Entity) ON (e.file, e.name)",
            "CREATE INDEX external_uid IF NOT EXISTS FOR (e:External) ON (e.uid)",
            "CREATE INDEX module_uid IF NOT EXISTS FOR (m:Module) ON (m.uid)",
        ]
        
        try:
            with driver.session() as session:
                for index_query in indexes:
                    session.run(index_query)
                logger.info("✓ Neo4j indexes ensured")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")

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

    # ========================================================================
    # M4 — AST-driven relationship edges
    # ========================================================================

    def process_relationships(self, repo_id: str, file_path: str, report) -> dict:
        """
        Ingest typed relationship records from a ParseReport into Neo4j.
        Creates CALLS, IMPORTS, and INHERITS edges driven by M1 AST output.

        Args:
            repo_id:   Repository identifier.
            file_path: Relative file path (used to locate Entity nodes by file).
            report:    ParseReport from UniversalParser.extract_relationships().

        Returns:
            Stats dict: calls_linked, calls_external, imports_linked,
                        inheritances_linked, inheritances_external.
        """
        stats = {
            "calls_linked": 0,
            "calls_external": 0,
            "imports_linked": 0,
            "inheritances_linked": 0,
            "inheritances_external": 0,
        }
        file_uid = f"{repo_id}::{file_path}"
        try:
            if report.calls:
                c = self._link_calls(repo_id, file_path, report.calls)
                stats["calls_linked"] += c["calls_linked"]
                stats["calls_external"] += c["calls_external"]
            if report.imports:
                stats["imports_linked"] += self._link_imports(repo_id, file_uid, report.imports)
            if report.inheritances:
                c = self._link_inheritances(repo_id, report.inheritances)
                stats["inheritances_linked"] += c["inheritances_linked"]
                stats["inheritances_external"] += c["inheritances_external"]
        except Exception as e:
            logger.error(f"[M4] process_relationships error for {file_path}: {e}")
        logger.debug(
            f"[M4] {file_path}: calls={stats['calls_linked']}+{stats['calls_external']}ext "
            f"imports={stats['imports_linked']} inherit={stats['inheritances_linked']}+{stats['inheritances_external']}ext"
        )
        return stats

    # Maximum rows per UNWIND statement to avoid Neo4j OOM on large files.
    _CYPHER_BATCH = 500

    def _run_batched(self, session, query: str, fixed_params: dict, batch: list) -> int:
        """Run a Cypher query in sub-batches of _CYPHER_BATCH and sum the 'n' counter."""
        total = 0
        for i in range(0, len(batch), self._CYPHER_BATCH):
            sub = batch[i : i + self._CYPHER_BATCH]
            rec = session.run(query, **fixed_params, batch=sub).single()
            total += rec["n"] if rec else 0
        return total

    def _link_calls(self, repo_id: str, file_path: str, calls) -> dict:
        """
        Batch-create CALLS edges with three prioritised passes:

        Pass 1 — same-file resolution: callee defined in the same file as caller.
                  Most precise; avoids name-collision false positives entirely.
        Pass 2 — cross-file unambiguous resolution: callee exists in the repo,
                  but ONLY if exactly one Entity carries that name (to prevent
                  O(N) false edges for common identifiers like 'process', 'init').
        Pass 3 — external stub: callee not found anywhere in this repo →
                  create an :External node so the edge is not silently dropped.

        Calls whose caller is "module_level" are excluded: no Entity node named
        "module_level" exists in Neo4j, so they would always produce 0 matches.
        """
        batch = [
            {"caller_name": c.caller, "callee_name": c.callee, "line": c.line}
            for c in calls
            # module_level is a synthetic sentinel, not a real Entity in the graph
            if c.caller and c.callee and c.caller != "module_level"
        ]
        if not batch:
            return {"calls_linked": 0, "calls_external": 0}

        # Pass 1: same-file callee (precise, no name-collision risk)
        q_same_file = """
        UNWIND $batch AS c
        MATCH (caller:Entity {repo_id: $repo_id, file: $file_path, name: c.caller_name})
        MATCH (callee:Entity {repo_id: $repo_id, file: $file_path, name: c.callee_name})
        WHERE caller.uid <> callee.uid
        MERGE (caller)-[r:CALLS]->(callee)
        ON CREATE SET r.line = c.line, r.file = $file_path
        RETURN count(r) AS n
        """

        # Pass 2: cross-file, but only when the name is unambiguous (exactly 1 match)
        q_cross_file = """
        UNWIND $batch AS c
        MATCH (caller:Entity {repo_id: $repo_id, file: $file_path, name: c.caller_name})
        WHERE NOT EXISTS {
            MATCH (:Entity {repo_id: $repo_id, file: $file_path, name: c.callee_name})
        }
        OPTIONAL MATCH (callee:Entity {repo_id: $repo_id, name: c.callee_name})
        WHERE caller.uid <> callee.uid
        WITH caller, c, collect(callee) AS callees
        WHERE size(callees) = 1
        WITH caller, c, callees[0] AS callee
        MERGE (caller)-[r:CALLS]->(callee)
        ON CREATE SET r.line = c.line, r.file = $file_path
        RETURN count(r) AS n
        """

        # Pass 3: external stub for callees not found anywhere in this repo
        q_external = """
        UNWIND $batch AS c
        MATCH (caller:Entity {repo_id: $repo_id, file: $file_path, name: c.caller_name})
        WHERE NOT EXISTS {
            MATCH (:Entity {repo_id: $repo_id, name: c.callee_name})
        }
        MERGE (ext:External {uid: $repo_id + '::ext::' + c.callee_name})
        ON CREATE SET ext.name = c.callee_name, ext.repo_id = $repo_id
        MERGE (caller)-[r:CALLS]->(ext)
        ON CREATE SET r.line = c.line, r.file = $file_path, r.external = true
        RETURN count(r) AS n
        """

        linked = 0
        external = 0
        params = {"repo_id": repo_id, "file_path": file_path}
        try:
            with self.driver.session() as session:
                linked += self._run_batched(session, q_same_file, params, batch)
                linked += self._run_batched(session, q_cross_file, params, batch)
                external = self._run_batched(session, q_external, params, batch)
        except Exception as e:
            logger.error(f"[M4] _link_calls error for {file_path}: {e}")
        return {"calls_linked": linked, "calls_external": external}

    def _link_imports(self, repo_id: str, file_uid: str, imports) -> int:
        """Create (File)-[:IMPORTS]->(Module) edges. Module nodes keyed by repo+name."""
        batch = [
            {"module": imp.module, "alias": imp.alias or "", "line": imp.line}
            for imp in imports
            if imp.module
        ]
        if not batch:
            return 0

        q = """
        MATCH (f:File {uid: $file_uid})
        UNWIND $batch AS imp
        MERGE (m:Module {uid: $repo_id + '::mod::' + imp.module})
        ON CREATE SET m.name = imp.module, m.repo_id = $repo_id
        MERGE (f)-[r:IMPORTS]->(m)
        ON CREATE SET r.line = imp.line, r.alias = imp.alias
        RETURN count(r) AS n
        """
        try:
            with self.driver.session() as session:
                return self._run_batched(session, q, {"file_uid": file_uid, "repo_id": repo_id}, batch)
        except Exception as e:
            logger.error(f"[M4] _link_imports error: {e}")
            return 0

    def _link_inheritances(self, repo_id: str, inheritances) -> dict:
        """
        Batch-create INHERITS edges.
        Pass 1: parent resolves to existing Entity in repo.
        Pass 2: parent unknown — stub as :External node.
        """
        batch = [
            {"child_name": inh.child, "parent_name": inh.parent, "line": inh.line}
            for inh in inheritances
            if inh.child and inh.parent
        ]
        if not batch:
            return {"inheritances_linked": 0, "inheritances_external": 0}

        q_resolved = """
        UNWIND $batch AS inh
        MATCH (child:Entity {repo_id: $repo_id, name: inh.child_name})
        MATCH (parent:Entity {repo_id: $repo_id, name: inh.parent_name})
        WHERE child.uid <> parent.uid
        MERGE (child)-[r:INHERITS]->(parent)
        ON CREATE SET r.line = inh.line
        RETURN count(r) AS n
        """

        q_external = """
        UNWIND $batch AS inh
        MATCH (child:Entity {repo_id: $repo_id, name: inh.child_name})
        WHERE NOT EXISTS {
            MATCH (:Entity {repo_id: $repo_id, name: inh.parent_name})
        }
        MERGE (ext:External {uid: $repo_id + '::ext::' + inh.parent_name})
        ON CREATE SET ext.name = inh.parent_name, ext.repo_id = $repo_id
        MERGE (child)-[r:INHERITS]->(ext)
        ON CREATE SET r.line = inh.line, r.external = true
        RETURN count(r) AS n
        """

        linked = 0
        external = 0
        params = {"repo_id": repo_id}
        try:
            with self.driver.session() as session:
                linked = self._run_batched(session, q_resolved, params, batch)
                external = self._run_batched(session, q_external, params, batch)
        except Exception as e:
            logger.error(f"[M4] _link_inheritances error: {e}")
        return {"inheritances_linked": linked, "inheritances_external": external}

    def get_dependencies(self, repo_id, file_path, start_line):
        """
        Robust Lookup: Uses File + Line Number to find the exact node 
        in the graph, then fetches what it calls.
        """
        # We use a small range (start_line +/- 1) to handle minor parser discrepancies
        query = """
        MATCH (source:Entity {repo_id: $repo_id, file: $file_path})
        WHERE abs(source.start_line - $start_line) <= 1
        MATCH (source)-[:CALLS|MAY_CALL]->(target)
        RETURN DISTINCT target.name AS name, target.file AS file, target.raw_code AS code, target.start_line AS start_line
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
                result = session.run(query, repo_id=repo_id)
                summary = result.consume()
                deleted_count = summary.counters.nodes_deleted
                logger.info(f"🗑️  Deleted {deleted_count} graph nodes for repo {repo_id}")
                return deleted_count
        except Exception as e:
            logger.error(f"⚠️ Graph delete failed: {e}")
            return 0
    
    def get_repo_node_count(self, repo_id: str) -> int:
        """Get count of nodes for a given repo_id."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n {repo_id: $repo_id}) RETURN count(n) as count", repo_id=repo_id)
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.error(f"Error counting nodes: {e}")
            return 0
    
    def delete_file(self, file_uid: str):
        """
        Delete all entities for a specific file (for incremental updates)
        
        Args:
            file_uid: File unique identifier (format: repo_id::file_path)
        """
        query = """
        MATCH (f:File {uid: $file_uid})-[:DEFINES]->(e:Entity)
        DETACH DELETE e, f
        """
        try:
            with self.driver.session() as session:
                session.run(query, file_uid=file_uid)
                logger.debug(f"Deleted graph nodes for file {file_uid}")
        except Exception as e:
            logger.error(f"Graph file delete failed: {e}")
    
    # ========================================================================
    # ADVANCED GRAPH QUERIES FOR CODE REVIEW (Phase 3.2)
    # ========================================================================
    
    # ========================================================================
    # M5 — Multi-hop graph queries & blast-radius analysis
    # ========================================================================

    def find_transitive_callers(
        self,
        repo_id: str,
        entity_name: str,
        max_depth: int = 4,
    ) -> list:
        """
        Multi-hop backward reachability: every Entity that transitively calls
        `entity_name`, up to `max_depth` CALLS hops away.

        Results are ordered by hop-depth (direct callers first) then file.
        Capped at 50 results to keep query time bounded on large graphs.
        max_depth is interpolated into the query string (it is always an int
        from internal code — no injection risk).
        """
        if not entity_name or max_depth < 1:
            return []
        query = f"""
        MATCH path = (caller:Entity {{repo_id: $repo_id}})
                     -[:CALLS|MAY_CALL*1..{max_depth}]->
                     (target:Entity {{repo_id: $repo_id, name: $entity_name}})
        RETURN DISTINCT
            caller.name       AS name,
            caller.file       AS file,
            caller.type       AS type,
            caller.start_line AS line,
            min(length(path)) AS depth
        ORDER BY depth, caller.file
        LIMIT 50
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, entity_name=entity_name)
                return [
                    {
                        "name": r["name"],
                        "file": r["file"],
                        "type": r["type"],
                        "line": r["line"],
                        "depth": r["depth"],
                    }
                    for r in result
                ]
        except Exception as e:
            logger.error(f"[M5] find_transitive_callers error for {entity_name}: {e}")
            return []

    def find_blast_radius(self, repo_id: str, entity_name: str) -> dict:
        """
        Compute the blast radius for a changed entity.

        Returns:
          affected_functions       — transitive callers (list of dicts)
          affected_files           — unique files containing a caller (sorted)
          untested_callers         — callers whose file path contains no "test" marker
          total_affected_functions — count
          total_affected_files     — count
        """
        callers = self.find_transitive_callers(repo_id, entity_name, max_depth=4)
        affected_files = sorted({c["file"] for c in callers if c.get("file")})
        # Heuristic: a caller file is "untested" when its path does not contain
        # "test_" or "/test/" — i.e. there is no obvious co-located test file.
        untested = [
            c for c in callers
            if c.get("file") and "test" not in c["file"].lower()
        ]
        return {
            "affected_functions": callers,
            "affected_files": affected_files,
            "untested_callers": untested,
            "total_affected_functions": len(callers),
            "total_affected_files": len(affected_files),
        }

    def find_cycles(
        self,
        repo_id: str,
        file_path: str,
        max_depth: int = 6,
    ) -> list:
        """
        Detect CALLS-edge cycles that involve at least one function in
        `file_path`.  Returns up to 10 unique cycles; each cycle is a list of
        entity names in canonical form (rotated so the lexicographically
        smallest name appears first, making [A,B,C] and [B,C,A] identical).
        """
        if not file_path:
            return []
        query = f"""
        MATCH (start:Entity {{repo_id: $repo_id, file: $file_path}})
        MATCH path = (start)-[:CALLS*2..{max_depth}]->(start)
        RETURN [n IN nodes(path) | n.name] AS cycle
        LIMIT 20
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, file_path=file_path)
                seen: set = set()
                cycles: list = []
                for r in result:
                    raw = r["cycle"]
                    if not raw:
                        continue
                    min_idx = raw.index(min(raw))
                    canonical = tuple(raw[min_idx:] + raw[:min_idx])
                    if canonical not in seen:
                        seen.add(canonical)
                        cycles.append(list(canonical))
                    if len(cycles) >= 10:
                        break
                return cycles
        except Exception as e:
            logger.error(f"[M5] find_cycles error for {file_path}: {e}")
            return []

    def find_related_by_file(self, repo_id: str, file_path: str, limit: int = 10):
        """
        Find all entities defined in a specific file
        Returns functions, classes, and other entities
        """
        logger.debug(f"[GraphDB] find_related_by_file: repo_id={repo_id}, file={file_path}, limit={limit}")
        
        query = """
        MATCH (e:Entity {repo_id: $repo_id, file: $file_path})
        RETURN e.name AS name, e.type AS type, e.start_line AS line, 
               e.end_line AS end_line, e.raw_code AS code
        ORDER BY e.start_line
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, file_path=file_path, limit=limit)
                entities = [
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "line": record["line"],
                        "end_line": record["end_line"],
                        "code": record["code"]
                    }
                    for record in result
                ]
                logger.debug(f"[GraphDB] find_related_by_file returned {len(entities)} entities")
                if len(entities) == 0:
                    logger.warning(f"[GraphDB] No entities found for repo_id={repo_id}, file={file_path}")
                return entities
        except Exception as e:
            logger.error(f"Error finding entities by file: {e}")
            return []
    
    def find_callers(self, repo_id: str, function_name: str, limit: int = 10):
        """
        Find all functions that call a specific function (reverse dependency)
        Useful for impact analysis
        """
        query = """
        MATCH (caller:Entity {repo_id: $repo_id})-[:CALLS|MAY_CALL]->(target:Entity {repo_id: $repo_id, name: $function_name})
        RETURN DISTINCT caller.name AS name, caller.file AS file, caller.type AS type,
               caller.start_line AS line
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, function_name=function_name, limit=limit)
                return [
                    {
                        "name": record["name"],
                        "file": record["file"],
                        "type": record["type"],
                        "line": record["line"],
                        "relationship": "calls"
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error finding callers: {e}")
            return []
    
    def find_call_chain(self, repo_id: str, function_name: str, max_depth: int = 3):
        """
        Find the call chain for a function (what it calls, recursively)
        Useful for understanding execution flow
        """
        query = """
        MATCH path = (source:Entity {repo_id: $repo_id, name: $function_name})-[:CALLS|MAY_CALL*1..$max_depth]->(target)
        WITH path, length(path) AS depth
        ORDER BY depth
        RETURN [node IN nodes(path) | {
            name: node.name,
            type: node.type,
            file: node.file,
            line: node.start_line
        }] AS chain
        LIMIT 10
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, function_name=function_name, max_depth=max_depth)
                return [record["chain"] for record in result]
        except Exception as e:
            logger.error(f"Error finding call chain: {e}")
            return []
    
    def find_file_dependencies(self, repo_id: str, file_path: str):
        """
        Find all files that this file depends on (via function calls)
        Useful for understanding file-level coupling
        """
        logger.debug(f"[GraphDB] find_file_dependencies: repo_id={repo_id}, file={file_path}")
        
        query = """
        MATCH (source:Entity {repo_id: $repo_id, file: $file_path})-[:CALLS|MAY_CALL]->(target:Entity {repo_id: $repo_id})
        WHERE target.file <> $file_path
        RETURN DISTINCT target.file AS file, COUNT(*) AS call_count
        ORDER BY call_count DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, file_path=file_path)
                dependencies = [
                    {
                        "file": record["file"],
                        "call_count": record["call_count"],
                        "relationship": "depends_on"
                    }
                    for record in result
                ]
                logger.debug(f"[GraphDB] find_file_dependencies returned {len(dependencies)} dependencies")
                return dependencies
        except Exception as e:
            logger.error(f"Error finding file dependencies: {e}")
            return []
    
    def find_similar_functions(self, repo_id: str, function_name: str, limit: int = 5):
        """
        Find functions with similar names (potential duplicates or related functionality)
        Uses fuzzy name matching
        """
        query = """
        MATCH (e:Entity {repo_id: $repo_id, type: 'Function'})
        WHERE e.name CONTAINS $search_term OR $search_term CONTAINS e.name
        AND e.name <> $function_name
        RETURN e.name AS name, e.file AS file, e.start_line AS line, e.raw_code AS code
        LIMIT $limit
        """
        # Extract base name (remove prefixes/suffixes for better matching)
        search_term = function_name.replace("get", "").replace("set", "").replace("_", "")
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query, 
                    repo_id=repo_id, 
                    function_name=function_name,
                    search_term=search_term,
                    limit=limit
                )
                return [
                    {
                        "name": record["name"],
                        "file": record["file"],
                        "line": record["line"],
                        "code": record["code"]
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error finding similar functions: {e}")
            return []
    
    def get_complexity_hotspots(self, repo_id: str, min_calls: int = 5, limit: int = 10):
        """
        Find functions with high number of outgoing calls (complexity hotspots)
        These functions are doing too much and may need refactoring
        """
        query = """
        MATCH (source:Entity {repo_id: $repo_id})-[:MAY_CALL]->(target)
        WITH source, COUNT(target) AS call_count
        WHERE call_count >= $min_calls
        RETURN source.name AS name, source.file AS file, source.type AS type,
               source.start_line AS line, call_count
        ORDER BY call_count DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, min_calls=min_calls, limit=limit)
                return [
                    {
                        "name": record["name"],
                        "file": record["file"],
                        "type": record["type"],
                        "line": record["line"],
                        "call_count": record["call_count"],
                        "issue": "High complexity - too many dependencies"
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error finding complexity hotspots: {e}")
            return []
    
    def get_highly_coupled_files(self, repo_id: str, min_connections: int = 5, limit: int = 10):
        """
        Find files that are highly coupled (many cross-file dependencies)
        Useful for identifying architectural issues
        """
        query = """
        MATCH (source:Entity {repo_id: $repo_id})-[:MAY_CALL]->(target:Entity {repo_id: $repo_id})
        WHERE source.file <> target.file
        WITH source.file AS source_file, target.file AS target_file, COUNT(*) AS connections
        WHERE connections >= $min_connections
        RETURN source_file, target_file, connections
        ORDER BY connections DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, min_connections=min_connections, limit=limit)
                return [
                    {
                        "source_file": record["source_file"],
                        "target_file": record["target_file"],
                        "connections": record["connections"],
                        "issue": "High coupling between files"
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error finding coupled files: {e}")
            return []
    
    def find_unused_functions(self, repo_id: str, limit: int = 20):
        """
        Find functions that are never called by other code
        Potential dead code candidates
        """
        query = """
        MATCH (e:Entity {repo_id: $repo_id, type: 'Function'})
        WHERE NOT (()-[:MAY_CALL]->(e))
        AND NOT e.name IN ['main', 'index', '__init__', 'handler', 'default']
        RETURN e.name AS name, e.file AS file, e.start_line AS line
        ORDER BY e.file, e.start_line
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, repo_id=repo_id, limit=limit)
                return [
                    {
                        "name": record["name"],
                        "file": record["file"],
                        "line": record["line"],
                        "issue": "Potentially unused function (no callers found)"
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Error finding unused functions: {e}")
            return []
