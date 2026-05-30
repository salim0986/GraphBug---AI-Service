"""
M8 — Agent tools for the reflection and agentic review nodes.

These tools wrap the existing graph/vector infrastructure with safe,
template-based interfaces so the LLM can never execute arbitrary Cypher.

Three tools are exposed:
  query_graph        — named Cypher templates, parameterised at call time
  search_similar_code — semantic vector search
  get_file_excerpt   — retrieve entity source code from the graph

The LLM can request any combination of these during reflection.
In the current implementation the reflection node calls them
programmatically; full tool-use (LLM-driven dispatch) is the next step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .logger import setup_logger

if TYPE_CHECKING:
    from .graph_builder import GraphBuilder
    from .vector_builder import VectorBuilder

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Safe Cypher templates — the LLM can only reference these by name.
# No arbitrary Cypher is ever passed through.
# ---------------------------------------------------------------------------

_SAFE_TEMPLATES: Dict[str, str] = {
    "callers": """
        MATCH (caller:Entity {repo_id: $repo_id})-[:CALLS|MAY_CALL]->
              (target:Entity {repo_id: $repo_id, name: $entity_name})
        RETURN DISTINCT caller.name AS name, caller.file AS file,
               caller.type AS type, caller.start_line AS line
        LIMIT 20
    """,
    "callees": """
        MATCH (source:Entity {repo_id: $repo_id, name: $entity_name})
              -[:CALLS|MAY_CALL]->(callee)
        RETURN DISTINCT callee.name AS name, callee.file AS file,
               callee.type AS type
        LIMIT 20
    """,
    "file_entities": """
        MATCH (e:Entity {repo_id: $repo_id, file: $file_path})
        RETURN e.name AS name, e.type AS type,
               e.start_line AS start_line, e.end_line AS end_line
        ORDER BY e.start_line
        LIMIT 50
    """,
    "cycle_check": """
        MATCH (start:Entity {repo_id: $repo_id, file: $file_path})
        MATCH path = (start)-[:CALLS*2..6]->(start)
        RETURN [n IN nodes(path) | n.name] AS cycle
        LIMIT 5
    """,
    "blast_radius": """
        MATCH path = (caller:Entity {repo_id: $repo_id})
                     -[:CALLS|MAY_CALL*1..4]->
                     (target:Entity {repo_id: $repo_id, name: $entity_name})
        RETURN DISTINCT caller.name AS name, caller.file AS file,
               min(length(path)) AS depth
        ORDER BY depth, caller.file
        LIMIT 30
    """,
}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def query_graph(
    graph_db: "GraphBuilder",
    repo_id: str,
    template_name: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Execute a named Cypher template against the knowledge graph.

    `template_name` must be one of the keys in `_SAFE_TEMPLATES`.
    Arbitrary Cypher strings are rejected.  `repo_id` is always injected
    so callers cannot query across tenants.

    Returns a list of record dicts; returns [] on any error.
    """
    if template_name not in _SAFE_TEMPLATES:
        logger.warning(
            f"[M8] query_graph: unknown template '{template_name}'. "
            f"Allowed: {list(_SAFE_TEMPLATES.keys())}"
        )
        return []

    cypher = _SAFE_TEMPLATES[template_name]
    query_params = {**params, "repo_id": repo_id}

    try:
        with graph_db.driver.session() as session:
            result = session.run(cypher, **query_params)
            return [dict(r) for r in result]
    except Exception as e:
        logger.error(f"[M8] query_graph('{template_name}') failed: {e}")
        return []


def search_similar_code(
    vector_db: "VectorBuilder",
    repo_id: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run a semantic similarity search against the vector index.

    Returns a list of dicts with keys: file, name, score, snippet.
    Returns [] on any error.
    """
    if not query or not query.strip():
        return []
    try:
        results = vector_db.search(
            repo_id=repo_id,
            query=query,
            top_k=top_k,
        )
        return [
            {
                "file": r.get("file", ""),
                "name": r.get("name", ""),
                "score": r.get("score", 0.0),
                "snippet": r.get("code", "")[:300],
            }
            for r in (results or [])
        ]
    except Exception as e:
        logger.error(f"[M8] search_similar_code failed: {e}")
        return []


def get_file_excerpt(
    graph_db: "GraphBuilder",
    repo_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Retrieve entity source code from the graph for a given line range.

    Finds the Entity whose start_line is closest to `start_line` inside
    `file_path` and returns its `raw_code`.  Returns "" on miss or error.
    """
    if not file_path or start_line < 0:
        return ""
    cypher = """
    MATCH (e:Entity {repo_id: $repo_id, file: $file_path})
    WHERE e.start_line >= $start_line AND e.start_line <= $end_line
    RETURN e.raw_code AS code, e.name AS name
    ORDER BY e.start_line
    LIMIT 3
    """
    try:
        with graph_db.driver.session() as session:
            result = session.run(
                cypher,
                repo_id=repo_id,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            )
            rows = [r for r in result]
        if not rows:
            return ""
        return "\n\n".join(
            f"# {r['name']}\n{r['code']}" for r in rows if r["code"]
        )
    except Exception as e:
        logger.error(f"[M8] get_file_excerpt failed: {e}")
        return ""


def available_templates() -> List[str]:
    """Return the list of valid template names (for prompt injection)."""
    return list(_SAFE_TEMPLATES.keys())
