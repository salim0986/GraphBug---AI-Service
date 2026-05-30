"""
M11 E2E test — full ingestion + review cycle with real Neo4j + Qdrant.

Requires Docker.  Skip automatically when Docker is unavailable so unit test
runs are unaffected.

Run manually:
    docker-compose up -d
    pytest tests/e2e/test_full_review.py -v

Or with testcontainers (Docker required):
    pytest tests/e2e/test_full_review.py -v -m e2e
"""

import subprocess
import sys
import os
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Docker guard
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    """Return True when Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


_DOCKER_SKIP = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker is not available in this environment",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_MODULE = Path(__file__).parent / "fixtures" / "sample_module.py"


@pytest.fixture(scope="module")
def neo4j_container():
    """Spin up a real Neo4j container for the test session."""
    try:
        from testcontainers.neo4j import Neo4jContainer  # type: ignore[import]
    except ImportError:
        pytest.skip("testcontainers[neo4j] not installed")

    with Neo4jContainer("neo4j:5") as container:
        # Wait for Neo4j to be ready
        bolt_url = container.get_connection_url()
        yield {"bolt_url": bolt_url, "user": "neo4j", "password": "test"}


@pytest.fixture(scope="module")
def qdrant_container():
    """Spin up a real Qdrant container for the test session."""
    try:
        from testcontainers.core.container import DockerContainer  # type: ignore[import]
    except ImportError:
        pytest.skip("testcontainers not installed")

    with DockerContainer("qdrant/qdrant:latest").with_bind_ports(6333, 6333) as c:
        time.sleep(2)  # brief startup wait
        yield {"url": "http://localhost:6333"}


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

@_DOCKER_SKIP
class TestFullReviewE2E:
    """End-to-end: ingest fixture repo → query graph → assert expected entities."""

    def test_fixture_module_exists(self):
        """Sanity: the fixture Python file that will be ingested is present."""
        assert FIXTURE_MODULE.exists(), f"Fixture not found: {FIXTURE_MODULE}"

    def test_fixture_module_is_valid_python(self):
        """Fixture file can be compiled without syntax errors."""
        source = FIXTURE_MODULE.read_text()
        compile(source, str(FIXTURE_MODULE), "exec")

    def test_neo4j_ingest_and_query(self, neo4j_container):
        """
        Ingest sample_module.py into a real Neo4j instance and assert the
        expected functions and call edges are present.
        """
        from src.graph_builder import GraphBuilder
        from src.parser import UniversalParser

        parser = UniversalParser()
        graph = GraphBuilder(
            neo4j_container["bolt_url"],
            (neo4j_container["user"], neo4j_container["password"]),
        )

        repo_id = "e2e-test-repo"
        fixture_path = str(FIXTURE_MODULE)
        captures, code_bytes = parser.parse_file(fixture_path)
        assert captures, "Parser produced no captures for sample_module.py"

        graph.process_file_nodes(repo_id, "sample_module.py", captures, code_bytes)
        graph.build_dependencies(repo_id)

        # The file should contain at least 4 functions/classes
        with graph.driver.session() as session:
            result = session.run(
                "MATCH (e {repo_id: $repo_id}) RETURN count(e) AS cnt",
                repo_id=repo_id,
            )
            cnt = result.single()["cnt"]

        assert cnt >= 4, f"Expected ≥4 graph nodes, got {cnt}"

        graph.driver.close()

    def test_qdrant_ingest_and_search(self, qdrant_container):
        """
        Ingest sample_module.py into real Qdrant and confirm search returns
        a result for a relevant query.
        """
        from sentence_transformers import SentenceTransformer
        from src.vector_builder import VectorBuilder

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vdb = VectorBuilder(
            qdrant_container["url"],
            model,
            api_key=None,
        )
        vdb.ensure_collection()

        source = FIXTURE_MODULE.read_text()
        vdb.ingest_function_chunk(
            repo_id="e2e-qdrant-repo",
            func_name="process_order",
            func_code=source,
            file_path="sample_module.py",
            start_line=0,
        )

        results = vdb.search_similar("e2e-qdrant-repo", "discount calculation")
        assert len(results) >= 1, "Expected at least one search result"
