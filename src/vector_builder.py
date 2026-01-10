from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance
from qdrant_client.http import models
import uuid
from .logger import setup_logger

logger = setup_logger(__name__)

class VectorBuilder:
    def __init__(self, url, model):
        self.client = QdrantClient(url=url)
        self.model = model
        self.collection = "repo_code"

    def ensure_collection(self):
        """
        Creates the collection if it doesn't exist.
        """
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"Created vector collection '{self.collection}'")
        except Exception as e:
            logger.error(f"Vector DB connection error: {e}")
            raise

    def ingest_function_chunk(self, repo_id, func_name, func_code, file_path, start_line):
        """
        Ingest a chunk of code with repo_id for tenant isolation.
        """
        try:
            vector = self.model.encode(func_code).tolist()

            payload = {
                "repo_id": repo_id,  # Tenant Badge
                "name": func_name,
                "file": file_path,
                "type": "function",
                "start_line": start_line,
                "raw_code": func_code 
            }

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            logger.warning(f"Vector ingest error for {func_name}: {e}")
            raise

    def search_similar(self, repo_id, query_text, limit=5):
        """
        Retrieves code relevant to the query, strictly filtered by repo_id.
        """
        try:
            query_vector = self.model.encode(query_text).tolist()
            
            result = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                ),
                limit=limit
            ).points
            return result
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def delete_repo(self, repo_id: str):
        """Deletes all vectors associated with a specific repo_id."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="repo_id",
                                match=models.MatchValue(value=repo_id),
                            ),
                        ],
                    )
                ),
            )
            print(f"🗑️  Deleted vectors for repo {repo_id}")
        except Exception as e:
            print(f"⚠️ Vector delete failed: {e}")