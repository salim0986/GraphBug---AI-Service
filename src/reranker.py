from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from .logger import setup_logger

logger = setup_logger(__name__)

class Reranker:
    """
    Reranks semantic search results using a cross-encoder model (e.g. BAAI/bge-reranker-base).
    Cross-encoders score pairs of (query, document) directly and are more accurate 
    than bi-encoders, but slower.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on a query.
        Assumes documents have a 'code' or 'raw_code' field to evaluate.
        """
        if not documents:
            return []
            
        # Extract the text to score against the query
        # Usually it's 'raw_code' or 'code' from the VectorBuilder output
        pairs = []
        for doc in documents:
            # Handle both dictionary keys and Qdrant payload objects
            if isinstance(doc, dict):
                text = doc.get("raw_code") or doc.get("code") or doc.get("text", "")
            else:
                # If it's a Qdrant point object
                text = getattr(doc, "payload", {}).get("raw_code", "")
                
            pairs.append((query, str(text)))
            
        # Score pairs
        try:
            scores = self.model.predict(pairs)
            
            # Attach scores to documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy() if isinstance(doc, dict) else doc
                if isinstance(doc_copy, dict):
                    doc_copy["rerank_score"] = float(score)
                scored_docs.append((score, doc_copy))
                
            # Sort by score descending
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k
            return [doc for _, doc in scored_docs[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order if reranking fails
            return documents[:top_k]
