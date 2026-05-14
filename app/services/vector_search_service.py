"""Vector search service module"""

from typing import Any, Dict, List

from loguru import logger
from pymilvus import Collection

from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class SearchResult:
    """Search result class"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """Vector search service - responsible for searching similar vectors in Milvus"""

    def __init__(self):
        """Initialize the vector search service"""
        logger.info("Vector search service initialized")

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Search for similar documents

        Args:
            query: Query text
            top_k: Number of most-similar results to return

        Returns:
            List[SearchResult]: List of search results

        Raises:
            RuntimeError: Raised when the search fails
        """
        try:
            logger.info(f"Starting similar document search, query: {query}, top_k: {top_k}")

            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"Query vector generated, dimensions: {len(query_vector)}")

            collection: Collection = milvus_manager.get_collection()

            search_params = {
                "metric_type": "L2",  # Euclidean distance
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # L2 distance; lower is more similar
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"Search complete, found {len(search_results)} similar document(s)")
            return search_results

        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e


# Global singleton
vector_search_service = VectorSearchService()
