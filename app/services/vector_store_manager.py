"""Vector store manager - wraps Milvus VectorStore operations"""

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


# Use the unified biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """Vector store manager"""

    def __init__(self):
        """Initialize the vector store manager"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the Milvus VectorStore"""
        try:
            # A connection must be established before PyMilvus / langchain_milvus accesses
            # the Collection, otherwise a ConnectionNotExistException is raised.
            # This runs at module import time, before the milvus_manager.connect call
            # in the FastAPI lifespan handler.
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # Create LangChain Milvus VectorStore
            # Uses the biz collection with field mapping: text_field -> content, vector_field -> vector
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            logger.info(
                f"VectorStore initialized: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Batch-add documents to the vector store (with automatic batch embedding)

        Args:
            documents: List of documents

        Returns:
            List[str]: List of document IDs
        """
        try:
            import time
            import uuid
            start_time = time.time()

            ids = [str(uuid.uuid4()) for _ in documents]

            # LangChain Milvus add_documents automatically calls embedding_function
            # and handles batching for better performance
            result_ids = self.vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"Batch-added {len(documents)} document(s) to VectorStore, "
                f"elapsed: {elapsed:.2f}s, average: {elapsed/len(documents):.2f}s/doc"
            )
            return result_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        Delete all documents associated with the specified file

        Args:
            file_path: File path

        Returns:
            int: Number of documents deleted
        """
        try:
            collection = milvus_manager.get_collection()

            # metadata is a JSON field; use JSON path query syntax
            # _source is the source file path of the document
            expr = f'metadata["_source"] == "{file_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"Deleted old data for file: {file_path}, count: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"Failed to delete old data (may be first-time indexing): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """
        Get the VectorStore instance

        Returns:
            Milvus: VectorStore instance
        """
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform a similarity search

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List[Document]: List of relevant documents
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Similarity search complete: query='{query}', results={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []


# Global singleton
vector_store_manager = VectorStoreManager()
