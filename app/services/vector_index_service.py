"""Vector index service module"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """Indexing result class"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        """Increment the success counter"""
        self.success_count += 1

    def increment_fail_count(self):
        """Increment the failure counter"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """Record a failed file"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """Vector index service - responsible for reading files, generating vectors, and storing them in Milvus"""

    def __init__(self):
        """Initialize the vector index service"""
        self.upload_path = "./uploads"
        logger.info("Vector index service initialized")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        Index all files in the specified directory

        Args:
            directory_path: Directory path (optional; defaults to the configured upload directory)

        Returns:
            IndexingResult: Indexing result
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory does not exist or is not a valid directory: {target_path}")

            result.directory_path = str(dir_path)

            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"No supported files found in directory: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"Starting directory indexing: {target_path}, found {len(files)} file(s)")

            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ File indexed successfully: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ File indexing failed: {file_path.name}, error: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"Directory indexing complete: total={result.total_files}, "
                f"succeeded={result.success_count}, failed={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to index directory: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        """
        Index a single file (using the LangChain-based splitter)

        Args:
            file_path: Path to the file

        Raises:
            ValueError: Raised when the file does not exist
            RuntimeError: Raised when indexing fails
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {file_path}")

        logger.info(f"Starting file indexing: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"File read: {path}, content length: {len(content)} characters")

            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"Document splitting complete: {file_path} -> {len(documents)} chunk(s)")

            if documents:
                vector_store_manager.add_documents(documents)
                logger.info(f"File indexed: {file_path}, total {len(documents)} chunk(s)")
            else:
                logger.warning(f"File content is empty or cannot be split: {file_path}")

        except Exception as e:
            logger.error(f"Failed to index file: {file_path}, error: {e}")
            raise RuntimeError(f"File indexing failed: {e}") from e


# Global singleton
vector_index_service = VectorIndexService()
