"""vector index service module"""
'''
3. Milvus 的核心特点
高性能：它专门针对向量运算（如余弦相似度、欧氏距离）进行了极致优化。

海量容量：它支持分布式部署，能处理万亿级别的向量数据。

支持多种索引：你可以把它想象成书本的目录，Milvus 提供了多种“目录算法”（如 IVF, HNSW, DiskANN），让查询变得飞快。

云原生：它非常适合现在流行的微服务架构，可以很方便地用 Docker 或 Kubernetes 部署。
'''

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """indexing result class"""

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
        """increment success count"""
        self.success_count += 1

    def increment_fail_count(self):
        """increment fail count"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """add failed file"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """get duration in milliseconds"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary"""
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
    """vector index service - responsible for reading files, generating vectors, and storing to Milvus"""

    def __init__(self):
        """initialize vector index service"""
        self.upload_path = "./uploads"
        logger.info("vector index service initialized")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        index all files in the specified directory

        Args:
            directory_path: directory path (optional, default to the configured upload directory)

        Returns:
            IndexingResult: indexing result
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # use specified directory or default upload directory
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"directory does not exist or is not a valid directory: {target_path}")

            result.directory_path = str(dir_path)

            # get all supported files
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md")) #文件筛选：只处理 .txt 和 .md 文件（通过 glob 实现）。

            if not files:
                logger.warning(f"no supported files found in directory: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"indexing directory: {target_path}, found {len(files)} files")

            # iterate and index each file
            #容错处理：用了 try...except 嵌套。即使文件夹里某一个文件损坏导致失败，程序也不会崩溃，而是记录在 failed_files 里，继续处理下一个。
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"file indexing successful: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"file indexing failed: {file_path.name}, error: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"directory indexing completed: total={result.total_files}, "
                f"success={result.success_count}, fail={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"directory indexing failed: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        """
        index single file (using new LangChain splitter)

        Args:
            file_path: file path

        Raises:
            ValueError: file does not exist
            RuntimeError: indexing failed
        """
        path = Path(file_path).resolve() #读取 (Read)：使用 path.read_text() 把文件内容读成字符串。

# 清理 (Delete Old)：vector_store_manager.delete_by_source(...)。这是为了防止重复索引。如果这个文件之前传过，先删掉旧的向量，再存新的。
        if not path.exists() or not path.is_file():
            raise ValueError(f"file does not exist: {file_path}")

        logger.info(f"starting to index file: {path}")

        try:
            # 1. read file content
            content = path.read_text(encoding="utf-8")
            logger.info(f"reading file: {path}, content length: {len(content)} characters")

            # 2. delete old data for this file (if exists)
            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            # 3. use new document splitter调用 document_splitter_service。
            # 为什么切分？ 因为 LLM（大模型）有上下文长度限制，且小段落的向量检索更精准。
            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"document splitting completed: {file_path} -> {len(documents)} chunks")

            # 4. add documents to vector store
            #入库 (Store)：vector_store_manager.add_documents(documents)。
            # 这一步在底层会调用 Embedding 模型（把文字转成一串数字）并存入 Milvus。
            if documents:
                vector_store_manager.add_documents(documents)
                logger.info(f"file indexing completed: {file_path}, total {len(documents)} chunks")
            else:
                logger.warning(f"file content is empty or cannot be split: {file_path}")

        except Exception as e:
            logger.error(f"file indexing failed: {file_path}, error: {e}")
            raise RuntimeError(f"file indexing failed: {e}") from e


# global singleton
vector_index_service = VectorIndexService()
