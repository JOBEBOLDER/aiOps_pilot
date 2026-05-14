"""Document splitter service module - intelligent document splitting based on LangChain"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


class DocumentSplitterService:
    """Document splitter service - uses LangChain splitters"""

    def __init__(self):
        """Initialize the document splitter service"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown header splitter (only splits on H1 and H2 to reduce the number of chunks)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # H3 splitting omitted to avoid excessive fragmentation
            ],
            strip_headers=False,
        )

        # Recursive character splitter (used for secondary splitting with a larger chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # doubled chunk_size to reduce fragment count
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Document splitter service initialized, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        Split a Markdown document (two-stage splitting + small-chunk merging)

        Args:
            content: Markdown content
            file_path: File path (used for metadata)

        Returns:
            List[Document]: List of document chunks
        """
        if not content or not content.strip():
            logger.warning(f"Markdown document content is empty: {file_path}")
            return []

        try:
            # Stage 1: split by headings
            md_docs = self.markdown_splitter.split_text(content)

            # Stage 2: further split by size
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # Stage 3: merge chunks that are too small (< 300 characters)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown splitting complete: {file_path} -> {len(final_docs)} chunk(s)")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown splitting failed: {file_path}, error: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        Split a plain text document

        Args:
            content: Text content
            file_path: File path (used for metadata)

        Returns:
            List[Document]: List of document chunks
        """
        if not content or not content.strip():
            logger.warning(f"Text document content is empty: {file_path}")
            return []

        try:
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"Text splitting complete: {file_path} -> {len(docs)} chunk(s)")
            return docs

        except Exception as e:
            logger.error(f"Text splitting failed: {file_path}, error: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        Intelligently split a document (selects splitter based on file type)

        Args:
            content: Document content
            file_path: File path

        Returns:
            List[Document]: List of document chunks
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        Merge chunks that are too small

        Args:
            documents: List of documents
            min_size: Minimum chunk size in characters

        Returns:
            List[Document]: Merged document list
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # Current chunk is too small and merging won't exceed the size limit
                current_doc.page_content += "\n\n" + doc.page_content
                # Retain the metadata of the primary document
            else:
                merged_docs.append(current_doc)
                current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


# Global singleton
document_splitter_service = DocumentSplitterService()
