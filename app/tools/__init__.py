"""
LangChain tools available to every agent in this application.

Tools are plain async (or sync) Python functions decorated with @tool.
LangChain serialises the function signature + docstring into a JSON-schema
that the LLM reads to decide *when* and *how* to call the tool.

Two tools are provided here:
  - get_current_time  : returns the current wall-clock time (no external deps)
  - retrieve_knowledge: queries the Milvus vector store for relevant chunks

Design note — lazy imports inside retrieve_knowledge
------------------------------------------------------
vector_store_manager connects to Milvus at *import time* (module-level singleton).
If we imported it at the top of this file, starting the server would fail unless
Milvus is already running.  By importing inside the function body we defer the
connection until the tool is actually *called*, which decouples startup from
database availability.
"""

from datetime import datetime
from typing import List, Tuple

from langchain_core.tools import tool
from loguru import logger


# ---------------------------------------------------------------------------
# Tool 1: get_current_time
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """
    Return the current date and time in ISO-8601 format.

    Use this tool whenever you need to know the exact current time,
    for example when timestamping a diagnostic report or calculating
    how long ago an alert was triggered.
    """
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.debug(f"get_current_time called, returning: {formatted}")
    return formatted


# ---------------------------------------------------------------------------
# Tool 2: retrieve_knowledge
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List]:
    """
    Search the internal knowledge base (Milvus vector store) for documents
    relevant to the given query and return their text content.

    Use this tool to look up:
    - Standard Operating Procedures (SOPs)
    - Runbooks for known failure modes
    - Post-mortem reports
    - Any operational documentation that was previously uploaded

    Args:
        query: A natural-language description of what you are looking for.

    Returns:
        A tuple of (formatted_text_content, raw_document_list).
        When invoked directly (ainvoke / invoke), only the text content is
        returned as a string.
    """
    # Lazy import — avoids connecting to Milvus at server startup
    from app.services.vector_store_manager import vector_store_manager

    logger.info(f"retrieve_knowledge called, query: {query!r}")

    try:
        docs = vector_store_manager.similarity_search(query, k=3)

        if not docs:
            logger.info("retrieve_knowledge: no relevant documents found")
            return "No relevant documents found in the knowledge base.", []

        # Format retrieved chunks into a readable block
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("_file_name", "unknown")
            parts.append(f"[Document {i} — {source}]\n{doc.page_content}")

        content = "\n\n---\n\n".join(parts)
        logger.info(f"retrieve_knowledge: returned {len(docs)} document(s)")
        return content, docs

    except Exception as e:
        logger.error(f"retrieve_knowledge failed: {e}")
        error_msg = f"Knowledge retrieval failed: {e}"
        return error_msg, []
