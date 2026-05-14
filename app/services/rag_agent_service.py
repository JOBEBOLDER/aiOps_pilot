"""RAG Agent service - LangGraph-based intelligent agent

Uses the native ChatQwen integration from langchain_qwq,
supporting true streaming output and better model compatibility.
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry

# Alibaba Qwen + LangChain integration reference: https://docs.langchain.com/oss/python/integrations/chat/qwen
# Note: set DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1, otherwise the
# Singapore endpoint is used by default.
# Also set DASHSCOPE_API_KEY=your_api_key.


class AgentState(TypedDict):
    """Agent state"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    Trim message history to keep only the most recent messages within the context window.

    Strategy:
    - Retain the first system message
    - Retain the most recent 6 messages (3 conversation turns)
    - Skip trimming when there are 7 or fewer messages

    Args:
        state: Agent state

    Returns:
        Dict with trimmed messages, or None if no trimming is needed
    """
    messages = state["messages"]

    if len(messages) <= 7:
        return None

    first_msg = messages[0]

    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"Trimmed message history: {len(messages)} -> {len(new_messages)} message(s)")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent service - uses LangGraph + native ChatQwen integration"""

    def __init__(self, streaming: bool = True):
        """Initialize the RAG Agent service

        Args:
            streaming: Whether to enable streaming output (default: True)
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        self.tools = [retrieve_knowledge, get_current_time]

        # MCP client (lazily initialized, managed globally)
        self.mcp_tools: list = []

        self.checkpointer = MemorySaver()

        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent service initialized (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """Asynchronously initialize the agent (including MCP tools)"""
        if self._agent_initialized:
            return

        mcp_client = await get_mcp_client_with_retry()

        mcp_tools = await mcp_client.get_tools()
        logger.info(f"Successfully loaded {len(mcp_tools)} MCP tool(s)")

        self.mcp_tools = mcp_tools

        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"Available tools: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt.

        Note: The LangChain framework automatically passes tool information to the LLM,
        so there is no need to enumerate specific tools in the system prompt.

        Returns:
            str: System prompt
        """
        from textwrap import dedent

        return dedent("""
            You are a professional AI assistant capable of using various tools to help users solve problems.

            Working principles:
            1. Understand user requirements and choose appropriate tools to complete tasks.
            2. Proactively use relevant tools when real-time information or specialized knowledge is needed.
            3. Provide accurate, professional answers based on tool results.
            4. If tools cannot provide sufficient information, honestly inform the user.

            Response requirements:
            - Maintain a friendly, professional tone.
            - Keep answers concise and to the point.
            - Base responses on facts; do not fabricate information.
            - Clearly state any uncertainties.

            Use the available tools flexibly to provide high-quality assistance.
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        Process a user question without streaming (returns the complete answer at once)

        Args:
            question: User question
            session_id: Session ID (used as thread_id)

        Returns:
            str: Complete answer
        """
        try:
            await self._initialize_agent()

            logger.info(f"[Session {session_id}] RAG Agent received query (non-streaming): {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[Session {session_id}] Agent called tools: {tool_names}")

                logger.info(f"[Session {session_id}] RAG Agent query complete (non-streaming)")
                return answer

            logger.warning(f"[Session {session_id}] Agent returned an empty result")
            return ""

        except Exception as e:
            logger.error(f"[Session {session_id}] RAG Agent query failed (non-streaming): {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user question with streaming (yields answer fragments incrementally)

        Args:
            question: User question
            session_id: Session ID (used as thread_id)

        Yields:
            Dict[str, Any]: Dict containing streaming data
                - type: "content" | "tool_call" | "complete" | "error"
                - data: Specific content
        """
        try:
            await self._initialize_agent()

            logger.info(f"[Session {session_id}] RAG Agent received query (streaming): {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[Session {session_id}] RAG Agent query complete (streaming)")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[Session {session_id}] RAG Agent query failed (streaming): {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        Retrieve session history from the MemorySaver checkpointer

        Args:
            session_id: Session ID (i.e. thread_id)

        Returns:
            list: Message history [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            config = {"configurable": {"thread_id": session_id}}

            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                logger.info(f"Session history retrieved: {session_id}, message count: 0")
                return []

            # checkpoint_tuple may be a named tuple or a plain tuple; extract checkpoint safely
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"Session history retrieved: {session_id}, message count: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"Failed to retrieve session history: {session_id}, error: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        Clear session history from the MemorySaver checkpointer

        Args:
            session_id: Session ID (i.e. thread_id)

        Returns:
            bool: Whether the operation succeeded
        """
        try:
            self.checkpointer.delete_thread(session_id)

            logger.info(f"Session history cleared: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear session history: {session_id}, error: {e}")
            return False

    async def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up RAG Agent service resources...")
            # The MCP client is managed globally and does not need to be cleaned up manually
            logger.info("RAG Agent service resources cleaned up")
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")


# Global singleton - streaming enabled
rag_agent_service = RagAgentService(streaming=True)
