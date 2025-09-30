"""LangGraph configuration and tools for conversation management."""

import logging
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# from livekit.plugins import langchain
from custom_plugins import langchain

from .user_profile import UserProfileManager
from .mongodb_manager import MongoDBManager

logger = logging.getLogger("graph-builder")

# LangGraph 지원 모델 설정
MODEL_CONFIGS = {
    "gemini": "google_genai:gemini-2.5-flash",
    "claude": "anthropic:claude-sonnet-4-20250514",
    "gpt": "openai:gpt-4o",
}


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_tools(thread_identity: str, mongodb_manager: MongoDBManager) -> list:
    """Create LangGraph tools with participant_id bound via closure."""
    
    @tool
    def save_user_name(name: str) -> str:
        """
        Save user's name when they introduce themselves.
        
        Args:
            name: The user's name to save
        
        Updates permanently in MongoDB Store. Will be reflected in next session.
        """
        if not thread_identity:
            return "[SYSTEM_CONTEXT: Unable to save name - session error.]"
        
        success = UserProfileManager.update_user_name(thread_identity, name, mongodb_manager.store)
        
        if success:
            return f"[SYSTEM_CONTEXT: User '{name}' introduced themselves. Remember their name and respond naturally.]"
        else:
            logger.error(f"[LangGraph Tool] Failed to save name for {thread_identity}")
            return f"[SYSTEM_CONTEXT: User '{name}' introduced themselves. Remember their name and respond naturally.]"
    
    return [save_user_name]


def create_graph_with_mongodb(model_name: str, thread_identity: str, mongodb_manager) -> StateGraph:
    """Create LangGraph with MongoDB checkpointer and store.
    
    Args:
        model_name: Name of the model to use
        thread_identity: Thread identifier
        mongodb_manager: MongoDBManager instance
    """
    
    if thread_identity:
        logger.info(f"[LangGraph] MongoDB checkpointer initialized for: {thread_identity}")
    else:
        logger.warning("[LangGraph] No thread_identity provided for MongoDB memory")
    
    # 모델 설정 확인
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    else:
        logger.info(f"[LangGraph] 🤖 Using model: {model_name} 🤖")
    
    # 채팅 모델 초기화
    chat_model = init_chat_model(
        model=MODEL_CONFIGS[model_name],
    )
    
    # 참가자별 도구 생성 (store 전달)
    tools = create_tools(thread_identity, mongodb_manager)
    llm_with_tools = chat_model.bind_tools(tools=tools, tool_choice="auto")
    logger.info(f"[LangGraph] Tools bound to model: {[tool.name for tool in tools]}")

    # 채팅봇 노드
    def chatbot_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
        logger.debug(f"[LangGraph] Response: {response}")
        return {"messages": [response]}

    # 도구 실행 노드
    tool_node = ToolNode(tools)
    
    # 조건부 라우팅 함수
    def should_continue(state: State):
        """도구 호출이 필요한지 확인하고 라우팅"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"[LangGraph] Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tools"
        logger.debug("[LangGraph] No tool calls, ending conversation turn")
        return "end"

    # 그래프 구성
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    
    # 엣지 설정
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    builder.add_edge("tools", "chatbot")
    
    # MongoDB checkpointer 사용 (단기 메모리)
    checkpointer = mongodb_manager.checkpointer
    
    logger.info("[LangGraph] Graph compiled with MongoDB checkpointer and store")
    return builder.compile(checkpointer=checkpointer, store=mongodb_manager.store)


def get_langgraph(model_name: str = "claude", user_id: str = None, thread_id: str = None, mongodb_manager = None):
    """Get LangGraph LLM with MongoDB checkpointer.""" 
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    graph = create_graph_with_mongodb(
        model_name=model_name, 
        thread_identity=thread_id,
        mongodb_manager=mongodb_manager
    )
    
    # Thread-based configuration for MongoDB persistence
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    
    return langchain.LLMAdapter(graph, config=config)