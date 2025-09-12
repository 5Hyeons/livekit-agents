"""Simple model configurations for STT, LLM, TTS, and STF."""
from typing import Annotated, TypedDict
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.plugins import deepgram, langchain

from config.mongodb_config import get_checkpointer, get_store, test_mongodb_connection

logger = logging.getLogger("models-config")

# LangGraph 지원 모델 설정
MODEL_CONFIGS = {
    "gemini": "google_genai:gemini-2.5-flash",
    "claude": "anthropic:claude-sonnet-4-20250514",
}

def create_tools_for_participant(participant_id: str) -> list:
    """Create tools with participant_id bound via closure."""
    
    @tool
    def save_user_name(name: str) -> str:
        """
        Save user's name when they introduce themselves.
        
        Args:
            name: The user's name to save
        
        Updates permanently in MongoDB Store. Will be reflected in next session.
        """
        if not participant_id:
            return "[SYSTEM_CONTEXT: Unable to save name - session error.]"
        
        from datetime import datetime
        store = get_store()
        
        # Get existing profile
        profile_data = store.get(
            namespace=("user_profile", participant_id),
            key="basic_info"
        )
        
        # Update existing profile
        updated_profile = profile_data.value
        updated_profile["name"] = name
        updated_profile["last_seen"] = datetime.now().isoformat()
        
        # Save updated profile to MongoDB Store
        store.put(
            namespace=("user_profile", participant_id),
            key="basic_info",
            value=updated_profile
        )
        
        logger.info(f"[LongTerm] Updated user name: {profile_data.value['name']} -> {name}")
        return f"[SYSTEM_CONTEXT: User '{name}' introduced themselves. Remember their name and respond naturally.]"
            
    return [save_user_name]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_graph_with_mongodb(model_name: str, participant_id: str = None) -> StateGraph:
    """Create LangGraph with MongoDB checkpointer only (short-term memory)."""
    # MongoDB 연결 테스트
    if not test_mongodb_connection():
        raise ConnectionError("MongoDB connection failed")
    
    # 모델 설정 확인
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    else:
        logger.info(f"[LangGraph] 🤖 Using model: {model_name} 🤖")
    
    # 채팅 모델 초기화
    chat_model = init_chat_model(
        model=MODEL_CONFIGS[model_name],
    )
    
    # 참가자별 도구 생성
    tools = create_tools_for_participant(participant_id)
    llm_with_tools = chat_model.bind_tools(tools=tools, tool_choice="auto")
    logger.info(f"[LangGraph] Tools bound to model: {[tool.name for tool in tools]}")

    # 채팅봇 노드
    def chatbot_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
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
    
    # MongoDB checkpointer + store 사용 (단기 + 장기 메모리)
    checkpointer = get_checkpointer()
    store = get_store()
    
    logger.info("[LangGraph] Graph compiled with MongoDB checkpointer and store")
    return builder.compile(checkpointer=checkpointer, store=store)


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)

def get_langgraph(model_name: str = "gemini", scene_name: str = "default_scene", participant_id: str = None):
    """Get LangGraph LLM with MongoDB checkpointer.""" 
    
    graph = create_graph_with_mongodb(model_name=model_name, participant_id=participant_id)
    
    # Thread-based configuration for MongoDB persistence
    thread_id = f"{scene_name}_{participant_id}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": participant_id
        }
    }
    
    return langchain.LLMAdapter(graph, config=config)


def get_tts(voice_name: str = "FEMALE_1"):
    """Get ElevenLabs TTS configuration."""
    from config.voices import ElevenLabsConfig
    return ElevenLabsConfig.from_voice_name(voice_name).create_tts()


def get_stf():
    """Get FaceAnimator STF configuration."""
    return FaceAnimator(
        chunk_duration_sec=2.0, 
        output_mode=OutputMode.ANIMATION_ONLY
    )