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
from livekit.plugins import deepgram, anthropic, openai, google, langchain

from config.mongodb_config import get_checkpointer, test_mongodb_connection

logger = logging.getLogger("models-config")

# MongoDB Checkpointer를 위한 전역 변수
_current_participant_id = None

@tool
def save_user_name_langgraph(name: str) -> str:
    """
    Save user's name when they introduce themselves.
    
    Args:
        name: The user's name to save
    
    Note: Currently stores only in conversation context (checkpointer).
    Long-term storage will be implemented later.
    """
    global _current_participant_id
    
    if not _current_participant_id:
        logger.error("[LangGraph Tool] Participant ID not available")
        return "[ERROR: Session not available]"
    
    # For now, just acknowledge the name without permanent storage
    logger.info(f"[LangGraph Tool] User name noted: {_current_participant_id} -> {name}")
    return f"Nice to meet you, {name}! I'll remember your name during our conversation."

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_graph_with_mongodb(participant_id: str = None) -> StateGraph:
    """Create LangGraph with MongoDB checkpointer only (short-term memory)."""
    global _current_participant_id
    
    # MongoDB 연결 테스트
    if not test_mongodb_connection():
        raise ConnectionError("MongoDB connection failed")
    
    _current_participant_id = participant_id
    
    if participant_id:
        logger.info(f"[LangGraph] MongoDB checkpointer initialized for: {participant_id}")
    else:
        logger.warning("[LangGraph] No participant_id provided for MongoDB memory")
    
    # 채팅 모델 초기화 - Claude 사용
    chat_model = init_chat_model(
        model="google_genai:gemini-2.5-flash",
    )
    
    # 도구를 모델에 바인딩
    tools = [save_user_name_langgraph]
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
    
    # MongoDB checkpointer만 사용 (단기 메모리)
    checkpointer = get_checkpointer()
    
    logger.info("[LangGraph] Graph compiled with MongoDB checkpointer only")
    return builder.compile(checkpointer=checkpointer)


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)

def get_llm(model_name: str = "claude-4-sonnet-20250514", user_data=None):
    """Get LLM configuration.""" 
    match model_name:
        case "claude-4-sonnet-20250514":
            return anthropic.LLM(
                model="claude-4-sonnet-20250514",
                caching="ephemeral",
                max_tokens=192,
            )
        case "gpt-4o-mini":
            return openai.LLM(
                model="gpt-4o-mini",
                max_completion_tokens=192,
            )
        case "gemini":
            return google.LLM(
                model="gemini-2.5-flash",
                # max_output_tokens=192,
            )
        case "langgraph":
            participant_id = user_data.participant_id if user_data else None
            graph = create_graph_with_mongodb(participant_id=participant_id)
            
            # Thread-based configuration for MongoDB persistence
            config = {
                "configurable": {
                    "thread_id": f"user_{participant_id}" if participant_id else "default_thread",
                    "user_id": participant_id
                }
            }
            
            return langchain.LLMAdapter(graph, config=config)
        case _:
            raise ValueError(f"Unsupported model: {model_name}")


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