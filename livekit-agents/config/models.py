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

logger = logging.getLogger("models-config")

# 전역 변수로 데이터베이스 접근을 위한 참조 저장
_current_db = None
_current_user_data = None

@tool
def save_user_name_langgraph(name: str) -> str:
    """
    Save user's name when they introduce themselves (LangGraph version).
    
    Args:
        name: The user's name to save
    """
    global _current_db, _current_user_data
    
    if not _current_db or not _current_user_data:
        logger.error("[LangGraph Tool] Database or user data not available")
        return "[ERROR: Database not available]"
    
    # 중복 저장 방지
    if _current_user_data.display_name and _current_user_data.display_name == name:
        logger.info(f"[LangGraph Tool] Name already saved: {name}")
        return ""
    
    # 데이터베이스 업데이트
    _current_db.update_user_name(_current_user_data.participant_id, name)
    _current_user_data.display_name = name
    logger.info(f"[LangGraph Tool] User name saved: {_current_user_data.participant_id} -> {name}")
    
    return f"Successfully saved your name as '{name}'. Nice to meet you, {name}!"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Complete StateGraph with chatbot and tool nodes
def create_graph(db=None, user_data=None) -> StateGraph:
    global _current_db, _current_user_data
    
    # 전역 참조 설정 (LangGraph 도구에서 사용)
    if db and user_data:
        _current_db = db
        _current_user_data = user_data
        logger.info(f"[LangGraph] Database and user data set for: {user_data.participant_id}")
    else:
        logger.warning("[LangGraph] No database or user data provided - tools may not work properly")
    
    chat_model = init_chat_model(
        model="google_genai:gemini-2.5-flash",
    )
    
    # 도구를 모델에 바인딩
    tools = [save_user_name_langgraph]
    llm_with_tools = chat_model.bind_tools(tools=tools, tool_choice="auto")
    logger.info(f"[LangGraph] Tools bound to model: {[tool.name for tool in tools]}")

    # 채팅봇 노드 (LLM이 도구 호출 결정)
    def chatbot_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 도구 실행 노드
    tool_node = ToolNode(tools)
    
    # 조건부 라우팅 함수
    def should_continue(state: State):
        """도구 호출이 필요한지 확인하고 라우팅"""
        last_message = state["messages"][-1]
        
        # LLM이 도구 호출을 했다면 tools 노드로
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"[LangGraph] Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tools"
        # 아니면 종료
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
    builder.add_edge("tools", "chatbot")  # 도구 실행 후 다시 LLM으로
    
    logger.info("[LangGraph] Graph compiled with chatbot -> conditional -> tools -> chatbot cycle")
    return builder.compile()


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)

def get_llm(model_name: str = "claude-4-sonnet-20250514", db=None, user_data=None):
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
            graph = create_graph(db=db, user_data=user_data)
            return langchain.LLMAdapter(graph)
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