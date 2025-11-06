"""Simple model configurations for STT, LLM, TTS, and STF."""

import logging

from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.plugins import deepgram, anthropic, openai, google

from .graph_builder import get_langgraph

logger = logging.getLogger("model-factory")


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)


def get_llm(model_provider: str = "google"):
    """Get LLM configuration."""
    match model_provider:
        case "chat":
            # Chat mode - text-only agent with Gemini 2.5 Flash
            return google.LLM(model='gemini-2.5-flash')
        case "realtime":
            # Avatar mode - realtime voice agent with OpenAI Realtime API
            return openai.realtime.RealtimeModel(model="gpt-realtime", voice="Ash")
        case "openai":
            return openai.LLM(model='gpt-4o')
        case "anthropic":
            return anthropic.LLM(model='claude-sonnet-4-20250514')
        case "google":
            return google.LLM(model='gemini-2.5-flash')
        case _:
            raise ValueError(f"Invalid model provider: {model_provider}")

def get_tts(voice_name: str = "FEMALE_1", voice_speed_offset: float = 0.0):
    """Get ElevenLabs TTS configuration."""
    from config.voices import ElevenLabsConfig
    return ElevenLabsConfig.from_voice_name(voice_name, voice_speed_offset).create_tts()


def get_stf():
    """Get FaceAnimator STF configuration."""
    return FaceAnimator(
        chunk_duration_sec=1.0, 
        output_mode=OutputMode.ANIMATION_ONLY
    )