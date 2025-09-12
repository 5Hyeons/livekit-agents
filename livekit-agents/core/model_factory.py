"""Simple model configurations for STT, LLM, TTS, and STF."""

import logging

from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.plugins import deepgram

from .graph_builder import get_langgraph

logger = logging.getLogger("model-factory")


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)


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