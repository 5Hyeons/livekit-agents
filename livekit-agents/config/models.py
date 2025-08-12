"""Simple model configurations for STT, LLM, TTS, and STF."""

from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.plugins import deepgram, anthropic


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)


def get_llm():
    """Get Anthropic LLM configuration.""" 
    return anthropic.LLM(
        model="claude-4-sonnet-20250514",
        caching="ephemeral",
        max_tokens=192,
    )


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