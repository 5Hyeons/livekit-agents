"""Simple model configurations for STT, LLM, TTS, and STF."""

from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.plugins import deepgram, anthropic, openai, google


def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration."""
    return deepgram.STT(model="nova-2-general", language=language)


def get_llm(model_name: str = "claude-4-sonnet-20250514"):
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
                model="gemini-2.5-flash-lite",
                max_output_tokens=192,
            )
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