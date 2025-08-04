"""Configuration package for face animation agent."""

from .voice_config import VoiceSettings, ElevenLabsConfig
from .language_config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    get_language_name,
    map_language_to_deepgram,
    is_supported_language,
    validate_language
)
from .session_config import USER_INACTIVITY_TIMEOUT_SECONDS

__all__ = [
    'VoiceSettings',
    'ElevenLabsConfig', 
    'SUPPORTED_LANGUAGES',
    'DEFAULT_LANGUAGE',
    'get_language_name',
    'map_language_to_deepgram',
    'is_supported_language',
    'validate_language',
    'USER_INACTIVITY_TIMEOUT_SECONDS'
]