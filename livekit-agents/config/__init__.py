"""Configuration package for LiveKit agents."""

# Persona configuration
from .personas import create_instructions

# Voice configuration
from .voices import VoiceSettings, ElevenLabsConfig

# Language configuration
from .languages import (
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE,
    get_language_name, map_language_to_deepgram, 
    is_supported_language, validate_language
)

# Logging configuration
from .logging import (
    is_user_logging_enabled, is_tts_logging_enabled, is_stt_logging_enabled, is_metrics_logging_enabled,
    get_tts_output_directory, get_stt_input_directory, get_metrics_output_directory, ensure_logging_directories,
    get_logging_settings, get_allowed_users, ALLOWED_LOGGING_USERS, LOGGING_SETTINGS
)

__all__ = [
    # Persona 
    'create_instructions',
    
    # Voice
    'VoiceSettings', 'ElevenLabsConfig',
    
    # Language
    'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE',
    'get_language_name', 'map_language_to_deepgram',
    'is_supported_language', 'validate_language',
    
    # Logging
    'is_user_logging_enabled', 'is_tts_logging_enabled', 'is_stt_logging_enabled', 'is_metrics_logging_enabled',
    'get_tts_output_directory', 'get_stt_input_directory', 'get_metrics_output_directory', 'ensure_logging_directories',
    'get_logging_settings', 'get_allowed_users', 'ALLOWED_LOGGING_USERS', 'LOGGING_SETTINGS'
]