"""Simple configuration package for LiveKit agents."""

# Simple model functions
from .models import get_stt, get_langgraph, get_tts, get_stf

# Persona configuration
from .persona_config import create_instructions

# Voice configuration
from .voices import VoiceSettings, ElevenLabsConfig

# Language configuration
from .language_config import (
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE,
    get_language_name, map_language_to_deepgram, 
    is_supported_language, validate_language
)

# Session setup  
from .session import setup_session

# Logging configuration
from .logging_config import (
    is_user_logging_enabled, is_tts_logging_enabled, is_stt_logging_enabled, is_metrics_logging_enabled,
    get_tts_output_directory, get_stt_input_directory, get_metrics_output_directory, ensure_logging_directories,
    get_logging_settings, get_allowed_users, ALLOWED_LOGGING_USERS, LOGGING_SETTINGS
)

__all__ = [
    # Models
    'get_stt', 'get_langgraph', 'get_tts', 'get_stf',
    
    # Persona 
    'create_instructions',
    
    # Voice
    'VoiceSettings', 'ElevenLabsConfig',
    
    # Language
    'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE',
    'get_language_name', 'map_language_to_deepgram',
    'is_supported_language', 'validate_language',
    
    # Session
    'setup_session',
    
    # Logging
    'is_user_logging_enabled', 'is_tts_logging_enabled', 'is_stt_logging_enabled', 'is_metrics_logging_enabled',
    'get_tts_output_directory', 'get_stt_input_directory', 'get_metrics_output_directory', 'ensure_logging_directories',
    'get_logging_settings', 'get_allowed_users', 'ALLOWED_LOGGING_USERS', 'LOGGING_SETTINGS'
]