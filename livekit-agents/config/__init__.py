"""Simple configuration package for LiveKit agents."""

# Simple model functions
from .models import get_stt, get_llm, get_tts, get_stf

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

__all__ = [
    # Models
    'get_stt', 'get_llm', 'get_tts', 'get_stf',
    
    # Persona 
    'create_instructions',
    
    # Voice
    'VoiceSettings', 'ElevenLabsConfig',
    
    # Language
    'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE',
    'get_language_name', 'map_language_to_deepgram',
    'is_supported_language', 'validate_language',
    
    # Session
    'setup_session'
]