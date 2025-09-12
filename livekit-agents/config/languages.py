"""
Language configuration and mapping utilities.
"""

from typing import List

# Supported languages
SUPPORTED_LANGUAGES: List[str] = ["ko", "en", "ja", "zh"]

# Language code to display name mapping
LANGUAGE_NAMES = {
    "ko": "한국어",
    "en": "English", 
    "ja": "日本語",
    "zh": "中文",
}

# Language code to Deepgram STT mapping
DEEPGRAM_LANGUAGE_MAPPING = {
    "ko": "ko",
    "en": "en", 
    "ja": "ja",
    "zh": "zh",
}

# Default language settings
DEFAULT_LANGUAGE = "ko"


def get_language_name(language_code: str) -> str:
    """
    Convert language code to display name.
    
    Args:
        language_code: ISO language code (ko, en, ja, zh)
        
    Returns:
        Display name of the language. Returns Korean name for unsupported languages.
    """
    return LANGUAGE_NAMES.get(language_code, LANGUAGE_NAMES[DEFAULT_LANGUAGE])


def map_language_to_deepgram(language_code: str) -> str:
    """
    Map language code to Deepgram STT language code.
    
    Args:
        language_code: ISO language code (ko, en, ja, zh)
        
    Returns:
        Deepgram-compatible language code. Returns default language for unsupported codes.
    """
    return DEEPGRAM_LANGUAGE_MAPPING.get(language_code, DEFAULT_LANGUAGE)


def is_supported_language(language_code: str) -> bool:
    """
    Check if language code is supported.
    
    Args:
        language_code: ISO language code to check
        
    Returns:
        True if language is supported, False otherwise
    """
    return language_code in SUPPORTED_LANGUAGES


def validate_language(language_code: str) -> str:
    """
    Validate and normalize language code.
    
    Args:
        language_code: Language code to validate
        
    Returns:
        Valid language code. Returns default language if invalid.
    """
    if is_supported_language(language_code):
        return language_code
    return DEFAULT_LANGUAGE