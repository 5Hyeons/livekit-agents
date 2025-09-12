"""
Logging configuration for selective user TTS and metrics logging.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger("logging-config")

# Allowed users for comprehensive logging
ALLOWED_LOGGING_USERS = [
    # "user-7bf5e21059d2",
    # Add more users as needed
]

# Logging settings
LOGGING_SETTINGS = {
    "enabled": True,
    "base_directory": "./logging",
    
    # TTS Audio Logging
    "tts_logging": {
        "enabled": True,
        "directory": "tts_output",
        "create_user_folders": True,
        "log_text_input": True,
        "save_audio_files": True
    },
    
    # STT Audio Logging
    "stt_logging": {
        "enabled": True,
        "directory": "stt_input",
        "create_user_folders": True,
        "log_transcription_output": True,
        "save_audio_files": True
    },
    
    # Metrics Logging  
    "metrics_logging": {
        "enabled": True,
        "directory": "metrics",
        "create_user_folders": True,
        "format": "json",  # or "txt"
        "include_timestamp": True
    }
}


def is_user_logging_enabled(participant_identity: str) -> bool:
    """
    Check if logging is enabled for a specific user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        bool: True if logging is enabled for this user
    """
    if not LOGGING_SETTINGS["enabled"]:
        return False
    
    return participant_identity in ALLOWED_LOGGING_USERS


def is_tts_logging_enabled(participant_identity: str) -> bool:
    """
    Check if TTS logging is enabled for a specific user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        bool: True if TTS logging is enabled for this user
    """
    if not is_user_logging_enabled(participant_identity):
        return False
    
    return LOGGING_SETTINGS["tts_logging"]["enabled"]


def is_stt_logging_enabled(participant_identity: str) -> bool:
    """
    Check if STT logging is enabled for a specific user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        bool: True if STT logging is enabled for this user
    """
    if not is_user_logging_enabled(participant_identity):
        return False
    
    return LOGGING_SETTINGS["stt_logging"]["enabled"]


def is_metrics_logging_enabled(participant_identity: str) -> bool:
    """
    Check if metrics logging is enabled for a specific user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        bool: True if metrics logging is enabled for this user
    """
    if not is_user_logging_enabled(participant_identity):
        return False
    
    return LOGGING_SETTINGS["metrics_logging"]["enabled"]


def get_tts_output_directory(participant_identity: str) -> str | None:
    """
    Get TTS output directory path for a user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        str | None: Directory path if logging is enabled, None otherwise
    """
    if not is_tts_logging_enabled(participant_identity):
        return None
    
    base_dir = LOGGING_SETTINGS["base_directory"]
    tts_dir = LOGGING_SETTINGS["tts_logging"]["directory"]
    
    if LOGGING_SETTINGS["tts_logging"]["create_user_folders"]:
        # Create user-specific folder
        safe_user_id = participant_identity.replace("/", "_").replace("\\", "_")
        return os.path.join(base_dir, tts_dir, safe_user_id)
    else:
        return os.path.join(base_dir, tts_dir)


def get_stt_input_directory(participant_identity: str) -> str | None:
    """
    Get STT input directory path for a user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        str | None: Directory path if logging is enabled, None otherwise
    """
    if not is_stt_logging_enabled(participant_identity):
        return None
    
    base_dir = LOGGING_SETTINGS["base_directory"]
    stt_dir = LOGGING_SETTINGS["stt_logging"]["directory"]
    
    if LOGGING_SETTINGS["stt_logging"]["create_user_folders"]:
        # Create user-specific folder
        safe_user_id = participant_identity.replace("/", "_").replace("\\", "_")
        return os.path.join(base_dir, stt_dir, safe_user_id)
    else:
        return os.path.join(base_dir, stt_dir)


def get_metrics_output_directory(participant_identity: str) -> str | None:
    """
    Get metrics output directory path for a user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        str | None: Directory path if logging is enabled, None otherwise
    """
    if not is_metrics_logging_enabled(participant_identity):
        return None
    
    base_dir = LOGGING_SETTINGS["base_directory"]
    metrics_dir = LOGGING_SETTINGS["metrics_logging"]["directory"]
    
    if LOGGING_SETTINGS["metrics_logging"]["create_user_folders"]:
        # Create user-specific folder
        safe_user_id = participant_identity.replace("/", "_").replace("\\", "_")
        return os.path.join(base_dir, metrics_dir, safe_user_id)
    else:
        return os.path.join(base_dir, metrics_dir)


def ensure_logging_directories(participant_identity: str) -> Dict[str, str | None]:
    """
    Ensure logging directories exist for a user.
    
    Args:
        participant_identity: User's participant identity
        
    Returns:
        Dict with directory paths (or None if disabled)
    """
    directories = {
        "tts_output": None,
        "stt_input": None,
        "metrics": None
    }
    
    # Create TTS output directory
    tts_dir = get_tts_output_directory(participant_identity)
    if tts_dir:
        try:
            os.makedirs(tts_dir, exist_ok=True)
            directories["tts_output"] = tts_dir
            logger.info(f"TTS logging directory ready: {tts_dir}")
        except Exception as e:
            logger.error(f"Failed to create TTS logging directory {tts_dir}: {e}")
    
    # Create STT input directory
    stt_dir = get_stt_input_directory(participant_identity)
    if stt_dir:
        try:
            os.makedirs(stt_dir, exist_ok=True)
            directories["stt_input"] = stt_dir
            logger.info(f"STT logging directory ready: {stt_dir}")
        except Exception as e:
            logger.error(f"Failed to create STT logging directory {stt_dir}: {e}")
    
    # Create metrics directory
    metrics_dir = get_metrics_output_directory(participant_identity)
    if metrics_dir:
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            directories["metrics"] = metrics_dir
            logger.info(f"Metrics logging directory ready: {metrics_dir}")
        except Exception as e:
            logger.error(f"Failed to create metrics logging directory {metrics_dir}: {e}")
    
    return directories


def get_logging_settings() -> Dict[str, Any]:
    """
    Get complete logging settings.
    
    Returns:
        Dict with all logging settings
    """
    return LOGGING_SETTINGS.copy()


def get_allowed_users() -> list[str]:
    """
    Get list of allowed users for logging.
    
    Returns:
        List of allowed user identities
    """
    return ALLOWED_LOGGING_USERS.copy()